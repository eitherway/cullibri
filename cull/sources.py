"""Where photos come from and where rejects go: Immich (real use) or a plain folder (testing)."""

from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageOps


@dataclass
class Asset:
    id: str
    name: str
    taken: float  # unix seconds
    key: str  # changes when the file content changes; used as cache key


class ImmichSource:
    kind = "immich"

    def __init__(self, url: str, api_key: str, path_prefix: str) -> None:
        self.url = url.rstrip("/")
        self.path_prefix = path_prefix
        self.label = "Immich"
        self.http = httpx.Client(base_url=f"{self.url}/api", headers={"x-api-key": api_key}, timeout=120)

    def choices(self) -> tuple[list[dict], str]:
        """Selectable photo sources and the default one: each external library, plus Immich's own uploads.

        Listing libraries needs an admin API key with library.read. Without it, only the configured path is offered.
        """
        fallback = [{"id": "path", "label": self.path_prefix}]
        try:
            r = self.http.get("/libraries")
            r.raise_for_status()
            libraries = r.json()
        except Exception:
            return fallback, "path"
        out = [{"id": lib["id"], "label": f"{lib['name']} · {', '.join(lib['importPaths'])}"} for lib in libraries]
        out.append({"id": "uploads", "label": "Uploaded to Immich"})
        default = next((lib["id"] for lib in libraries if self.path_prefix in lib["importPaths"]), None)
        if default is None:
            out, default = fallback + out, "path"
        return out, default

    def _matches(self, asset: dict, choice: str) -> bool:
        if choice == "path":
            return asset.get("originalPath", "").startswith(self.path_prefix)
        if choice == "uploads":
            return not asset.get("libraryId")
        return asset.get("libraryId") == choice

    def list(self, after: datetime, before: datetime, choice: str = "path") -> list[Asset]:
        body: dict = {
            "takenAfter": after.isoformat(),
            "takenBefore": before.isoformat(),
            "type": "IMAGE",
            "size": 1000,
        }
        assets: list[Asset] = []
        while True:
            r = self.http.post("/search/metadata", json=body)
            r.raise_for_status()
            page = r.json()["assets"]
            for a in page["items"]:
                if not self._matches(a, choice) or a.get("isTrashed"):
                    continue
                taken = datetime.fromisoformat(a["fileCreatedAt"].replace("Z", "+00:00")).timestamp()
                assets.append(Asset(a["id"], a["originalFileName"], taken, a["checksum"]))
            if page.get("nextCursor"):
                body["cursor"] = page["nextCursor"]
            elif page.get("nextPage"):
                body["page"] = int(page["nextPage"])
            else:
                return assets

    def image(self, asset_id: str, size: str) -> tuple[bytes, str]:
        """size: 'preview' or 'full'. Full falls back to the preview when Immich has no full-size JPEG."""
        sizes = ["fullsize", "preview"] if size == "full" else ["preview"]
        for s in sizes:
            r = self.http.get(f"/assets/{asset_id}/thumbnail", params={"size": s})
            if r.status_code == 200:
                return r.content, r.headers.get("content-type", "image/jpeg")
        r.raise_for_status()
        raise RuntimeError("unreachable")

    def trash(self, ids: list[str]) -> None:
        # force=false moves the assets to Immich's trash; they stay restorable there.
        r = self.http.request("DELETE", "/assets", json={"ids": ids, "force": False})
        r.raise_for_status()


class FolderSource:
    """Test stand-in: reads JPEGs from a folder, 'trash' moves files into a _trash subfolder."""

    kind = "folder"

    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.label = "Folder"

    def _file(self, asset_id: str) -> Path:
        f = (self.path / asset_id).resolve()
        if f.parent != self.path or not f.is_file():
            raise FileNotFoundError(asset_id)
        return f

    def choices(self) -> tuple[list[dict], str]:
        return [{"id": "folder", "label": str(self.path)}], "folder"

    def list(self, after: datetime, before: datetime, choice: str = "folder") -> list[Asset]:
        assets = []
        for f in sorted(self.path.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            taken = _taken(f)
            if after.timestamp() <= taken <= before.timestamp():
                st = f.stat()
                assets.append(Asset(f.name, f.name, taken, f"{f.name}:{st.st_size}:{int(st.st_mtime)}"))
        return assets

    def image(self, asset_id: str, size: str) -> tuple[bytes, str]:
        f = self._file(asset_id)
        if size == "full":
            return f.read_bytes(), "image/jpeg"
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        img.thumbnail((1440, 1440))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"

    def trash(self, ids: list[str]) -> None:
        dest = self.path / "_trash"
        dest.mkdir(exist_ok=True)
        for i in ids:
            shutil.move(self._file(i), dest / i)


def _taken(f: Path) -> float:
    try:
        exif = Image.open(f).getexif()
        raw = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        pass
    m = re.search(r"(\d{8})_(\d{6})", f.name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
    return f.stat().st_mtime
