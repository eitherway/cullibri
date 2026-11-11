"""Local web server: scan a date range, show groups, move rejects to trash."""

from __future__ import annotations

import threading
import tomllib
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline, vlm
from .sources import FolderSource, ImmichSource

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "cull" / "static"), name="static")
source = None
job: dict = {"state": "idle", "done": 0, "total": 0, "groups": [], "error": None, "cancel": False}


class Cancelled(Exception):
    pass


def configure(folder: str | None) -> None:
    global source
    if folder:
        source = FolderSource(folder)
        return
    cfg_file = ROOT / "config.toml"
    if not cfg_file.exists():
        raise SystemExit("config.toml not found. Copy config.example.toml to config.toml and add your Immich API key.")
    cfg = tomllib.loads(cfg_file.read_text())["immich"]
    source = ImmichSource(cfg["url"], cfg["api_key"], cfg["path_prefix"])


class ScanRequest(BaseModel):
    after: datetime
    before: datetime
    source: str


class Ids(BaseModel):
    ids: list[str]


def _run(req: ScanRequest) -> None:
    def progress(done: int, total: int) -> None:
        if job["cancel"]:
            raise Cancelled
        job.update(done=done, total=total)

    try:
        job["groups"] = pipeline.scan(source, req.after, req.before, req.source, progress)
        job["state"] = "done"
    except Cancelled:  # photos analysed so far stay cached
        job["state"] = "idle"
    except Exception as e:  # shown in the UI
        job.update(state="error", error=f"{type(e).__name__}: {e}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "cull" / "static" / "index.html")


@app.get("/api/info")
def info() -> dict:
    choices, default = source.choices()
    return {"kind": source.label, "sources": choices, "default": default}


@app.post("/api/scan")
def start_scan(req: ScanRequest) -> dict:
    if job["state"] == "running":
        raise HTTPException(409, "A scan is already running")
    if req.source not in {c["id"] for c in source.choices()[0]}:
        raise HTTPException(400, "Unknown photo source")
    job.update(state="running", done=0, total=0, groups=[], error=None, cancel=False)
    threading.Thread(target=_run, args=(req,), daemon=True).start()
    return {"ok": True}


@app.post("/api/scan/cancel")
def cancel_scan() -> dict:
    job["cancel"] = True  # the scan stops before its next photo
    return {"ok": True}


@app.get("/api/scan")
def scan_status() -> dict:
    return job


@app.get("/api/img/{asset_id}")
def image(asset_id: str, size: str = "preview") -> Response:
    try:
        data, mime = source.image(asset_id, "full" if size == "full" else "preview")
    except FileNotFoundError:
        raise HTTPException(404)
    return Response(data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/trash")
def trash(req: Ids) -> dict:
    known = {p["id"] for g in job["groups"] for p in g}
    if not req.ids or not set(req.ids) <= known:
        raise HTTPException(400, "Only photos from the current scan can be trashed")
    source.trash(req.ids)
    gone = set(req.ids)
    job["groups"] = [[p for p in g if p["id"] not in gone] for g in job["groups"]]
    return {"trashed": len(req.ids)}


@app.post("/api/explain")
def explain(req: Ids) -> dict:
    return vlm.explain([source.image(i, "preview")[0] for i in req.ids])
