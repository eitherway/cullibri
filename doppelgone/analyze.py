"""Per-photo analysis, all local: similarity embedding, sharpness, faces, exposure."""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

MODELS = Path(__file__).resolve().parent.parent / "models"
# MediaPipe Face Landmarker (float16, version 1) by Google, Apache License 2.0.
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
VERSION = 2  # bump to invalidate cached analyses
WORK_SIDE = 2048  # every photo is measured at this longest side so numbers are comparable


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def face_model_path() -> Path:
    """Path of the face landmarker model, downloaded from Google on first use and checked against its checksum."""
    path = MODELS / "face_landmarker.task"
    if path.exists():
        return path
    print(f"Downloading face landmarker model from {FACE_MODEL_URL}")
    with urllib.request.urlopen(FACE_MODEL_URL, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FACE_MODEL_SHA256:
        raise RuntimeError(f"Face landmarker download has checksum {digest}, expected {FACE_MODEL_SHA256}")
    MODELS.mkdir(exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(path)
    return path


def _gradient_energy(gray: np.ndarray) -> np.ndarray:
    """Tenengrad focus measure (squared Sobel gradient) per pixel, after light smoothing.

    The smoothing makes it ignore sensor noise: on the test photos, added noise raised a Laplacian-variance
    score by 20-95% (a blurred, noisy frame could score like a sharp one) but this measure by at most 2%.
    """
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def _tile_sharpness(energy: np.ndarray, grid: int = 8) -> float:
    """Mean gradient energy of the sharpest tiles, so a blurred background does not count against a sharp subject."""
    h, w = energy.shape
    th, tw = max(h // grid, 1), max(w // grid, 1)
    scores = [
        float(energy[y : y + th, x : x + tw].mean())
        for y in range(0, th * grid, th)
        for x in range(0, tw * grid, tw)
    ]
    scores.sort(reverse=True)
    top = scores[: max(len(scores) // 8, 1)]
    return float(np.mean(top))


class Analyzer:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # DINOv2 embeddings decide which photos show the same motif.
        from transformers import AutoImageProcessor, AutoModel

        self.dino_pre = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

        self._mp = mp
        self.faces = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(face_model_path())),
                output_face_blendshapes=True,
                num_faces=10,
            )
        )

    @torch.no_grad()
    def analyze(self, data: bytes) -> tuple[dict, np.ndarray]:
        img = load_image(data)
        img.thumbnail((WORK_SIDE, WORK_SIDE), Image.LANCZOS)
        rgb = np.asarray(img)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        energy = _gradient_energy(gray)

        faces = []
        result = self.faces.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        for landmarks, shapes in zip(result.face_landmarks, result.face_blendshapes or []):
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            x0, x1 = int(max(min(xs), 0) * w), int(min(max(xs), 1) * w)
            y0, y1 = int(max(min(ys), 0) * h), int(min(max(ys), 1) * h)
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            blink = {s.category_name: s.score for s in shapes if s.category_name.startswith("eyeBlink")}
            faces.append(
                {
                    "box": [x0 / w, y0 / h, x1 / w, y1 / h],
                    "sharpness": float(energy[y0:y1, x0:x1].mean()),
                    "blink": float(min(blink.values())) if blink else 0.0,
                }
            )

        metrics = {
            "sharpness": _tile_sharpness(energy),
            "blown": float((gray >= 250).mean()),
            "crushed": float((gray <= 5).mean()),
            "faces": faces,
        }
        small = img.copy()
        small.thumbnail((1024, 1024))
        cls = self.dino(**self.dino_pre(images=small, return_tensors="pt").to(self.device)).last_hidden_state[0, 0]
        cls = cls / cls.norm()
        return metrics, cls.cpu().numpy().astype(np.float32)
