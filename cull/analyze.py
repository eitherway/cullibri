"""Per-photo analysis with local models: similarity embedding, aesthetic score, sharpness, faces, exposure."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

MODELS = Path(__file__).resolve().parent.parent / "models"
VERSION = 2  # bump to invalidate cached analyses
WORK_SIDE = 2048  # every photo is measured at this longest side so numbers are comparable


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


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
        import open_clip
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.clip = self.clip.to(self.device).eval()

        # LAION improved aesthetic predictor: an MLP on top of the CLIP ViT-L/14 embedding.
        self.aesthetic = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
        )
        state = torch.load(MODELS / "laion_aesthetic_l14.pth", map_location="cpu")
        self.aesthetic.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()})
        self.aesthetic = self.aesthetic.to(self.device).eval()

        # DINOv2 separates "same motif" from "similar kind of scene" far better than CLIP does.
        from transformers import AutoImageProcessor, AutoModel

        self.dino_pre = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

        self._mp = mp
        self.faces = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
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

        emb = self.clip.encode_image(self.preprocess(img).unsqueeze(0).to(self.device))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        aesthetic = float(self.aesthetic(emb.float()).item())

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
            "aesthetic": aesthetic,
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
