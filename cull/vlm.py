"""Optional written verdict from a local vision-language model (runs on this Mac via MLX)."""

from __future__ import annotations

import io
import json
import re
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
MAX_PHOTOS = 10
SIDE = 768

_state: dict = {}
_lock = threading.Lock()

PROMPT = """These are {n} photos of the same subject, numbered 1 to {n} in the order given. The photographer wants to keep only the best one.
Compare framing, composition, distracting objects (passers-by, vehicles, cut-off elements), light, and closed eyes or awkward expressions.
Answer with JSON only:
{{"best": <number>, "reason": "<one sentence on why it wins>", "notes": {{"<number>": "<max 8 words on this photo's main strength or flaw>", ...}}}}"""


def explain(images: list[bytes]) -> dict:
    """Returns {'best': index, 'reason': str, 'notes': [str per photo]} for the first MAX_PHOTOS images."""
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    images = images[:MAX_PHOTOS]
    with _lock, tempfile.TemporaryDirectory() as tmp:
        if not _state:
            _state["model"], _state["processor"] = load(MODEL)
        model, processor = _state["model"], _state["processor"]

        paths = []
        for i, data in enumerate(images):
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            img.thumbnail((SIDE, SIDE))
            p = Path(tmp) / f"{i + 1}.jpg"
            img.save(p, "JPEG", quality=90)
            paths.append(str(p))

        prompt = apply_chat_template(processor, model.config, PROMPT.format(n=len(paths)), num_images=len(paths))
        text = generate(model, processor, prompt, image=paths, max_tokens=500, temperature=0.0).text

    try:
        raw = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        best = int(raw["best"]) - 1
        if not 0 <= best < len(images):
            raise ValueError("pick out of range")
        notes = [str(raw.get("notes", {}).get(str(i + 1), "")) for i in range(len(images))]
        return {"best": best, "reason": str(raw.get("reason", "")), "notes": notes}
    except Exception:
        return {"best": None, "reason": text.strip()[:400], "notes": [""] * len(images)}
