"""Optional written verdict from a local vision-language model.

Apple Silicon runs it through MLX. Everything else runs it through transformers: 4-bit on an NVIDIA GPU,
and a smaller model on the CPU when there is none.
"""

from __future__ import annotations

import io
import json
import platform
import re
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

USE_MLX = sys.platform == "darwin" and platform.machine() == "arm64"
MLX_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
CUDA_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # loaded in 4-bit, about 7 GB of GPU memory
CPU_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # the 8B model is too slow without a GPU
MAX_PHOTOS = 15
SIDE = 768
MAX_TOKENS = 500

_state: dict = {}
_lock = threading.Lock()

PROMPT = """These are {n} photos of the same subject, numbered 1 to {n} in the order given. The photographer wants to keep only the best one.
Compare framing, composition, distracting objects (passers-by, vehicles, cut-off elements), light, and closed eyes or awkward expressions.
Answer with JSON only:
{{"best": <number>, "reason": "<one sentence on why it wins>", "notes": {{"<number>": "<max 8 words on this photo's main strength or flaw>", ...}}}}"""


def _generate_mlx(images: list[Image.Image], prompt: str) -> str:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    if not _state:
        _state["model"], _state["processor"] = load(MLX_MODEL)
    model, processor = _state["model"], _state["processor"]

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(images):
            p = Path(tmp) / f"{i + 1}.jpg"
            img.save(p, "JPEG", quality=90)
            paths.append(str(p))
        chat = apply_chat_template(processor, model.config, prompt, num_images=len(paths))
        return generate(model, processor, chat, image=paths, max_tokens=MAX_TOKENS, temperature=0.0).text


def _generate_transformers(images: list[Image.Image], prompt: str) -> str:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not _state:
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            name = CUDA_MODEL
            model = AutoModelForImageTextToText.from_pretrained(name, quantization_config=quantization, device_map="auto")
        else:
            name = CPU_MODEL
            model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float32)
        _state["model"], _state["processor"] = model.eval(), AutoProcessor.from_pretrained(name)
    model, processor = _state["model"], _state["processor"]

    content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    return processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def explain(images: list[bytes]) -> dict:
    """Returns {'best': index, 'reason': str, 'notes': [str per photo]} for the first MAX_PHOTOS images."""
    photos = []
    for data in images[:MAX_PHOTOS]:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        img.thumbnail((SIDE, SIDE))
        photos.append(img)

    with _lock:
        generate = _generate_mlx if USE_MLX else _generate_transformers
        text = generate(photos, PROMPT.format(n=len(photos)))

    try:
        raw = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        best = int(raw["best"]) - 1
        if not 0 <= best < len(photos):
            raise ValueError("pick out of range")
        notes = [str(raw.get("notes", {}).get(str(i + 1), "")) for i in range(len(photos))]
        return {"best": best, "reason": str(raw.get("reason", "")), "notes": notes}
    except Exception:
        return {"best": None, "reason": text.strip()[:400], "notes": [""] * len(photos)}
