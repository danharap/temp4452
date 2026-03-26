from io import BytesIO
from typing import Optional

from PIL import Image


def make_degradation_config(mode: str, quality: int = 60, scale: float = 0.5) -> dict:
    mode = mode.lower()
    if mode not in {"jpeg", "resize"}:
        raise ValueError("mode must be 'jpeg' or 'resize'")
    return {"mode": mode, "quality": int(quality), "scale": float(scale)}


def apply_degradation_pil(image: Image.Image, degradation: Optional[dict]) -> Image.Image:
    if degradation is None:
        return image

    mode = degradation.get("mode", "").lower()
    if mode == "jpeg":
        quality = int(degradation.get("quality", 60))
        return apply_jpeg_compression(image, quality=quality)

    if mode == "resize":
        scale = float(degradation.get("scale", 0.5))
        return apply_resize_degradation(image, scale=scale)

    raise ValueError(f"Unsupported degradation mode: {mode}")


def apply_jpeg_compression(image: Image.Image, quality: int = 60) -> Image.Image:
    rgb = image.convert("RGB")
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_resize_degradation(image: Image.Image, scale: float = 0.5) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    small_w = max(8, int(w * scale))
    small_h = max(8, int(h * scale))
    smaller = rgb.resize((small_w, small_h), Image.BILINEAR)
    restored = smaller.resize((w, h), Image.BILINEAR)
    return restored
