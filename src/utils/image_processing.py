from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def image_caption_stub(path: Path) -> str:
    """Fallback text for retrieval when no vision LLM is configured."""
    return f"Image document: {path.name}"
