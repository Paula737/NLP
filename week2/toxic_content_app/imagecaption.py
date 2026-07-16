"""
imagecaption.py
================
Image captioning module using BLIP (Salesforce/blip-image-captioning-base)
from Hugging Face.

This module is imported into main.py and exposes a single function,
generate_caption(), which takes an image (file path, bytes, or PIL Image)
and returns a natural-language caption string.

The model is loaded lazily (only on first use) so importing this module
does not immediately trigger a large download or use GPU/CPU memory
until captioning is actually needed.
"""

import io
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Model name - BLIP-1 base captioning model from Hugging Face.
# Swap for "Salesforce/blip2-opt-2.7b" if you want BLIP-2 (much larger/slower).
MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    """Lazily load the BLIP processor and model (only once)."""
    global _processor, _model
    if _model is None:
        print(f"[imagecaption] Loading {MODEL_NAME} on {_device} ...")
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        print("[imagecaption] Model loaded.")
    return _processor, _model


def _load_image(image_input):
    """
    Accepts:
      - a file path (str)
      - raw image bytes
      - an already-open PIL.Image.Image
    Returns a PIL.Image.Image in RGB mode.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        raise ValueError(
            "image_input must be a file path, bytes, or PIL.Image.Image"
        )
    return img.convert("RGB")


def generate_caption(image_input, max_new_tokens: int = 40) -> str:
    """
    Generate a natural-language caption for the given image.

    Parameters
    ----------
    image_input : str | bytes | PIL.Image.Image
        The image to caption - a file path, raw bytes, or a PIL Image.
    max_new_tokens : int
        Maximum length of the generated caption.

    Returns
    -------
    str
        The generated caption text.
    """
    processor, model = _load_model()
    image = _load_image(image_input)

    inputs = processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


if __name__ == "__main__":
    # Quick manual test:
    #   python imagecaption.py path/to/image.jpg
    import sys
    if len(sys.argv) > 1:
        cap = generate_caption(sys.argv[1])
        print("Caption:", cap)
    else:
        print("Usage: python imagecaption.py <path_to_image>")