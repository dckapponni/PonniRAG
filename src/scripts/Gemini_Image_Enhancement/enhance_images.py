"""
Image Quality Enhancement using Google Gemini (google-genai SDK).

This module provides an automated pipeline for enhancing the quality of images
using Google's Gemini generative AI models. It supports both general images and
text-heavy images, applying appropriate enhancement prompts for each type.

Workflow:
    1. Load images from the ``input_images/`` directory.
    2. Auto-detect the best available Gemini image-generation model.
    3. Classify each image as ``"text"`` or ``"image"`` to select the right prompt.
    4. Send the image to the Gemini API and receive an enhanced version.
    5. Save each enhanced image as a PNG to the ``enhanced_images/`` directory.

Supported input formats:
    JPEG, PNG, BMP, WebP, TIFF

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY: Required. Your Google Gemini API key,
    loaded automatically from a ``.env`` file if present.

Dependencies:
    - google-genai  (``pip install google-genai``)
    - Pillow        (``pip install Pillow``)
    - python-dotenv (``pip install python-dotenv``)

Example:
    Place images in ``input_images/`` and run::

        python enhance_images.py

    Enhanced outputs will appear in ``enhanced_images/`` as
    ``<original_stem>_enhanced.png``.
"""

import io
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit("  Set GEMINI_API_KEY in your .env file.")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit(" Run:  pip install google-genai")

client = genai.Client(api_key=API_KEY)

# Ordered list of candidate model IDs to probe for image-generation support.
# The first model that successfully returns an image will be used for all
# subsequent enhancement calls.
IMAGE_GEN_CANDIDATES = [
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
]

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("enhanced_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PROMPT_IMAGE = """Enhance the quality of this cropped image.
Instructions:
- Improve image sharpness and clarity.
- Remove blur, noise, and compression artifacts.
- Preserve the original content and structure.
- Improve lighting and readability.
- Keep the image natural and clean.
- Generate a high-resolution enhanced version.
- Use a clean white background if applicable.
- Do not change the actual objects or drawing content.
- Return only the enhanced image."""

PROMPT_TEXT = """Enhance this cropped text image while preserving original typography.
Instructions:
- Improve text clarity and readability.
- Keep the original font style unchanged.
- Preserve the exact text content.
- Remove noise, blur, shadows, and unwanted marks.
- Place the content on a pure white background.
- Align the text neatly.
- Do not modify the wording, spacing, or font appearance.
- Generate a clean high-resolution version suitable for layouts and publishing.
- Return only the enhanced image."""


def classify_image(pil_img: Image.Image) -> str:
    """Classify a PIL image as either ``"text"`` or ``"image"``.

    Uses a heuristic based on the proportion of near-white pixels and the
    number of distinct colour clusters in a 128×128 thumbnail. Images with a
    large white area and low colour diversity are treated as text scans or
    diagrams; everything else is treated as a photographic image.

    Args:
        pil_img: An opened PIL ``Image`` object to classify.

    Returns:
        ``"text"`` if the image appears to be a text or diagram scan,
        ``"image"`` otherwise.
    """
    try:
        small = pil_img.convert("RGB").resize((128, 128))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pixels = list(small.getdata())
        white = sum(1 for r, g, b in pixels if r > 200 and g > 200 and b > 200)
        div = len({(r >> 4, g >> 4, b >> 4) for r, g, b in pixels})
        return "text" if (white / len(pixels) > 0.55 and div < 60) else "image"
    except Exception:
        return "image"


def to_png_bytes(pil_img: Image.Image) -> bytes:
    """Encode a PIL image as raw PNG bytes.

    The image is first converted to RGB (stripping any alpha channel) before
    encoding, ensuring the output is a plain 24-bit PNG compatible with the
    Gemini API.

    Args:
        pil_img: The source ``Image`` object to encode.

    Returns:
        A ``bytes`` object containing the PNG-encoded image data.
    """
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def save_white_bg(img: Image.Image, path: Path) -> None:
    """Composite a PIL image onto a white background and save it as PNG.

    If the image has an alpha channel (mode ``RGBA`` or ``LA``), it is
    flattened against a solid white canvas before saving. Images in other
    non-RGB modes are converted to ``RGB``. The result is always saved as an
    optimised PNG.

    Args:
        img:  The ``Image`` object to save.
        path: Destination file path (should end in ``.png``).
    """
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, "PNG", optimize=True)


def call_gemini(model: str, img_bytes: bytes, prompt: str) -> bytes | None:
    """Send an image and a text prompt to a Gemini model and return image bytes.

    Constructs a multimodal request containing the PNG image and the
    enhancement prompt, then parses the first ``inline_data`` part from the
    model's response. If the model returns text instead of an image (e.g.
    because the prompt was declined or the model lacks image-output support),
    the text is logged as a warning and ``None`` is returned.

    Args:
        model:     The Gemini model identifier string (e.g.
                   ``"gemini-3.1-flash-image-preview"``).
        img_bytes: Raw PNG bytes of the image to enhance.
        prompt:    The natural-language enhancement instruction.

    Returns:
        Raw image bytes returned by the model, or ``None`` if no image part
        was found in the response.
    """
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            return part.inline_data.data
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            log.warning("  Model returned text instead of image: %s", part.text[:200])
    return None


def probe_models(candidates: list) -> str | None:
    """Find the first model in *candidates* that can generate images.

    Sends a tiny 32×32 white PNG to each candidate model in order and returns
    the identifier of the first model that responds with an image. This probing
    step avoids hard-coding a single model name and gracefully handles API
    rollouts where a model may not yet be available in a given region or tier.

    Args:
        candidates: Ordered list of Gemini model identifier strings to test.

    Returns:
        The identifier of the first working model, or ``None`` if every
        candidate fails or returns no image.
    """
    log.info("Probing models for image-generation capability…")
    tiny = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 255, 255)).save(tiny, format="PNG")
    tiny_bytes = tiny.getvalue()
    for model in candidates:
        try:
            log.info("  Testing: %s", model)
            raw = call_gemini(model, tiny_bytes, "Return this image unchanged.")
            if raw:
                log.info("  ✓ '%s' confirmed — using this model.", model)
                return model
            log.warning("  ✗ '%s' returned no image.", model)
        except Exception as e:
            log.warning("  ✗ '%s' error: %s", model, str(e)[:120])
    return None


FAIL_LOG: list[str] = []  # tracks all failed files


def enhance_single(model: str, img_path: Path, retries: int = 3) -> bool:
    """Enhance a single image file and write the result to ``OUTPUT_DIR``.

    Opens the image, classifies it as text or photographic, selects the
    appropriate Gemini prompt, and calls the API with automatic retry logic.
    Rate-limit (HTTP 429) and server-error (HTTP 500/503) responses trigger
    progressively longer back-off delays. Already-enhanced files (where an
    output PNG already exists) are skipped without an API call.

    On failure, the filename and reason are appended to the module-level
    ``FAIL_LOG`` list so a summary can be printed and saved at the end of a
    batch run.

    Args:
        model:   The Gemini model identifier to use for enhancement.
        img_path: Path to the source image file.
        retries: Maximum number of API call attempts before giving up.
                 Defaults to ``3``.

    Returns:
        ``True`` if the image was successfully enhanced and saved (or was
        already present), ``False`` if all attempts failed.
    """
    try:
        pil_img = Image.open(img_path)
    except Exception as e:
        log.error("  Cannot open file: %s", e)
        FAIL_LOG.append(f"{img_path.name}  →  cannot open: {e}")
        return False

    out_path = OUTPUT_DIR / f"{img_path.stem}_enhanced.png"
    if out_path.exists():
        log.info("  ⏭  Already enhanced, skipping.")
        return True

    ctype = classify_image(pil_img)
    prompt = PROMPT_TEXT if ctype == "text" else PROMPT_IMAGE
    img_bytes = to_png_bytes(pil_img)
    log.info("  → type: %s  |  size: %dx%d", ctype, pil_img.width, pil_img.height)

    for attempt in range(1, retries + 1):
        try:
            raw = call_gemini(model, img_bytes, prompt)
            if raw:
                enhanced = Image.open(io.BytesIO(raw))
                save_white_bg(enhanced, out_path)
                log.info(
                    "  ✓ Saved → %s  (%dx%d)",
                    out_path.name,
                    enhanced.width,
                    enhanced.height,
                )
                return True

            log.warning(
                "  Attempt %d/%d: no image returned, retrying…", attempt, retries
            )
            time.sleep(2 * attempt)

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 20 * attempt
                log.warning("  Rate limited – waiting %ds before retry…", wait)
                time.sleep(wait)
            elif "500" in err or "503" in err:
                wait = 10 * attempt
                log.warning("  Server error – waiting %ds before retry…", wait)
                time.sleep(wait)
            else:
                log.warning("  Attempt %d/%d error: %s", attempt, retries, err[:150])
                time.sleep(3 * attempt)

    reason = f"no image after {retries} attempts"
    FAIL_LOG.append(f"{img_path.name}  →  {reason}")
    log.error("  ✗ Failed: %s", reason)
    return False


def main() -> None:
    """Run the full image-enhancement batch pipeline.

    Steps performed:

    1. Validate that ``INPUT_DIR`` exists and contains at least one supported
       image file.
    2. Probe ``IMAGE_GEN_CANDIDATES`` to find a working Gemini model.
    3. Iterate over every input image, skipping files that already have a
       corresponding output in ``OUTPUT_DIR``.
    4. Call :func:`enhance_single` for each remaining file, sleeping 1.5 s
       between requests to stay within typical rate limits.
    5. Print a summary of enhanced, skipped, and failed counts. If any files
       failed, write the details to ``enhanced_images/_failed.txt``.

    Exits with a non-zero status via ``sys.exit`` if ``INPUT_DIR`` is missing,
    contains no supported images, or no working model can be found.
    """
    if not INPUT_DIR.exists():
        sys.exit(f"  Folder '{INPUT_DIR}' not found.")

    images = sorted(
        p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not images:
        sys.exit(f"  No images found in '{INPUT_DIR}'.")

    model = probe_models(IMAGE_GEN_CANDIDATES)
    if not model:
        sys.exit("  No working image-generation model found.")

    log.info("=" * 60)
    log.info("Images : %d  |  Model : %s", len(images), model)
    log.info("Output : %s", OUTPUT_DIR.resolve())
    log.info("=" * 60)

    success = failed = skipped = 0
    for i, p in enumerate(images, 1):
        log.info("[%d/%d] %s", i, len(images), p.name)

        out_path = OUTPUT_DIR / f"{p.stem}_enhanced.png"
        if out_path.exists():
            log.info("  ⏭  Already done, skipping.")
            skipped += 1
            continue

        if enhance_single(model, p):
            success += 1
        else:
            failed += 1

        if i < len(images):
            time.sleep(1.5)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("   Enhanced : %d", success)
    log.info("   Skipped  : %d  (already existed)", skipped)
    log.info("   Failed   : %d", failed)

    if FAIL_LOG:
        log.info("")
        log.info("Failed files:")
        for entry in FAIL_LOG:
            log.info("  • %s", entry)
        fail_file = OUTPUT_DIR / "_failed.txt"
        fail_file.write_text("\n".join(FAIL_LOG))
        log.info("")
        log.info("Failed list saved → %s", fail_file)

    log.info("=" * 60)


if __name__ == "__main__":
    main()
