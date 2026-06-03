"""
Image Quality Enhancement using Google Gemini (google-genai SDK).

This version:
- Recursively scans all folders inside INPUT_DIR
- Preserves the same folder structure in OUTPUT_DIR
- Enhances all supported images
- Skips already processed files
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

# ============================================================
# ENV
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    sys.exit("Set GEMINI_API_KEY in .env")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Run: pip install google-genai")

client = genai.Client(api_key=API_KEY)

# ============================================================
# CONFIG
# ============================================================

IMAGE_GEN_CANDIDATES = ["gemini-3.1-flash-image"]

SUPPORTED = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

# INPUT ROOT FOLDER
INPUT_DIR = Path(
    "/home/ubuntu/Ponni_Rag/Tagging_feature/src/scripts/Gemini_Image_Enhancement/input"
)

# OUTPUT ROOT FOLDER
OUTPUT_DIR = Path(
    "/home/ubuntu/Ponni_Rag/Tagging_feature/src/scripts/Gemini_Image_Enhancement/enhanced_image"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROMPTS
# ============================================================
# ============================================================
# PROMPTS
# ============================================================

PROMPT_IMAGE = """You are an image enhancement tool. Your ONLY job is to improve visual quality.

STRICT RULES — NEVER VIOLATE:
- DO NOT add, remove, or alter any text, letters, words, or characters in the image.
- DO NOT translate, correct, or rewrite any text you see.
- DO NOT add new objects, shapes, or elements.
- DO NOT change the background colour or scene composition.
- DO NOT change the colours of existing objects.

ALLOWED enhancements ONLY:
- Increase sharpness and clarity.
- Reduce blur, noise, grain, and compression artifacts.
- Improve brightness and contrast for better readability.
- Produce a high-resolution clean output.

Return ONLY the enhanced image. No text. No explanation."""


PROMPT_TEXT = """You are an image cleaning tool. Your ONLY job is to clean and sharpen this text image.

STRICT RULES — NEVER VIOLATE:
- DO NOT change, correct, translate, or rewrite any word, letter, character, or punctuation.
- DO NOT add any new text or remove any existing text.
- Every character in the output must be pixel-for-pixel faithful to the input text.
- If you cannot guarantee the text is unchanged, return the image as-is with only background cleaning.

ALLOWED enhancements ONLY:
- Replace the background with a pure solid white (#FFFFFF).
- Remove stains, shadows, noise, blur, and smudges from the background.
- Sharpen and darken the existing text strokes for better legibility.
- Preserve the exact font style, size, weight, and spacing.
- Preserve the exact original layout and line breaks.

Return ONLY the enhanced image. No text. No explanation."""


PROMPT_IMAGE_TEXT = """You are an image enhancement tool for images that contain both visuals and text.

STRICT RULES — NEVER VIOLATE:
- DO NOT change, correct, translate, or rewrite any word, letter, character, or punctuation visible in the image.
- DO NOT add any new text or symbols.
- DO NOT remove any existing text.
- DO NOT change the colours, shapes, or positions of any objects.
- DO NOT alter the layout or composition of the image.
- If in doubt about any text character, preserve it exactly as it appears.

ALLOWED enhancements ONLY:
- Replace the background with a pure solid white (#FFFFFF).
- Remove noise, blur, stains, shadows, and compression artifacts.
- Improve sharpness and contrast of both text and visual elements.
- Increase overall resolution and clarity.

Return ONLY the enhanced image. No text. No explanation."""

# ============================================================
# HELPERS
# ============================================================


def classify_image(pil_img: Image.Image) -> str:
    try:
        small = pil_img.convert("RGB").resize((128, 128))

        pixels = list(small.getdata())

        white = sum(1 for r, g, b in pixels if r > 200 and g > 200 and b > 200)

        white_ratio = white / len(pixels)

        div = len({(r >> 4, g >> 4, b >> 4) for r, g, b in pixels})

        if white_ratio > 0.55 and div < 60:
            return "text"

        elif white_ratio > 0.25 and div < 120:
            return "image_text"

        return "image"

    except Exception:
        return "image"


def to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()

    pil_img.convert("RGB").save(buf, format="PNG")

    return buf.getvalue()


def save_white_bg(img: Image.Image, path: Path) -> None:

    if img.mode in ("RGBA", "LA"):

        bg = Image.new("RGB", img.size, (255, 255, 255))

        bg.paste(img, mask=img.split()[-1])

        img = bg

    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.save(path, "PNG", optimize=True)


def call_gemini(model: str, img_bytes: bytes, prompt: str):

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/png",
            ),
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:

        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            return part.inline_data.data

    return None


def probe_models(candidates):

    log.info("Probing models...")

    tiny = io.BytesIO()

    Image.new(
        "RGB",
        (32, 32),
        (255, 255, 255),
    ).save(tiny, format="PNG")

    tiny_bytes = tiny.getvalue()

    for model in candidates:

        try:
            log.info("Testing: %s", model)

            raw = call_gemini(
                model,
                tiny_bytes,
                "Return this image unchanged.",
            )

            if raw:
                log.info("Using model: %s", model)
                return model

        except Exception as e:
            log.warning("%s failed: %s", model, str(e)[:120])

    return None


FAIL_LOG = []

# ============================================================
# ENHANCE SINGLE IMAGE
# ============================================================


def enhance_single(model: str, img_path: Path, retries: int = 3):

    try:
        pil_img = Image.open(img_path)

    except Exception as e:

        FAIL_LOG.append(f"{img_path} -> cannot open: {e}")

        return False

    # Preserve folder structure
    relative_path = img_path.relative_to(INPUT_DIR)

    out_path = OUTPUT_DIR / relative_path.parent / f"{img_path.stem}_enhanced.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log.info("Already exists -> skipping")
        return True

    # classify
    ctype = classify_image(pil_img)

    if ctype == "text":
        prompt = PROMPT_TEXT

    elif ctype == "image_text":
        prompt = PROMPT_IMAGE_TEXT

    else:
        prompt = PROMPT_IMAGE

    img_bytes = to_png_bytes(pil_img)

    log.info(
        "Type: %s | Size: %dx%d",
        ctype,
        pil_img.width,
        pil_img.height,
    )

    for attempt in range(1, retries + 1):

        try:

            raw = call_gemini(model, img_bytes, prompt)

            if raw:

                enhanced = Image.open(io.BytesIO(raw))

                save_white_bg(enhanced, out_path)

                log.info("Saved -> %s", out_path)

                return True

            log.warning(
                "No image returned (%d/%d)",
                attempt,
                retries,
            )

            time.sleep(2 * attempt)

        except Exception as e:

            err = str(e)

            if "429" in err:

                wait = 20 * attempt

                log.warning("Rate limit -> waiting %ds", wait)

                time.sleep(wait)

            else:

                log.warning(
                    "Attempt %d failed: %s",
                    attempt,
                    err[:150],
                )

                time.sleep(3 * attempt)

    FAIL_LOG.append(f"{img_path} -> failed after retries")

    return False


# ============================================================
# MAIN
# ============================================================


def main():

    if not INPUT_DIR.exists():
        sys.exit(f"Folder not found: {INPUT_DIR}")

    # RECURSIVE SEARCH
    images = sorted(
        p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not images:
        sys.exit("No images found")

    model = probe_models(IMAGE_GEN_CANDIDATES)

    if not model:
        sys.exit("No working Gemini image model found")

    log.info("=" * 60)
    log.info("Total Images : %d", len(images))
    log.info("Output Folder: %s", OUTPUT_DIR)
    log.info("=" * 60)

    success = 0
    failed = 0
    skipped = 0

    for idx, img_path in enumerate(images, 1):

        relative_path = img_path.relative_to(INPUT_DIR)

        log.info(
            "[%d/%d] %s",
            idx,
            len(images),
            relative_path,
        )

        out_path = OUTPUT_DIR / relative_path.parent / f"{img_path.stem}_enhanced.png"

        if out_path.exists():

            skipped += 1

            log.info("Skipping existing")

            continue

        if enhance_single(model, img_path):
            success += 1
        else:
            failed += 1

        time.sleep(1.5)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("Enhanced : %d", success)
    log.info("Skipped  : %d", skipped)
    log.info("Failed   : %d", failed)

    if FAIL_LOG:

        fail_file = OUTPUT_DIR / "_failed.txt"

        fail_file.write_text("\n".join(FAIL_LOG))

        log.info("Failed list saved -> %s", fail_file)

    log.info("=" * 60)


if __name__ == "__main__":
    main()
