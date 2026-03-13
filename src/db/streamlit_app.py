import base64
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
import streamlit as st
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import get_magazine_config

# Load magazine registry (single source of truth)
_magazine = get_magazine_config("ponni")
_s3_conf = _magazine["s3"]

# Build PDF_LINKS dict from registry for backward compatibility
PDF_LINKS = {}
for _vol in _magazine["volumes"]:
    for _iss in _vol["issues"]:
        PDF_LINKS[f"vol_{_vol['id']}_issue_{_iss['num']}"] = _iss["pdf_url"]

# S3 client for direct image fetching
try:
    _s3_client = boto3.client("s3", region_name=_s3_conf["region"])
except Exception:
    _s3_client = None

# In-memory cache: S3 key -> PIL Image (already thumbnailed)
_s3_image_cache: Dict[str, Image.Image] = {}


def _s3_key_with_fallback(s3_key: str) -> List[str]:
    """Return list of S3 keys to try: original + alternate extensions (.jpg/.png/.jpeg)."""
    base, ext = s3_key.rsplit(".", 1) if "." in s3_key else (s3_key, "")
    alternates = ["jpg", "png", "jpeg"]
    keys = [s3_key]
    for alt in alternates:
        if alt != ext.lower():
            keys.append(f"{base}.{alt}")
    return keys


def _load_s3_image(s3_key: str, thumbnail_size: tuple = (250, 375)) -> Optional[Image.Image]:
    """Fetch image from S3, thumbnail it, cache, and return PIL Image. Tries alternate extensions on failure."""
    if s3_key in _s3_image_cache:
        return _s3_image_cache[s3_key]
    if not _s3_client:
        logging.warning("S3 client not available")
        return None
    for key in _s3_key_with_fallback(s3_key):
        if key in _s3_image_cache:
            return _s3_image_cache[key]
        try:
            resp = _s3_client.get_object(Bucket=_s3_conf["bucket"], Key=key)
            raw = resp["Body"].read()
            img = Image.open(io.BytesIO(raw))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            _s3_image_cache[s3_key] = img
            logging.info("S3 image loaded: %s (%d bytes -> %dx%d)", key, len(raw), img.width, img.height)
            return img
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                continue
            logging.warning("S3 fetch failed %s: %s", key, e)
        except (NoCredentialsError, BotoCoreError) as e:
            logging.warning("S3 fetch failed %s: %s", key, e)
            break
        except Exception as e:
            logging.warning("S3 image error %s: %s", key, e)
    return None


def _pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 string for embedding in HTML."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()


def _volume_cover_s3_key(volume_id: int) -> Optional[str]:
    """Derive S3 key for a volume cover image."""
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data or not vol_data.get("cover_image"):
        return None
    return f"{_s3_conf['covers_prefix']}Volumes/{vol_data['cover_image']}"


def _issue_cover_s3_key(volume_id: int, issue_name: str) -> Optional[str]:
    """Derive S3 key for an issue cover image."""
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data:
        return None
    iss_data = next((i for i in vol_data["issues"] if str(i["num"]) == str(issue_name)), None)
    if not iss_data:
        return None
    year = iss_data.get("year", vol_data["year"])
    folder = _s3_conf["cover_folder_pattern"].format(vol_id=volume_id)
    filename = _s3_conf["cover_file_pattern"].format(
        vol_id=volume_id, issue_num=issue_name, year=year,
    )
    return f"{_s3_conf['covers_prefix']}{folder}{filename}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(BASE_DIR / "qdrant_data_tags")

try:
    from qdrant_client import models as qdrant_models
except ImportError:
    qdrant_models = None

try:
    from article_tagger import TAXONOMY
except ImportError:
    TAXONOMY = {}

try:
    from hybrid_search import (
        ask_question, ask_question_stream,
        get_qdrant_client, COLLECTION_NAME,
    )
    logger.info("Successfully imported hybrid_search module")
except ImportError as e:
    logger.error(f"Failed to import hybrid_search: {e}")
    ask_question = None
    ask_question_stream = None
    get_qdrant_client = None
    COLLECTION_NAME = None


TRANSLATIONS = {
    "ta": {
        "app_title": "பொன்னி களஞ்சியம்",
        "nav_ask_ai": "AI-யிடம் கேளுங்கள்",
        "nav_library": "நூலகம்",
        "nav_about": "பற்றி",
        "nav_toggle": "English",
        "hero_input_placeholder": "பொன்னி வரலாறு பற்றி கேளுங்கள்...",
        "sugg_founder": "பொன்னி இதழ் ஆசிரியர்கள்",
        "sugg_poets": "பாரதிதாசன் எழுதிய கட்டுரைகள்",
        "sugg_dravidian": "திராவிட இயக்கம்",
        "sugg_archive": "வேண்டாத ஆசை ஆசிரியர்",
        "lib_title": "பொன்னி மின் நூலகம்",
        "lib_desc": "1947–1955 வரையிலான அரிய தொகுப்புகளை ஆராயுங்கள்.",
        "lib_vol": "தொகுதி",
        "lib_back": "நூலகத்திற்கு திரும்பு",
        "lib_back_issues": "இதழ்களுக்கு திரும்பு",
        "sources_title": "ஆதாரங்கள்",
        "searching": "தேடுகிறது...",
        "issue": "இதழ்",
        "issue_label": "இதழ்",
        "malar_label": "மலர்",
        "title_label": "தலைப்பு",
        "author_label": "எழுத்தாளர்",
        "read_more": "மேலும் படிக்க",
        "show_less": "குறைவாக காட்டு",
        "nav_tags": "வகைகள்",
        "browse_tags": "கட்டுரை வகைகள்",
        "browse_tags_desc": "பொன்னி இதழின் கட்டுரைகளை வகை வாரியாக ஆராயுங்கள்.",
        "articles_in_category": "கட்டுரைகள்",
        "articles_count": "கட்டுரைகள்",
        "back_to_tags": "வகைகளுக்கு திரும்பு",
        "untitled": "தலைப்பு இல்லை",
        "tags_select_issue": "கட்டுரைகளைப் பார்க்க ஒரு இதழைத் தேர்ந்தெடுக்கவும்",
        "tags_filter_placeholder": "தலைப்பு, எழுத்தாளர் மூலம் தேடுங்கள்...",
        "tags_filter_category": "வகை வடிகட்டி",
        "tags_all_categories": "அனைத்து வகைகள்",
        "tags_articles_for": "கட்டுரைகள்",
        "tags_read_pdf": "முழு இதழ் PDF படிக்க",
        "tags_no_articles": "கட்டுரைகள் எதுவும் கிடைக்கவில்லை",
        "tags_back_to_articles": "கட்டுரைகளுக்கு திரும்பு",
        "tags_showing": "காட்டுகிறது",
        "tags_of": "இல்",
        "tags_browse_desc": "தொகுதி மற்றும் இதழ் வாரியாக கட்டுரைகளை உலாவுங்கள்",
        "tags_volumes": "தொகுதிகள்",
        "tags_content_list": "உள்ளடக்க பட்டியல்",
        "tags_search_placeholder": "தலைப்பு, எழுத்தாளர், வகை மூலம் தேடுங்கள்...",
        "tags_filter_authors": "எழுத்தாளர்கள்",
        "tags_filter_titles": "தலைப்புகள்",
        "tags_filter_tags": "வகைகள்",
        "tags_filters": "வடிகட்டிகள்",
    },
    "en": {
        "app_title": "Ponni Archive",
        "nav_ask_ai": "Ask AI",
        "nav_library": "Library",
        "nav_about": "About",
        "nav_toggle": "தமிழ்",
        "hero_input_placeholder": "Ask about Ponni history...",
        "sugg_founder": "Writers of Ponni Magazine",
        "sugg_poets": "Articles by Bharathidasan",
        "sugg_dravidian": "Dravidian Movement",
        "sugg_archive": "The Unwanted Desire Teacher",
        "lib_title": "Ponni Digital Library",
        "lib_desc": "Browse restored volumes from 1947–1955.",
        "lib_vol": "Volume",
        "lib_back": "Back to Library",
        "lib_back_issues": "Back to Issues",
        "sources_title": "Sources",
        "searching": "Searching...",
        "issue": "Issue",
        "issue_label": "Issue",
        "malar_label": "Malar",
        "title_label": "Title",
        "author_label": "Author",
        "read_more": "Read More",
        "show_less": "Show Less",
        "nav_tags": "Categories",
        "browse_tags": "Article Categories",
        "browse_tags_desc": "Browse Ponni magazine articles by category.",
        "articles_in_category": "Articles",
        "articles_count": "articles",
        "back_to_tags": "Back to Categories",
        "untitled": "Untitled",
        "tags_select_issue": "Select an issue to view articles",
        "tags_filter_placeholder": "Search by title, author...",
        "tags_filter_category": "Filter by category",
        "tags_all_categories": "All Categories",
        "tags_articles_for": "Articles in",
        "tags_read_pdf": "Read Full Issue PDF",
        "tags_no_articles": "No articles found",
        "tags_back_to_articles": "Back to articles",
        "tags_showing": "Showing",
        "tags_of": "of",
        "tags_browse_desc": "Browse articles by volume and issue",
        "tags_volumes": "Volumes",
        "tags_content_list": "Content List",
        "tags_search_placeholder": "Search by title, author, category...",
        "tags_filter_authors": "Authors",
        "tags_filter_titles": "Titles",
        "tags_filter_tags": "Categories",
        "tags_filters": "Filters",
    }
}


def get_app_styles():
    """
    Return complete CSS styles for the application.

    Generates comprehensive CSS styling for the entire Streamlit web application,
    including navigation bar, chat interface, library cards, PDF viewer, and responsive
    design elements. Ensures consistent white background, Tamil-friendly typography,
    and smooth transitions.

    Returns:
        str: Complete CSS stylesheet as HTML string with <style> tags.

    Styling Includes:
        - Google Fonts (Inter) import for modern typography
        - Fixed navigation bar with blur effect
        - Hero section with gradient title
        - Chat input with fixed positioning and rounded corners
        - Source cards with hover effects
        - Volume and issue cards with image thumbnails
        - PDF viewer container
        - Expandable sections with custom colors
        - Responsive column layouts
        - Custom scrollbar styling

    """
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { scrollbar-color: #cbd5e1 #f1f5f9; }
    html, body { background: #ffffff !important; }
    .stApp { background: #ffffff !important; font-family: 'Inter', sans-serif; color: #1e3a8a; min-height: 100vh; }
    section.main, section.main > div, section[data-testid="stMain"], div[data-testid="stAppViewContainer"],
    div[data-testid="stApp"], .main, .block-container { background: #ffffff !important; }
    div[data-testid="stHorizontalBlock"], div[data-testid="column"], div[data-testid="stVerticalBlock"] {
        background: #ffffff !important; background-color: #ffffff !important; }
    .nav-container { display: flex; justify-content: flex-start; align-items: center; padding: 1rem 2rem;
        background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); position: fixed; top: 0; left: 0;
        width: 100%; z-index: 1000; border-bottom: 1px solid rgba(0,0,0,0.05); }
    .logo { font-weight: 800; font-size: 1.25rem; letter-spacing: -0.5px; color: #1e3a8a;
        text-decoration: none !important; display: flex; align-items: center; gap: 0.5rem; margin-right: 3rem; }
    .nav-links { display: flex; gap: 2rem; font-size: 0.95rem; font-weight: 500; align-items: center; flex-grow: 1; }
    .nav-link { color: #64748b; text-decoration: none !important; transition: color 0.2s; }
    .nav-link:hover { color: #1e3a8a; text-decoration: none !important; }
    .lang-toggle { font-size: 0.85rem; color: #64748b; text-decoration: none !important;
        border: 1px solid #e2e8f0; padding: 0.25rem 0.75rem; border-radius: 99px; margin-left: 1rem; }
    .lang-toggle:hover { background: #f1f5f9; color: #1e3a8a; text-decoration: none !important; }
    .hero-container { text-align: center; padding-top: 8rem; padding-bottom: 3rem; max-width: 800px; margin: 0 auto; }
    .hero-title { font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; line-height: 1.2; }
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"], section[data-testid="stBottom"],
    .stChatFloatingInputContainer, div[data-baseweb="base-input"] { background: #ffffff !important; }
    div[data-testid="stChatInput"], div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div { background: white !important; }
    div[data-testid="stChatInput"] { position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%);
        width: 100%; max-width: 800px !important; border-radius: 2rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important; background: white !important;
        border: 1px solid #e2e8f0 !important; padding: 0.6rem 0.75rem !important; z-index: 900; }
    div[data-testid="stSpinner"], div[data-testid="stSpinner"] > div { background: white !important; }
    div[data-testid="stChatInput"] input, div[data-testid="stChatInput"] textarea {
        background: white !important; color: #1e293b !important; }
    div[data-testid="stChatInput"] input::placeholder, div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important; }
    section[data-testid="stBottom"], section[data-testid="stBottom"] *, div[class*="bottom"],
    div[class*="Bottom"], div[class*="floating"], div[class*="Floating"] { background: white !important; }
    footer, .main footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"] { background: white !important; }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div, div[data-testid="stChatMessage"] li { color: #1e293b !important; }
    div[data-testid="stMarkdownContainer"] { color: #1e293b !important; }
    div[data-testid="stChatMessageContent"] { background-color: #ffffff; border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02); color: #1e293b !important; }
    div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] { background-color: #f8fafc; }
    div[data-testid="stChatMessage"].stChatMessage-user div[data-testid="stChatMessageContent"] { background-color: #ffffff; }
    .source-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.5rem;
        padding: 1rem; margin-bottom: 0.75rem; }
    .source-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .source-title { font-weight: 600; color: #1e3a8a; font-size: 0.95rem; }
    .source-meta { color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem; }
    .source-preview { color: #334155; font-size: 0.9rem; line-height: 1.5; font-style: italic;
        border-left: 3px solid #3b82f6; padding-left: 0.75rem; margin-top: 0.5rem; }
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stSidebar"] { display: none; }
    .stButton > button { background-color: white !important; color: #1e3a8a !important; border: none !important;
        border-radius: 0.75rem !important; padding: 0.75rem 1.5rem !important; font-weight: 600 !important;
        font-size: 0.95rem !important; transition: all 0.2s !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; }
    .stButton > button:hover { background-color: #f8fafc !important; transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }
    .stButton > button:active { transform: translateY(0) !important; }
    .pdf-container { width: 100%; height: 800px; border: 1px solid #e2e8f0; border-radius: 0.5rem;
        overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 1rem; }
    .pdf-container iframe { width: 100%; height: 100%; border: none; }
    div[data-testid="column"] { background: #ffffff !important; padding: 1.5rem; border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); transition: all 0.3s ease; }
    div[data-testid="column"]:hover { box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12); transform: translateY(-4px); }
    div[data-testid="column"] h3 { color: #1e3a8a; font-size: 1.1rem; margin-bottom: 1rem;
        text-align: center; font-weight: 600; }
    div[data-testid="column"] .stButton { width: 100%; }
    div[data-testid="column"] .stButton > button { width: 100%; background-color: #1e3a8a !important;
        color: white !important; border-radius: 0.5rem !important; padding: 0.75rem !important;
        font-weight: 600 !important; box-shadow: 0 2px 6px rgba(30, 58, 138, 0.2) !important; }
    div[data-testid="column"] .stButton > button:hover { background-color: #3b82f6 !important;
        transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3) !important; }
    div[data-testid="stExpander"] summary { background-color: #0f172a !important; color: #ffffff !important;
        border-radius: 0.5rem; padding: 0.75rem 1rem; font-weight: 600; }
    div[data-testid="stExpander"] summary span, div[data-testid="stExpander"] summary p { color: #ffffff !important; }
    div[data-testid="stExpander"] svg { fill: #ffffff !important; }
    .issue-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.75rem; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); transition: all 0.3s ease; cursor: pointer;
        height: 100%; text-decoration: none; display: block; }
    .issue-card:hover { box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12); transform: translateY(-4px); }
    .issue-card:active { transform: translateY(-2px); }
    .issue-card img { width: 100%; height: 280px; object-fit: cover; }
    .issue-card-title { padding: 1rem; text-align: center; color: #1e3a8a;
        font-weight: 600; font-size: 1.1rem; background: #f8fafc; }

    /* === Tags/Categories page — two-pane layout (design ref) === */
    .tags-sidebar-marker { display: none; }

    /* Reduce top gap — just enough for fixed nav bar; add side margins */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) .block-container {
        max-width: calc(100% - 4rem) !important; padding: 3.5rem 2rem 0 2rem !important; }

    /* Gap between left and right panes */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stHorizontalBlock"] {
        gap: 1.5rem !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"] {
        background: #ffffff !important; box-shadow: none !important;
        transform: none !important; padding: 0 !important;
        border-radius: 0.75rem !important; overflow: hidden; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:hover {
        box-shadow: none !important; transform: none !important; }

    /* Left column — light bg sidebar */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child {
        background: #f8fafc !important; border: 1px solid #e2e8f0; }

    /* Style the st.container(height=...) scrollable boxes — stretch to fill viewport */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important; border-radius: 0 !important;
        height: calc(100vh - 3.5rem) !important; max-height: calc(100vh - 3.5rem) !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0 !important; max-height: calc(100vh - 3.5rem) !important; }

    /* Right column — subtle border */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:last-child {
        border: 1px solid #e2e8f0; }

    /* Left scrollable container inner padding */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child
        div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 2.5rem !important; }

    /* Right scrollable container inner padding */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:last-child
        div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 2rem 2.5rem !important; }

    /* Button overrides inside tags page */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"] .stButton > button {
        background-color: #ffffff !important; color: #1e3a8a !important;
        width: auto !important; box-shadow: none !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"] h3 {
        color: #1e3a8a !important; text-align: left !important;
        font-size: 0.95rem !important; font-weight: 600 !important; }

    /* Sidebar collapse/expand toggle buttons — badge-style bg matching issues */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) .stButton > button,
    div[data-testid="stApp"]:has(.tags-sidebar-marker) .stButton > button[kind="secondary"],
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child > div > .stButton > button,
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child .stButton > button {
        background: #dbeafe !important; background-color: #dbeafe !important;
        color: #1e3a8a !important; -webkit-text-fill-color: #1e3a8a !important;
        border: 1px solid #bfdbfe !important; border-radius: 0.5rem !important;
        padding: 0.25rem 0.6rem !important; font-size: 1.1rem !important;
        font-weight: 700 !important; width: auto !important; min-height: 0 !important;
        box-shadow: none !important; line-height: 1 !important;
        transform: none !important; }

    /* Sidebar title and form elements */
    .tags-sidebar-title { font-size: 1rem; font-weight: 600; color: #1e3a8a;
        margin-bottom: 1.25rem; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child label {
        color: #1e3a8a !important; -webkit-text-fill-color: #1e3a8a !important;
        font-size: 0.85rem !important; font-weight: 500 !important; }
    /* Text input and selectbox styling in sidebar */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child input,
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child input[type="text"],
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child div[data-baseweb="input"] input,
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child div[data-testid="stTextInput"] input {
        color: #1e3a8a !important; -webkit-text-fill-color: #1e3a8a !important;
        background: #ffffff !important; border: 1px solid #e2e8f0 !important;
        caret-color: #1e3a8a !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="column"]:first-child input::placeholder {
        color: #94a3b8 !important; -webkit-text-fill-color: #94a3b8 !important; opacity: 1 !important; }
    /* Selectbox (category dropdown) — nuclear override: target ALL descendants */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stSelectbox"] * {
        background: #ffffff !important; background-color: #ffffff !important;
        color: #1e3a8a !important; -webkit-text-fill-color: #1e3a8a !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stSelectbox"]
        div[data-baseweb="select"] > div:first-child {
        border: 1px solid #e2e8f0 !important; border-radius: 0.5rem !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stSelectbox"] svg {
        fill: #64748b !important; }
    /* Dropdown list (popover) — renders as portal at root, target globally */
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] *,
    ul[role="listbox"] *,
    ul[role="listbox"],
    li[role="option"] {
        background: #ffffff !important; background-color: #ffffff !important;
        color: #1e3a8a !important; -webkit-text-fill-color: #1e3a8a !important; }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[data-baseweb="popover"] ul li:hover,
    div[data-baseweb="popover"] ul li[aria-selected="true"] {
        background: #dbeafe !important; background-color: #dbeafe !important; }
    .result-count { font-size: 0.85rem; color: #64748b; margin-top: 0.75rem;
        margin-bottom: 0.5rem; }
    .result-count strong { color: #1e3a8a; }

    /* Volume accordions in sidebar */
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] { margin-bottom: 0.5rem; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] summary {
        background-color: #ffffff !important; color: #1e3a8a !important;
        border-radius: 1rem !important; padding: 0.75rem 1rem !important;
        font-weight: 600 !important; font-size: 0.95rem !important;
        border: 1px solid #e2e8f0 !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] summary:hover {
        background-color: #f1f5f9 !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] summary span,
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] summary p {
        color: #1e3a8a !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] svg { fill: #64748b !important; }
    div[data-testid="stApp"]:has(.tags-sidebar-marker) div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        padding: 0.5rem 0.75rem !important; background: transparent !important; }

    /* File items inside accordion */
    .tags-file-list { display: flex; flex-direction: column; gap: 0.4rem; }
    .tags-file-item { display: flex; align-items: center; gap: 0.75rem;
        text-decoration: none; color: #1e293b; padding: 0.5rem 0.6rem;
        border-radius: 0.75rem; transition: all 0.2s; cursor: pointer;
        border: 1px solid transparent; }
    .tags-file-item:hover { background: #f1f5f9; text-decoration: none; color: #1e3a8a; }
    .tags-file-item.active { background: #dbeafe;
        border-color: #3b82f6; }
    .tags-file-item img { width: 48px; height: 48px; object-fit: cover;
        border-radius: 0.5rem; flex-shrink: 0; border: 1px solid #e2e8f0; }
    .tags-file-item .file-title { font-size: 0.85rem; font-weight: 500;
        color: #1e3a8a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        max-width: 180px; margin-bottom: 0.15rem; }
    .tags-file-item .file-badge { background: #f1f5f9; color: #64748b;
        font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 99px;
        display: inline-block; border: 1px solid #e2e8f0; }

    /* Empty state in main content pane */
    .tags-main-empty { display: flex; align-items: center; justify-content: center;
        min-height: 60vh; text-align: center; }
    .tags-main-empty .icon { font-size: 3.5rem; color: #cbd5e1; margin-bottom: 1rem; }
    .tags-main-empty p { font-size: 1.1rem; color: #64748b; }

    /* Badges row */
    .tags-badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .tags-badge-vol { background: #dbeafe; color: #1e3a8a;
        border: 1px solid #bfdbfe; padding: 0.25rem 0.75rem;
        border-radius: 99px; font-size: 0.8rem; font-weight: 500; }
    .tags-badge-cat { background: #dbeafe; color: #1e3a8a;
        border: 1px solid #bfdbfe; padding: 0.25rem 0.75rem;
        border-radius: 99px; font-size: 0.8rem; font-weight: 500; }

    /* Article content in main pane */
    .tags-article-block { background: #f8fafc; border-radius: 1rem;
        padding: 1.5rem; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
    .tags-article-block h3 { color: #1e3a8a; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.75rem; }
    .tags-article-block .article-text { color: #1e293b; font-size: 0.95rem;
        line-height: 1.8; white-space: pre-wrap; }
    .tags-article-block .article-meta { color: #64748b; font-size: 0.85rem;
        margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e2e8f0; }

    /* Stats row at bottom of main content */
    .tags-stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
        margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; }
    .tags-stat-card { background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 0.75rem; padding: 1rem; text-align: center; }
    .tags-stat-card .stat-val { font-size: 1.5rem; font-weight: 700; color: #3b82f6; }
    .tags-stat-card .stat-label { font-size: 0.75rem; color: #64748b; margin-top: 0.25rem; }

    .source-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
    .tag-badge { background: #dbeafe; color: #1e3a8a; font-size: 0.75rem; font-weight: 500;
        padding: 0.2rem 0.6rem; border-radius: 99px; border: 1px solid #bfdbfe; }

    .tags-pdf-btn { display: inline-flex; align-items: center; gap: 0.4rem;
        background: #dbeafe; color: #1e3a8a !important; padding: 0.5rem 1rem;
        border-radius: 0.5rem; text-decoration: none; font-size: 0.85rem;
        font-weight: 600; transition: background 0.2s; border: 1px solid #bfdbfe; }
    .tags-pdf-btn:hover { background: #bfdbfe; text-decoration: none;
        color: #1e3a8a !important; }
</style>
"""


def extract_file_id(pdf_url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from URL.

    Parses Google Drive URLs in various formats to extract the unique file identifier
    needed for embedding PDFs in the viewer.

    Args:
        pdf_url (str): Google Drive URL in any common format.

    Returns:
        str or None: Extracted file ID string, or None if extraction fails or URL is invalid.
    """
    if not pdf_url:
        return None
    try:
        if "id=" in pdf_url:
            return pdf_url.split("id=")[1].split("&")[0]
        elif "/d/" in pdf_url:
            return pdf_url.split("/d/")[1].split("/")[0]
    except (IndexError, AttributeError):
        logger.error(f"Failed to extract file ID from URL: {pdf_url}")
    return None


def initialize_session_state():
    """
    Initialize Streamlit session state variables.

    Sets up required session state variables on first app load to prevent KeyErrors
    and ensure consistent application state across reruns.

    Session State Variables Created:
        - language (str): Current UI language, defaults to "ta" (Tamil)
        - messages (list): Chat message history, defaults to empty list
    """
    if "language" not in st.session_state:
        st.session_state.language = "ta"
        logger.info("Initialized language to Tamil")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized empty message history")


def configure_page():
    """
    Configure Streamlit page settings.

    Sets up Streamlit page configuration including title, icon, layout, and sidebar
    visibility for the Ponni Archive application.

    Page Settings:
        - page_title: "Ponni Archive"
        - page_icon: "📜" (scroll emoji)
        - layout: "wide" (full-width layout)
        - initial_sidebar_state: "collapsed" (sidebar hidden by default)

    """


    st.set_page_config(
        page_title="Ponni Archive",
        page_icon="scroll",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    logger.info("Page configuration set")


def handle_query_parameters() -> Tuple[str, Optional[str], Optional[str]]:
    """
    Process URL query parameters for navigation and language switching.

    Reads and processes URL query parameters to determine current page, selected volume,
    selected issue, and handle language toggle requests. Updates session state and
    triggers rerun when language changes.

    Returns:
        tuple: Three-element tuple containing:
            - current_page (str): Page identifier ("home", "library", "issues", 
                                "pdf_viewer", "about")
            - selected_volume (str or None): Volume ID if viewing issues/PDF
            - selected_issue (str or None): Issue number if viewing PDF

    Query Parameters Processed:
        - page: Current page to display
        - volume: Selected volume ID
        - issue: Selected issue number
        - lang: Language switch trigger (removed after processing)
    """
    query_params = st.query_params
    current_page = query_params.get("page", "home")
    selected_volume = query_params.get("volume", None)
    selected_issue = query_params.get("issue", None)
    
    if "lang" in query_params:
        st.session_state.language = query_params["lang"]
        logger.info(f"Language switched to: {query_params['lang']}")
        current_params = st.query_params.to_dict()
        current_params.pop("lang", None)
        st.query_params.clear()
        st.query_params.update(current_params)
        st.rerun()
    
    logger.debug(f"Query params - page: {current_page}, volume: {selected_volume}, issue: {selected_issue}")
    return current_page, selected_volume, selected_issue


def t(key: str) -> str:
    """
    Get translation for current language.

    Retrieves translated text for a given key based on the current language setting
    in session state. Provides bilingual support (Tamil/English) for the entire UI.

    Args:
        key (str): Translation key to lookup (e.g., 'app_title', 'nav_ask_ai').

    Returns:
        str: Translated text for current language, or the key itself if translation
            not found (fallback behavior).
    """
    return TRANSLATIONS.get(st.session_state.language, {}).get(key, key)


def render_navigation_bar():
    """
    Render the top navigation bar with language toggle.

    Creates a fixed navigation bar at the top of the page with logo, navigation links
    (Ask AI, Library, About), and language toggle button. Maintains current page context
    during language switches.

    Navigation Elements:
        - Logo: App title with home link
        - Links: Ask AI, Library, About pages
        - Language Toggle: Switches between Tamil/English
    """
    lang = st.session_state.language
    target_lang = "en" if lang == "ta" else "ta"
    query_params = st.query_params
    current_page = query_params.get("page", "home")
    toggle_page_param = f"&page={current_page}" if current_page else ""
    
    nav_html = f"""
    <div class="nav-container">
        <a href="/?page=home" target="_self" class="logo">{t('app_title')}</a>
        <div class="nav-links">
            <a href="/?page=home" target="_self" class="nav-link">{t('nav_ask_ai')}</a>
            <a href="/?page=library" target="_self" class="nav-link">{t('nav_library')}</a>
            <a href="/?page=tags" target="_self" class="nav-link">{t('nav_tags')}</a>
            <a href="/?page=about" target="_self" class="nav-link">{t('nav_about')}</a>
            <a href="/?lang={target_lang}{toggle_page_param}" target="_self" class="lang-toggle">{t('nav_toggle')}</a>
        </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)
    logger.debug("Navigation bar rendered")


def handle_suggestion_click(prompt_text: str):
    """
    Handle click on suggestion button.

    Stores the suggestion prompt in session state for processing by the chat interface.
    Used to pre-populate chat input when user clicks a suggested query button.

    Args:
        prompt_text (str): The suggested question/prompt text to submit.
    """
    st.session_state.temp_submit = prompt_text
    logger.info(f"Suggestion clicked: {prompt_text[:50]}...")


def render_home_page():
    """
    Render the home page with AI chat interface.
    Displays the main chat interface with hero section, suggestion buttons (when no messages),
    chat history, and user input field. Handles both initial state and ongoing conversation.
    """
    logger.info("Rendering home page")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    lang = st.session_state.language
    
    if not st.session_state.messages:
        hero_html = f'<div class="hero-container"><h1 class="hero-title">{t("app_title")}</h1></div>'
        st.markdown(hero_html, unsafe_allow_html=True)
        
        col_spacer_left, col_main, col_spacer_right = st.columns([1, 2, 1])
        with col_main:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t('sugg_founder'), use_container_width=True):
                    prompt = "பொன்னி இதழில் எழுதிய ஆசிரியர்கள் யார்?" if lang == "ta" else "Who are the authors in ponni magazine?"
                    handle_suggestion_click(prompt)
                    st.rerun()
                if st.button(t('sugg_poets'), use_container_width=True):
                    prompt = "பாரதிதாசன் எழுதிய கட்டுரைகள் பட்டியலிடுக" if lang == "ta" else "List the articles written by Bharathidasan"
                    handle_suggestion_click(prompt)
                    st.rerun()
            with col2:
                if st.button(t('sugg_dravidian'), use_container_width=True):
                    prompt = "பொன்னி திராவிட இயக்கத்திற்கு எவ்வாறு பங்களித்தது?" if lang == "ta" else "How did Ponni contribute to the Dravidian movement?"
                    handle_suggestion_click(prompt)
                    st.rerun()
                if st.button(t('sugg_archive'), use_container_width=True):
                    prompt = "வேண்டாத ஆசை ஆசிரியர் யார்?" if lang == "ta" else "Who is the founder of Ponni magazine?"
                    handle_suggestion_click(prompt)
                    st.rerun()
    else:
        st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg_idx, msg["sources"])
    handle_user_input()


def render_sources(msg_idx: int, sources: List):
    """
    Render source citations for a message with correct field mapping.

    Displays an expandable section showing all source documents that informed the AI's
    response. Each source shows metadata (volume, issue, title, author) and content
    preview with read more/less functionality for long content.
    """
    with st.expander(f"{t('sources_title')} — {len(sources)}"):
        for idx, src in enumerate(sources, 1):
            # Extract metadata based on source type
            if hasattr(src, 'payload') and src.payload:
                # Direct from Qdrant
                payload = src.payload
                metadata = payload.get("metadata", {})
                full_content = payload.get("content", "").strip()
                
                # CORRECT MAPPING for new JSON format:
                malar = metadata.get("doc_id", "unknown")          # மலர் = doc_id
                issue = metadata.get("doc_issue", "unknown")       # இதழ் = doc_issue  
                heading = metadata.get("title", "")                # தலைப்பு = title
                author = metadata.get("author_name", "")           # எழுத்தாளர் = author_name
            else:
                # From formatted sources dict (hybrid_search.py format_sources)
                full_content = src.get("content", "").strip()
                malar = src.get("volume", "unknown")               # மலர்
                issue = src.get("doc_issue", "unknown")            # இதழ்
                heading = src.get("heading", "")                   # தலைப்பு
                author = src.get("author_name", "")                # எழுத்தாளர்
            
            # Build metadata display string
            meta_parts = []
            if issue and issue != "unknown":
                meta_parts.append(f"{t('issue_label')}: {issue}")  # இதழ்: 6
            if malar and malar != "unknown":
                meta_parts.append(f"{t('malar_label')}: {malar}")  # மலர்: 1
            if heading:
                meta_parts.append(f"{t('title_label')}: {heading}")
            if author:
                meta_parts.append(f"{t('author_label')}: {author}")
            
            meta_str = " • ".join(meta_parts) if meta_parts else "மெட்டாடேட்டா கிடைக்கவில்லை"
            
            # Handle expandable content for long sources
            read_more_key = f"read_more_{msg_idx}_{idx}"
            if read_more_key not in st.session_state:
                st.session_state[read_more_key] = False
            
            if len(full_content) > 300:
                preview_content = full_content[:300] + "..."
                display_content = full_content if st.session_state[read_more_key] else preview_content
                
                source_html = f"""
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-title">{t('sources_title')} {idx}</span>
                    </div>
                    <div class="source-meta">{meta_str}</div>
                    <div class="source-preview">{display_content}</div>
                </div>
                """
                st.markdown(source_html, unsafe_allow_html=True)
                
                button_label = t('show_less') if st.session_state[read_more_key] else t('read_more')
                if st.button(button_label, key=f"btn_{read_more_key}"):
                    st.session_state[read_more_key] = not st.session_state[read_more_key]
                    st.rerun()
            else:
                source_html = f"""
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-title">{t('sources_title')} {idx}</span>
                    </div>
                    <div class="source-meta">{meta_str}</div>
                    <div class="source-preview">{full_content}</div>
                </div>
                """
                st.markdown(source_html, unsafe_allow_html=True)

def handle_user_input():
    """
    Handle user input from chat interface with streaming support.
    Uses st.write_stream for real-time token display when available,
    falls back to non-streaming for mock mode.
    """
    if "temp_submit" in st.session_state:
        user_input = st.session_state.temp_submit
        del st.session_state.temp_submit
    else:
        user_input = st.chat_input(t("hero_input_placeholder"))

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        logger.info(f"User query: {user_input[:100]}...")
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        try:
            last_user_msg = st.session_state.messages[-1]["content"]
            if ask_question is None:
                raise ImportError("Hybrid search module not available")

            if ask_question_stream is not None:
                # Streaming path
                collected_sources = []

                def token_generator():
                    nonlocal collected_sources
                    for event in ask_question_stream(last_user_msg):
                        if event["type"] == "token":
                            yield event["content"]
                        elif event["type"] == "sources":
                            collected_sources = event["sources"]

                with st.chat_message("assistant"):
                    collected_answer = st.write_stream(token_generator())

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": collected_answer,
                    "sources": collected_sources,
                })
            else:
                # Non-streaming fallback (mock mode)
                with st.spinner(t("searching")):
                    result = ask_question(last_user_msg) or {}
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "sources": result.get("sources", []),
                    })
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_msg = "மன்னிக்கவும், பிழை ஏற்பட்டது" if st.session_state.language == "ta" else "Sorry, an error occurred"
            st.session_state.messages.append({"role": "assistant", "content": f"{error_msg}: {str(e)}", "sources": []})
        st.rerun()


def render_library_page():
    """
    Render the digital library page showing all volumes.
    Layout matches React frontend: 4-column grid, card with image + title + year.
    Each card rendered individually to avoid oversized HTML blocks.
    """
    logger.info("Rendering library page")

    # Inject CSS matching React styles
    st.markdown("""
    <style>
    .sl-volume-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.75rem;
        overflow: hidden; text-align: center; text-decoration: none; display: block;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04); transition: all 0.35s ease;
    }
    .sl-volume-card:hover { box-shadow: 0 8px 24px rgba(30,58,138,0.1); transform: translateY(-4px); border-color: #93c5fd; }
    .sl-volume-card-img { width: 100%; height: auto; display: block; }
    .sl-volume-card-info { padding: 0.85rem 0.75rem; background: #ffffff; }
    .sl-volume-card-title { font-family: 'Lora', 'Noto Serif Tamil', Georgia, serif; font-weight: 600; font-size: 1rem; color: #1e3a8a; line-height: 1.4; }
    .sl-volume-card-year { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.78rem; color: #94a3b8; margin-top: 0.25rem; letter-spacing: 0.04em; }
    </style>
    """, unsafe_allow_html=True)

    # Header matching React .library-header
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:Lora,Noto Serif Tamil,Georgia,serif;font-size:2rem;font-weight:700;color:#1e3a8a;margin-bottom:0.5rem;">{t("lib_title")}</div>'
        '<div style="width:40px;height:2px;background:linear-gradient(90deg,#1e3a8a,#3b82f6);border-radius:1px;margin:0.75rem 0;"></div>'
        f'<div style="color:#64748b;font-size:0.95rem;line-height:1.6;margin-bottom:2.5rem;">{t("lib_desc")}</div>',
        unsafe_allow_html=True,
    )

    # Build volume data
    volumes = []
    for vol in _magazine["volumes"]:
        issue_years = sorted(set(
            iss.get("year", vol["year"]) for iss in vol["issues"]
        ))
        year_display = f"{issue_years[0]}-{issue_years[-1]}" if len(issue_years) > 1 else (issue_years[0] if issue_years else vol["year"])
        volumes.append({"id": vol["id"], "year": year_display})

    # 4-column grid using st.columns, one card per cell
    for i in range(0, len(volumes), 4):
        cols = st.columns(4, gap="medium")
        for j in range(4):
            if i + j < len(volumes):
                vol = volumes[i + j]
                with cols[j]:
                    render_volume_card(vol)
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)


def render_volume_card(vol: Dict):
    """Render a single volume card matching React .volume-card layout."""
    s3_key = _volume_cover_s3_key(vol["id"])
    img = _load_s3_image(s3_key) if s3_key else None
    if img:
        img_str = _pil_to_base64(img)
        img_tag = f'<img class="sl-volume-card-img" src="data:image/jpeg;base64,{img_str}" alt="{t("lib_vol")} {vol["id"]}">'
    else:
        img_tag = f'<div style="height:250px;background:#f1f5f9;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:1.2rem;">{t("lib_vol")} {vol["id"]}</div>'
        logger.warning(f"Volume cover not found for vol {vol['id']}")
    card_html = f"""
    <a href="?page=issues&volume={vol['id']}" target="_self" class="sl-volume-card">
        {img_tag}
        <div class="sl-volume-card-info">
            <div class="sl-volume-card-title">{t('lib_vol')} {vol['id']}</div>
            <div class="sl-volume-card-year">{vol['year']}</div>
        </div>
    </a>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def set_page(**params):
    """Update query parameters for navigation.
    Helper function to modify URL query parameters for page navigation while
    preserving existing parameters.
    """
    qp = dict(st.query_params)
    qp.update({k: v for k, v in params.items() if v is not None})
    st.query_params.clear()
    st.query_params.update(qp)


def render_issues_page(volume_id: str):
    """
    Render the issues page for a specific volume.

    Displays all available issues for a selected volume in a grid layout, with
    each issue showing its cover image and number. Includes back navigation button.
    """
    logger.info(f"Rendering issues page for volume {volume_id}")
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    if st.button(t("lib_back")):
        set_page(page="library")
        st.rerun()
    st.markdown(f"## {t('lib_vol')} {volume_id}")
    st.markdown("<br>", unsafe_allow_html=True)

    issues_data = load_volume_issues(volume_id)
    if not issues_data:
        logger.warning(f"No issues found for volume {volume_id}")
        return
    render_issue_grid(issues_data, volume_id)


def load_volume_issues(volume_id: str) -> List[Dict]:
    """
    Load issue data for a specific volume from the magazine registry.

    Returns list of issues with API proxy cover URLs and PDF availability.
    """
    vol_id = int(volume_id)
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == vol_id), None)
    if not vol_data:
        return []

    issues_data = []
    for iss in vol_data["issues"]:
        issue_name = str(iss["num"])
        issue_year = iss.get("year", vol_data["year"])
        issues_data.append({
            "issue_num": issue_name,
            "has_pdf": bool(iss.get("pdf_url")),
            "s3_key": _issue_cover_s3_key(vol_id, issue_name),
        })
    logger.info(f"Loaded {len(issues_data)} issues for volume {volume_id}")
    return issues_data


def render_issue_grid(issues_data: List[Dict], volume_id: str):
    """Render grid of issue cards."""
    for i in range(0, len(issues_data), 4):
        cols = st.columns(4, gap="medium")
        for j in range(4):
            if i + j < len(issues_data):
                issue = issues_data[i + j]
                with cols[j]:
                    render_issue_card(issue, volume_id)
        st.markdown("<br>", unsafe_allow_html=True)


def render_issue_card(issue: Dict, volume_id: str):
    """Render a single issue card — same pattern as original local-file code."""
    s3_key = issue.get("s3_key")
    img = _load_s3_image(s3_key, thumbnail_size=(300, 400)) if s3_key else None
    if img:
        img_str = _pil_to_base64(img)
        card_html = f"""
        <a href="?page=pdf_viewer&volume={volume_id}&issue={issue['issue_num']}" target="_self" class="issue-card">
            <img src="data:image/jpeg;base64,{img_str}" alt="{t('issue')} {issue['issue_num']}">
            <div class="issue-card-title">{t('issue')} {issue['issue_num']}</div>
        </a>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    else:
        logger.warning(f"Issue cover not found: vol {volume_id} issue {issue['issue_num']}")


def render_pdf_viewer_page(volume_id: str, issue_num: str):
    """
    Render PDF viewer page for a specific issue.

    Displays an embedded PDF viewer for a selected magazine issue using Google Drive
    preview. Includes back navigation button and direct link to open PDF in new tab.

    """
    logger.info(f"Rendering PDF viewer for volume {volume_id}, issue {issue_num}")
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    if st.button(t("lib_back_issues")):
        set_page(page="issues", volume=volume_id)
        st.rerun()
    st.markdown(f"## {t('lib_vol')} {volume_id} - {t('issue')} {issue_num}")
    
    pdf_key = f"vol_{volume_id}_issue_{issue_num}"
    pdf_url = PDF_LINKS.get(pdf_key)
    if not pdf_url:
        logger.warning(f"PDF not available for {pdf_key}")
        return
    
    file_id = extract_file_id(pdf_url)
    if not file_id:
        logger.error(f"Invalid PDF URL format for {pdf_key}")
        return
    
    embed_url = f"https://drive.google.com/file/d/{file_id}/preview"
    pdf_html = f'<div class="pdf-container"><iframe src="{embed_url}" width="100%" height="800px" allow="autoplay"></iframe></div>'
    st.markdown(pdf_html, unsafe_allow_html=True)
    st.markdown(f"[Open PDF in new tab]({pdf_url})", unsafe_allow_html=True)


def load_image(image_name: str, use_container_width: bool = False):
    """
    Load and display an image from S3.
    Handles volume covers ("Volume1") and about images ("about1").
    """
    s3_key = None
    if image_name.startswith("Volume"):
        vol_num = image_name.replace("Volume", "")
        if vol_num.isdigit():
            s3_key = _volume_cover_s3_key(int(vol_num))
    elif image_name.startswith("about"):
        s3_key = f"about/{image_name}.jpg"

    if s3_key and _s3_client:
        for key in _s3_key_with_fallback(s3_key):
            try:
                resp = _s3_client.get_object(Bucket=_s3_conf["bucket"], Key=key)
                raw = resp["Body"].read()
                img = Image.open(io.BytesIO(raw))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                st.image(img, use_container_width=use_container_width)
                logger.debug(f"Loaded image: {image_name} (from {key})")
                return
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    continue
                logger.warning(f"Failed to load image {key}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load image {key}: {e}")

    logger.warning(f"Could not resolve image: {image_name}")


@st.cache_data(ttl=300)
def fetch_all_tags():
    """Fetch all tags with article counts from Qdrant (qdrant_indexer collection)."""
    if get_qdrant_client is None or qdrant_models is None or not TAXONOMY:
        # Mock/fallback: return TAXONOMY with zero counts
        return [
            {"id": cat_id, "tamil": info["tamil"], "english": info["english"], "count": 0}
            for cat_id, info in TAXONOMY.items()
        ] if TAXONOMY else []

    try:
        client = get_qdrant_client()
        tag_counts = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(
                        key="type", match=qdrant_models.MatchValue(value="article")),
                    qdrant_models.FieldCondition(
                        key="metadata.chunk_id", match=qdrant_models.MatchValue(value=0)),
                ]),
                limit=500,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                metadata = (p.payload or {}).get("metadata", {})
                for tag in metadata.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if offset is None:
                break

        tags_list = []
        for cat_id, cat_info in TAXONOMY.items():
            tags_list.append({
                "id": cat_id,
                "tamil": cat_info["tamil"],
                "english": cat_info["english"],
                "count": tag_counts.get(cat_id, 0),
            })
        tags_list.sort(key=lambda x: x["count"], reverse=True)
        return tags_list
    except Exception as e:
        logger.error(f"Error fetching tags: {e}")
        return [
            {"id": cat_id, "tamil": info["tamil"], "english": info["english"], "count": 0}
            for cat_id, info in TAXONOMY.items()
        ]


@st.cache_data(ttl=300)
def fetch_tag_articles(tag_id):
    """Fetch articles for a specific tag from Qdrant (synchronous)."""
    if get_qdrant_client is None or qdrant_models is None:
        return []

    try:
        client = get_qdrant_client()
        seen = set()
        articles = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(
                        key="type", match=qdrant_models.MatchValue(value="article")),
                    qdrant_models.FieldCondition(
                        key="metadata.tags", match=qdrant_models.MatchAny(any=[tag_id])),
                    qdrant_models.FieldCondition(
                        key="metadata.chunk_id", match=qdrant_models.MatchValue(value=0)),
                ]),
                limit=500,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                metadata = (p.payload or {}).get("metadata", {})
                dedup_key = (metadata.get("doc_id"), metadata.get("doc_issue"))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                articles.append({
                    "title": metadata.get("title"),
                    "author_name": metadata.get("author_name"),
                    "doc_id": metadata.get("doc_id"),
                    "doc_issue": metadata.get("doc_issue"),
                    "year": metadata.get("year"),
                    "tags": metadata.get("tags", []),
                })
            if offset is None:
                break
        return articles
    except Exception as e:
        logger.error(f"Error fetching tag articles: {e}")
        return []


@st.cache_data
def get_volume_issue_counts():
    """Count issues per volume from magazine registry."""
    return {vol["id"]: len(vol["issues"]) for vol in _magazine["volumes"]}


@st.cache_data(ttl=300)
def fetch_issue_articles(volume_id, issue_num):
    """Fetch articles for a specific volume + issue from Qdrant.

    Uses metadata.doc_id (மலர் number) for volume filtering and
    metadata.doc_issue (இதழ் number) for issue filtering, since
    metadata.volume (from S3 key) may be 'unknown'.
    """
    if get_qdrant_client is None or qdrant_models is None:
        logger.warning("Qdrant client or models not available")
        return []
    try:
        client = get_qdrant_client()
        seen = set()
        articles = []
        vol_str = str(volume_id)
        issue_str = str(issue_num)

        # Step 1: Query by type=article, chunk_id=0, doc_id=volume number.
        # doc_id is the மலர் (volume) number extracted from the document text.
        all_points = []
        scroll_offset = None
        while True:
            points, scroll_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(
                        key="type", match=qdrant_models.MatchValue(value="article")),
                    qdrant_models.FieldCondition(
                        key="metadata.chunk_id", match=qdrant_models.MatchValue(value=0)),
                    qdrant_models.FieldCondition(
                        key="metadata.doc_id", match=qdrant_models.MatchValue(value=vol_str)),
                ]),
                limit=500,
                offset=scroll_offset,
                with_payload=True,
            )
            all_points.extend(points)
            if scroll_offset is None:
                break

        logger.info(f"fetch_issue_articles: vol={volume_id} (doc_id filter), "
                     f"issue={issue_num}, chunk_0 points: {len(all_points)}")

        # Fallback: if doc_id filter found nothing, fetch all and filter client-side
        if not all_points:
            logger.info("No points with doc_id filter — fetching all chunk_0 articles")
            scroll_offset = None
            while True:
                points, scroll_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qdrant_models.Filter(must=[
                        qdrant_models.FieldCondition(
                            key="type", match=qdrant_models.MatchValue(value="article")),
                        qdrant_models.FieldCondition(
                            key="metadata.chunk_id", match=qdrant_models.MatchValue(value=0)),
                    ]),
                    limit=500,
                    offset=scroll_offset,
                    with_payload=True,
                )
                all_points.extend(points)
                if scroll_offset is None:
                    break
            logger.info(f"Total chunk_0 articles (unfiltered): {len(all_points)}")
            # Log sample doc_id and doc_issue values for debugging
            if all_points:
                sample_info = set()
                for p in all_points[:20]:
                    meta = (p.payload or {}).get("metadata", {})
                    sample_info.add(
                        f"doc_id={meta.get('doc_id', 'MISSING')}, "
                        f"doc_issue={meta.get('doc_issue', 'MISSING')}, "
                        f"volume={meta.get('volume', 'MISSING')}"
                    )
                logger.info(f"Sample metadata: {sample_info}")
            # Client-side filter by doc_id (volume number)
            all_points = [
                p for p in all_points
                if str((p.payload or {}).get("metadata", {}).get("doc_id", "")) == vol_str
            ]
            logger.info(f"After client-side doc_id filter: {len(all_points)}")

        # Collect all unique doc_issue values for this volume
        all_issue_vals = set()
        for p in all_points:
            meta = (p.payload or {}).get("metadata", {})
            raw_val = meta.get("doc_issue")
            all_issue_vals.add(str(raw_val).strip() if raw_val is not None else "NA")
        sorted_issues = sorted(
            [v for v in all_issue_vals if v not in ("NA", "None", "")],
            key=lambda x: int(x) if x.isdigit() else float("inf"),
        )
        logger.info(f"All doc_issue values in vol {volume_id}: {sorted_issues}")

        # Build a mapping: sidebar issue position (1-based) → actual doc_issue value.
        # The sidebar numbers issues sequentially (1, 2, 3...) based on image files,
        # but doc_issue in Qdrant is the actual magazine issue number (இதழ்).
        issue_position_map = {i + 1: v for i, v in enumerate(sorted_issues)}
        # The actual doc_issue to match: look up by position, fall back to direct match
        target_issue = issue_position_map.get(int(issue_num), issue_str)
        logger.info(f"Issue mapping: sidebar position {issue_num} → doc_issue '{target_issue}' "
                     f"(map: {issue_position_map})")

        # Step 2: Filter by issue number using the mapped value.
        for p in all_points:
            metadata = (p.payload or {}).get("metadata", {})
            raw_issue = metadata.get("doc_issue", "")
            doc_issue_str = str(raw_issue).strip()

            if doc_issue_str != str(target_issue):
                continue
            doc_id = metadata.get("doc_id")
            article_no = str(metadata.get("article_no", ""))
            unique_key = f"{doc_id}_{doc_issue_str}_{article_no}"
            if unique_key in seen:
                continue
            seen.add(unique_key)
            articles.append({
                "doc_id": doc_id,
                "title": metadata.get("title"),
                "author_name": metadata.get("author_name"),
                "year": metadata.get("year"),
                "tags": metadata.get("tags", []),
                "doc_issue": doc_issue_str,
                "article_no": article_no,
            })

        logger.info(f"fetch_issue_articles result: {len(articles)} articles "
                     f"for vol {volume_id} issue {issue_num} (doc_issue={target_issue})")
        return articles
    except Exception as e:
        logger.error(f"Error fetching issue articles: {e}", exc_info=True)
        return []


@st.cache_data(ttl=300)
def fetch_article_content(doc_id, doc_issue, article_no):
    """Fetch all chunks of a specific article and concatenate content.

    Identifies the article by the combination of doc_id (volume/மலர்),
    doc_issue (issue/இதழ்), and article_no (sequence within the issue).
    """
    if get_qdrant_client is None or qdrant_models is None:
        return None
    try:
        client = get_qdrant_client()
        chunks = []

        # Build filter: doc_id + doc_issue + article_no uniquely identify an article
        filter_conditions = [
            qdrant_models.FieldCondition(
                key="type", match=qdrant_models.MatchValue(value="article")),
            qdrant_models.FieldCondition(
                key="metadata.doc_id", match=qdrant_models.MatchValue(value=str(doc_id))),
            qdrant_models.FieldCondition(
                key="metadata.doc_issue", match=qdrant_models.MatchValue(value=str(doc_issue))),
        ]
        if article_no:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.article_no",
                    match=qdrant_models.MatchValue(value=int(article_no) if str(article_no).isdigit() else article_no)),
            )

        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qdrant_models.Filter(must=filter_conditions),
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                payload = p.payload or {}
                metadata = payload.get("metadata", {})
                chunks.append({
                    "chunk_id": metadata.get("chunk_id", 0),
                    "content": payload.get("content", ""),
                    "metadata": metadata,
                })
            if offset is None:
                break

        logger.info(f"fetch_article_content: doc_id={doc_id}, doc_issue={doc_issue}, "
                     f"article_no={article_no}, chunks found: {len(chunks)}")

        if not chunks:
            return None
        chunks.sort(key=lambda c: c["chunk_id"])
        first_meta = chunks[0]["metadata"]
        full_content = "\n".join(c["content"] for c in chunks)
        return {
            "title": first_meta.get("title"),
            "author_name": first_meta.get("author_name"),
            "year": first_meta.get("year"),
            "doc_issue": first_meta.get("doc_issue"),
            "tags": first_meta.get("tags", []),
            "volume": first_meta.get("volume"),
            "content": full_content,
        }
    except Exception as e:
        logger.error(f"Error fetching article content: {e}")
        return None


def _get_issue_thumbnail_base64(volume_id, issue_num):
    """Return base64 data URI for an issue cover thumbnail (used in sidebar HTML)."""
    s3_key = _issue_cover_s3_key(int(volume_id), str(issue_num))
    if not s3_key:
        return None
    img = _load_s3_image(s3_key, thumbnail_size=(40, 55))
    if img:
        return f"data:image/jpeg;base64,{_pil_to_base64(img)}"
    return None




def _tag_display_name(tag_id, lang):
    """Map tag ID to display name using TAXONOMY."""
    if tag_id in TAXONOMY:
        return TAXONOMY[tag_id]["tamil"] if lang == "ta" else TAXONOMY[tag_id]["english"]
    return tag_id


def render_tags_page():
    """Render tags/categories page — two-pane layout matching design reference."""
    logger.info("Rendering tags page")

    query_params = st.query_params
    selected_volume = query_params.get("volume", "1")
    selected_issue = query_params.get("issue", "1")
    selected_article = query_params.get("article", None)
    actual_doc_issue = query_params.get("di", None)  # actual doc_issue from Qdrant

    # Sidebar collapsed state — expanded by default
    if "tags_sidebar_open" not in st.session_state:
        st.session_state.tags_sidebar_open = True

    # Calculate pane height: viewport minus nav bar
    pane_height = 700  # px — fallback; CSS will stretch to calc(100vh - 3.5rem)

    # Marker div for CSS :has() selector (must be outside columns)
    st.markdown('<div class="tags-sidebar-marker"></div>', unsafe_allow_html=True)

    if st.session_state.tags_sidebar_open:
        # Two-pane layout
        left_col, right_col = st.columns([2.5, 7.5])

        with left_col:
            # Collapse button at top of sidebar
            if st.button("\u2039", key="tags_collapse_btn"):
                st.session_state.tags_sidebar_open = False
                st.rerun()
            with st.container(height=pane_height, border=False):
                render_tags_sidebar(selected_volume, selected_issue)

        with right_col:
            with st.container(height=pane_height, border=False):
                if selected_article:
                    render_article_detail(
                        selected_article, selected_volume, selected_issue,
                        actual_doc_issue)
                else:
                    render_issue_articles(selected_volume, int(selected_issue))
    else:
        # Collapsed — full-width right pane with expand button
        if st.button("\u203a", key="tags_expand_btn"):
            st.session_state.tags_sidebar_open = True
            st.rerun()
        with st.container(height=pane_height, border=False):
            if selected_article:
                render_article_detail(
                    selected_article, selected_volume, selected_issue,
                    actual_doc_issue)
            else:
                render_issue_articles(selected_volume, int(selected_issue))


def render_tags_sidebar(selected_volume, selected_issue):
    """Render sidebar with category filter, search, and volume accordions."""
    lang = st.session_state.language

    st.markdown(
        f'<div class="tags-sidebar-title">{t("browse_tags")}</div>',
        unsafe_allow_html=True,
    )

    # Category filter dropdown
    cat_options = [t("tags_all_categories")] + [
        _tag_display_name(cid, lang) for cid in TAXONOMY
    ]
    cat_ids = [None] + list(TAXONOMY.keys())
    st.selectbox(
        t("tags_filter_category"),
        options=cat_options,
        index=0,
        key="tags_cat_filter",
    )

    # Search input
    search_q = st.text_input(
        t("tags_search_placeholder"),
        value="",
        key="tags_sidebar_search",
        placeholder=f"\U0001F50D {t('tags_search_placeholder')}",
    )

    # Resolve selected category
    cat_choice = st.session_state.get("tags_cat_filter", cat_options[0])
    chosen_cat_id = cat_ids[cat_options.index(cat_choice)] if cat_choice in cat_options else None

    # Fetch all articles to filter sidebar items
    all_articles = fetch_issue_articles(selected_volume, int(selected_issue))
    if chosen_cat_id:
        all_articles = [a for a in all_articles if chosen_cat_id in a.get("tags", [])]
    if search_q:
        q = search_q.lower()
        all_articles = [
            a for a in all_articles
            if q in (a.get("title") or "").lower()
            or q in (a.get("author_name") or "").lower()
            or any(q in _tag_display_name(tg, lang).lower() for tg in a.get("tags", []))
        ]
    result_count = len(all_articles)
    st.markdown(
        f'<div class="result-count"><strong>{result_count}</strong> '
        f'{t("articles_count")} found</div>',
        unsafe_allow_html=True,
    )

    # Volume accordions
    volumes = [{"id": vol["id"], "year": vol["year"]} for vol in _magazine["volumes"]]
    issue_counts = get_volume_issue_counts()

    for vol in volumes:
        vid = str(vol["id"])
        count = issue_counts.get(vol["id"], 0)
        is_expanded = (selected_volume == vid)
        label = f"{t('lib_vol')} {vid}  \u00b7  {count} {t('issue')}"
        with st.expander(label, expanded=is_expanded):
            _render_sidebar_issues(vid, selected_issue)


def _render_sidebar_issues(volume_id, selected_issue):
    """Render issue file items inside a volume accordion."""
    lang = st.session_state.language
    issues_data = load_volume_issues(volume_id)

    if not issues_data:
        st.markdown(
            '<div style="color:#64748b;font-size:0.85rem;padding:0.5rem;">No issues available</div>',
            unsafe_allow_html=True,
        )
        return

    html = '<div class="tags-file-list">'
    for issue in issues_data:
        inum = issue["issue_num"]
        is_active = (selected_issue == str(inum))
        active_cls = " active" if is_active else ""
        thumb_url = _get_issue_thumbnail_base64(int(volume_id), inum)
        thumb_html = (
            f'<img src="{thumb_url}" style="width:48px;height:48px;object-fit:cover;border-radius:0.5rem;" onerror="this.style.display=\'none\'">'
            if thumb_url
            else '<div style="width:48px;height:48px;background:#f1f5f9;border-radius:0.5rem;border:1px solid #e2e8f0;"></div>'
        )
        issue_label = f"{t('issue')} {inum}"
        html += f"""
        <a href="?page=tags&volume={volume_id}&issue={inum}" target="_self"
           class="tags-file-item{active_cls}">
            {thumb_html}
            <div>
                <div class="file-title">{issue_label}</div>
                <span class="file-badge">{t('lib_vol')} {volume_id}</span>
            </div>
        </a>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_issue_articles(volume_id, issue_num):
    """Render main content pane — article list for the selected issue."""
    lang = st.session_state.language

    # Badge row
    st.markdown(
        f'<div class="tags-badge-row">'
        f'<span class="tags-badge-vol">{t("lib_vol")} {volume_id}</span>'
        f'<span class="tags-badge-cat">{t("issue")} {issue_num}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Title
    st.markdown(
        f'<h1 style="font-size:1rem;font-weight:600;color:#1e3a8a;margin-bottom:0.5rem;">'
        f'{t("lib_vol")} {volume_id} {t("issue")} {issue_num} — {t("tags_content_list")}</h1>',
        unsafe_allow_html=True,
    )

    # PDF link
    pdf_key = f"vol_{volume_id}_issue_{issue_num}"
    pdf_url = PDF_LINKS.get(pdf_key)
    if pdf_url:
        st.markdown(
            f'<a href="{pdf_url}" target="_blank" class="tags-pdf-btn">'
            f'&#128196; {t("tags_read_pdf")}</a>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<hr style="border:none;border-top:1px solid #e2e8f0;margin:1rem 0;">',
        unsafe_allow_html=True,
    )

    # Fetch articles and apply sidebar filters
    articles = fetch_issue_articles(volume_id, issue_num)

    cat_options = [t("tags_all_categories")] + [
        _tag_display_name(cid, lang) for cid in TAXONOMY
    ]
    cat_ids = [None] + list(TAXONOMY.keys())
    cat_choice = st.session_state.get("tags_cat_filter", cat_options[0])
    chosen_cat_id = cat_ids[cat_options.index(cat_choice)] if cat_choice in cat_options else None
    search_q = st.session_state.get("tags_sidebar_search", "")

    if chosen_cat_id:
        articles = [a for a in articles if chosen_cat_id in a.get("tags", [])]
    if search_q:
        q = search_q.lower()
        articles = [
            a for a in articles
            if q in (a.get("title") or "").lower()
            or q in (a.get("author_name") or "").lower()
            or any(q in _tag_display_name(tg, lang).lower() for tg in a.get("tags", []))
        ]

    total = len(articles)
    if total == 0:
        st.markdown(
            f'<div class="tags-main-empty"><div>'
            f'<div class="icon">&#128196;</div>'
            f'<p>{t("tags_no_articles")}</p></div></div>',
            unsafe_allow_html=True,
        )
        return

    # Article list — each in a card block, clickable title
    for idx, article in enumerate(articles, 1):
        title = article.get("title") or t("untitled")
        author = article.get("author_name", "")
        tags = article.get("tags", [])
        article_no = article.get("article_no", "")

        # Tag badges
        badges_html = ""
        if tags:
            badges_html = " ".join(
                f'<span class="tags-badge-cat">{_tag_display_name(tg, lang)}</span>'
                for tg in tags
            )

        doc_issue = article.get("doc_issue", "")
        author_html = f'<span style="color:#64748b;font-size:0.85rem;"> — {author}</span>' if author else ''
        link = f"?page=tags&volume={volume_id}&issue={issue_num}&article={article_no}&di={doc_issue}"

        st.markdown(
            f'<div class="tags-article-block">'
            f'<div style="display:flex;align-items:baseline;gap:0.5rem;flex-wrap:wrap;">'
            f'<span style="color:#3b82f6;font-weight:700;font-size:0.95rem;">{idx}.</span>'
            f'<a href="{link}" target="_self" style="text-decoration:none;">'
            f'<span style="color:#1e3a8a;font-weight:600;font-size:0.95rem;">{title}</span></a>'
            f'{author_html}'
            f'</div>'
            f'<div style="margin-top:0.5rem;">{badges_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Stats row
    unique_authors = len({a.get("author_name") for a in articles if a.get("author_name")})
    unique_tags = len({tg for a in articles for tg in a.get("tags", [])})
    st.markdown(
        f'<div class="tags-stats-row">'
        f'<div class="tags-stat-card"><div class="stat-val">{total}</div>'
        f'<div class="stat-label">{t("articles_count")}</div></div>'
        f'<div class="tags-stat-card"><div class="stat-val" style="color:#8b5cf6;">{unique_authors}</div>'
        f'<div class="stat-label">{t("tags_filter_authors")}</div></div>'
        f'<div class="tags-stat-card"><div class="stat-val" style="color:#10b981;">{unique_tags}</div>'
        f'<div class="stat-label">{t("tags_filter_tags")}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_article_detail(article_no, volume_id, issue_num, actual_doc_issue=None):
    """Render full article view when an article is clicked from the list."""
    lang = st.session_state.language

    # Back button
    if st.button(f"\u2190 {t('tags_back_to_articles')}"):
        new_params = dict(st.query_params)
        new_params.pop("article", None)
        new_params.pop("di", None)
        st.query_params.clear()
        st.query_params.update(new_params)
        st.rerun()

    # Use the actual doc_issue from Qdrant if available, otherwise fall back to sidebar issue
    doc_issue_for_query = actual_doc_issue if actual_doc_issue else issue_num
    article = fetch_article_content(volume_id, doc_issue_for_query, article_no)
    if not article:
        st.markdown(
            f'<div class="tags-main-empty"><div>'
            f'<div class="icon">&#128196;</div>'
            f'<p>{t("tags_no_articles")}</p></div></div>',
            unsafe_allow_html=True,
        )
        return

    title = article.get("title") or t("untitled")
    author = article.get("author_name", "")
    year = article.get("year", "")
    doc_issue = article.get("doc_issue", "")
    tags = article.get("tags", [])
    content = article.get("content", "")

    # Badge row
    badges = f'<span class="tags-badge-vol">{t("lib_vol")} {volume_id}</span>'
    for tg in tags:
        badges += f'<span class="tags-badge-cat">{_tag_display_name(tg, lang)}</span>'
    st.markdown(f'<div class="tags-badge-row">{badges}</div>', unsafe_allow_html=True)

    # Title
    st.markdown(
        f'<h1 style="font-size:1rem;font-weight:600;color:#1e3a8a;margin-bottom:1rem;">{title}</h1>',
        unsafe_allow_html=True,
    )

    # PDF link
    pdf_key = f"vol_{volume_id}_issue_{issue_num}"
    pdf_url = PDF_LINKS.get(pdf_key)
    if pdf_url:
        st.markdown(
            f'<a href="{pdf_url}" target="_blank" class="tags-pdf-btn">'
            f'&#128196; {t("tags_read_pdf")}</a>',
            unsafe_allow_html=True,
        )

    # Content block
    st.markdown(
        f'<div class="tags-article-block">'
        f'<h3>{t("tags_content_list")}</h3>'
        f'<div class="article-text">{content}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metadata footer
    meta_parts = []
    if author:
        meta_parts.append(f"{t('author_label')}: {author}")
    if doc_issue:
        meta_parts.append(f"{t('issue_label')}: {doc_issue}")
    if year:
        meta_parts.append(str(year))
    if meta_parts:
        meta_str = " &bull; ".join(meta_parts)
        st.markdown(
            f'<div class="tags-article-block"><div class="article-meta">{meta_str}</div></div>',
            unsafe_allow_html=True,
        )



def _about_image_b64(image_name: str) -> str:
    """Return base64 img tag for an about image, or empty string on failure."""
    s3_key = f"about/{image_name}.jpg"
    if not _s3_client:
        return ""
    for key in _s3_key_with_fallback(s3_key):
        try:
            resp = _s3_client.get_object(Bucket=_s3_conf["bucket"], Key=key)
            raw = resp["Body"].read()
            img = Image.open(io.BytesIO(raw))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            b64 = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/jpeg;base64,{b64}" alt="{image_name}">'
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                continue
        except Exception as e:
            logger.warning(f"About image load failed {key}: {e}")
    return ""


def render_about_page():
    """
    Render the About page — layout matches React frontend (ponniarchive.com/about).
    Images rendered individually to avoid oversized HTML blocks.
    """
    logger.info("Rendering about page")

    ta = st.session_state.get("language", "ta") == "ta"
    heading = "பொன்னி களஞ்சியம்" if ta else "Ponni Archive"
    subtitle = "1947–1955 வரையிலான தமிழ் கலை இலக்கிய இதழ்" if ta else "A Tamil literary magazine, 1947–1955"

    if ta:
        pull_quote = "'திராவிடர் கழகத்தை ஆதரிக்கும் ஏடுகள் மிகக் குறைவாக இருந்த காலம். அவையும் அழகில்லாமல், அச்சுப்பிழை மிகுந்து வெளிவந்தன. அந்த நேரத்தில் வண்ண முகப்பு அட்டை போட்டு அழகாக நடந்த இதழ் 'பொன்னி' தான்.'"
        pull_cite = "— கவியரசு கண்ணதாசன்"
    else:
        pull_quote = '"There were very few publications supporting the Dravidar Kazhagam at that time. Even those were published without aesthetics and full of printing errors. At that time, the only magazine that came out beautifully with a colour cover page was Ponni."'
        pull_cite = "— Poet Laureate Kannadasan"

    # CSS matching React about page
    st.markdown("""
    <style>
    .sl-about-header { text-align: center; margin-bottom: 3rem; padding-bottom: 2rem; border-bottom: 1px solid #e2e8f0; position: relative; }
    .sl-about-header::after { content: ''; position: absolute; bottom: -1px; left: 50%; transform: translateX(-50%); width: 40px; height: 2px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); }
    .sl-about-heading { font-family: 'Lora', 'Noto Serif Tamil', Georgia, serif; color: #1e3a8a; font-size: 2.25rem; font-weight: 700; margin-bottom: 0.75rem; }
    .sl-about-subtitle { font-size: 0.95rem; color: #64748b; max-width: 500px; margin: 0 auto; line-height: 1.6; }
    .sl-about-text { font-family: 'Noto Serif Tamil', 'Lora', Georgia, serif; line-height: 2; text-align: justify; color: #334155; font-size: 0.95rem; margin-bottom: 2rem; }
    .sl-about-pull-quote { font-family: 'Lora', 'Noto Serif Tamil', Georgia, serif; font-size: 1.1rem; font-style: italic; color: #1e3a8a; border-left: 3px solid #3b82f6; padding: 1.25rem 1.5rem; margin: 2rem 0; background: linear-gradient(135deg, rgba(219,234,254,0.3), rgba(241,245,249,0.3)); border-radius: 0 0.5rem 0.5rem 0; line-height: 1.8; }
    .sl-about-pull-quote cite { display: block; font-size: 0.85rem; font-style: normal; color: #64748b; margin-top: 0.75rem; font-weight: 500; }
    .sl-about-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #e2e8f0, transparent); margin: 2.5rem 0; }
    .sl-about-image { display: flex; justify-content: center; margin: 2.5rem 0; }
    .sl-about-image img { max-width: 55%; height: auto; border-radius: 0.75rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.04); }
    .sl-about-double-image { display: flex; justify-content: center; gap: 1.5rem; margin: 2.5rem 0; }
    .sl-about-double-image img { width: 45%; height: auto; border-radius: 0.75rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.04); }
    @media (max-width: 768px) {
        .sl-about-heading { font-size: 1.75rem; }
        .sl-about-text { font-size: 0.9rem; line-height: 1.9; }
        .sl-about-image img { max-width: 85%; }
        .sl-about-double-image { flex-direction: column; align-items: center; }
        .sl-about-double-image img { width: 85%; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div class="sl-about-header">
        <div class="sl-about-heading">{heading}</div>
        <div class="sl-about-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

    # Section 1
    st.markdown("""<div class="sl-about-text">
    திராவிட கருத்தியலைப் பட்டித்தொட்டி எங்கும் பரப்பும் முயற்சிக்குத் திராவிட கருத்தியலாளர்கள் பல்வேறு ஊடகங்களைப் கைக்கொண்டனர். அவற்றுள் இதழ்கள் குறிப்பிடத்தக்கன. குடியரசு, விடுதலை, திராவிடநாடு, திராவிடன், போர்வாள், தனியரசு, கிளர்ச்சி, குயில் போன்ற இதழ்கள் மிகப்பெரிய அளவில் அறிவு அரசியல் தளத்தில் தமிழ் மக்களிடையே பெரும் தாக்கத்தை ஏற்படுத்தின. இவ்விதழ்கள் பகுத்தறிவு, சுயமரியாதை, சமத்துவம் போன்ற கொள்கைகளை மக்களிடையே பரப்பியதுடன், சாதி, மத மூடநம்பிக்கைகளுக்கு எதிரான கருத்துகளை மிகக் காத்திரமாக முன்வைத்தன.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    1900களில் வெளிவந்த இதழ்கள் சமூக மாற்றத்திற்கும் முன்னேற்றத்திற்கும் பெருந்துணையாக அமைந்துள்ளன என்பது வரலாற்று ரீதியான உண்மை. 1947 முதல் 1955 வரை இயங்கிய கலை இலக்கிய இதழ் 'பொன்னி'. பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்றது. தொடங்கப்பட்ட முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது.
    </div>""", unsafe_allow_html=True)

    # Image 1 (single, centered)
    img1 = _about_image_b64("about1")
    st.markdown(f'<div class="sl-about-image">{img1}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sl-about-divider">', unsafe_allow_html=True)

    # Section 2
    st.markdown("""<div class="sl-about-text">
    திராவிட இதழ்களின் வரிசையில் வைத்து போற்றத்தக்க பெரிதும் அறியப்படாத இதழாகப் பொன்னி இதழ் திகழ்கிறது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகியவற்றை மிகத் தீவிரமாக எடுத்துரைக்கும் இதழாக இவ்விதழ் வெளிவந்தது. தமிழகத்தின் தலைசிறந்த எழுத்தாளர்களும் படைப்பாளர்களும் தம் சீரிய கருத்துகளை இவ்விதழின்வழி எடுத்துரைத்தனர். தமிழ்ச் சமூகத்தை அறிவுச் சமூகமாக்கும் முன்னெடுப்பில் பொன்னி இதழின் பணி தலையாயதாகும்.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="sl-about-pull-quote">
    {pull_quote}
    <cite>{pull_cite}</cite>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    திராவிடக் கருத்தியலை துப்பாக்கியாகச் செயல்பட்ட திரு. அரு. பெரியண்ணன் அவர்களும், உள்வாங்கி இரட்டைக்குழல் திரு. முருகு. சுப்பிரமணியம் அவர்களும் இணைந்து 1947ஆம் ஆண்டு பிப்ரவரி மாதம் பொன்னி இதழைத் தொடங்கினர். பொன்னி இதழ் வண்ண அட்டைப்படத்தில் மிக நேர்த்தியாக வடிவமைக்கப்பட்டு வெளியிடப்பெற்றது. பத்திரிக்கை துறையில் மற்றவர்கள் செய்துகாட்டாத புதுமை எல்லாம் அவர்கள் செய்து காட்டினார்கள். இன்றும் தமிழகத்தில் சிலரை அச்சுக்கலை நிபுணர்கள் என்று தேர்ந்தெடுத்தால், அவர்களில் பெரியண்ணன் மிக முக்கியமானவராக இருப்பார்' என்று குறிப்பிட்டுள்ளார்.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    தமிழ் இலக்கிய உலகில் முக்கியமான கவிஞர் பாரதிதாசன் அவரின் 'குயில்' இதழ் அரசால் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதினார். அவரின் கொள்கைகளையும் நடையையும் பின்பற்றி எழுதியவர்களை 'பாரதிதாசன் பரம்பரை கவிஞர்கள்' என்று அறிமுகப்படுத்தியது பொன்னி இதழ்.
    </div>""", unsafe_allow_html=True)

    # Images 2 & 3 (double, side by side)
    img2 = _about_image_b64("about2")
    img3 = _about_image_b64("about3")
    st.markdown(f'<div class="sl-about-double-image">{img2}{img3}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sl-about-divider">', unsafe_allow_html=True)

    # Section 3
    st.markdown("""<div class="sl-about-text">
    இவ்விதழில் மாநில சுயாட்சி, இந்தித் திணிப்பு, தனித்தமிழ் பற்று, விடுதலை, அரசியல், சமூகம் சார்ந்த திராவிடச் சிந்தனைகள் கட்டுரைகளாக, கதைகளாக, கவிதைகளாக. கிறனாய்வுகளாக, துணுக்குகளாக வெளிவந்தன.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர், திரு.வி.க., கலைஞர் மு. கருணாநிதி, கா. அப்பாதுரையார், கவியரசு கண்ணதாசன், டி. கே. சீனிவாசன், மு. வ., மு. அண்ணாமலை, கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்ற பல முதன்மையான இலக்கிய, அரசியல் ஆளுமைகள் பொன்னியில் எழுதியுள்ளனர்.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    கவிதைகள், சிறுகதைகள், தொடர்கதைகள், நொடிக் கதைகள், நாடகங்கள், பொதுக் கட்டுரைகள், ஆய்வுக் கட்டுரைகள், ஒப்பாய்வுக் கட்டுரைகள், தொடர் கட்டுரைகள், செய்திப் பாட்டு போன்ற இலக்கிய வகைமைகளில் பொன்னியில் படைப்புகள் வெளியாகியுள்ளன. இது மட்டுமன்றி அட்டைப்படக் குறிப்பு, மகளிர் அழகுக் குறிப்புகள், குழந்தை வளர்ப்புமுறை, பொன்னி வாழ்த்துகள், விகடங்கள், சிறுவர் அரங்கம் (சிறுவர் இலக்கியம்) போன்ற படைப்புகளும் இடம்பெற்றுள்ளன.
    </div>""", unsafe_allow_html=True)

    # Images 4 & 5 (double, side by side)
    img4 = _about_image_b64("about4")
    img5 = _about_image_b64("about5")
    st.markdown(f'<div class="sl-about-double-image">{img4}{img5}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sl-about-divider">', unsafe_allow_html=True)

    # Section 4
    st.markdown("""<div class="sl-about-text">
    1948ல் போராட்டச் செய்தி நாட்குறிப்பு என்னும் தலைப்பில் கா. அப்பாதுரையார் அவர்கள், அக்கால விடுதலைப் போராட்ட நிலவரங்களை பதிவு செய்துள்ளார். எங்கே, யார் எதற்காக கைது செய்யப் படுகிறார்கள்? அவர்களுக்கு என்ன தண்டனை? போன்றவற்றை இப்பகுதியில் காணமுடிகிறது. தமிழ் இலக்கியம் மட்டுமன்றி சீனம், யார் எதற்காகக் கைது செய்யப் உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர். பாரசீகம், ரஷ்ய, கன்னடம், உருது, வடமொழி, தெலுங்கு போன்ற உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர்.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    நாடக விளம்பரங்கள், புத்தக விளம்பரங்கள், திரைப்பட விளம்பரங்கள், வணிக விளம்பரங்கள் போன்றவை பொன்னி இதழில் இடம் பெற்றுள்ளன. கலையுலகம் என்ற பகுதியின் கீழ் திரைப்படங்கள், நாடகங்களின் விமர்சனங்களை எழுதியுள்ளனர். பொன்னியில் மேலும் ஒரு சிறப்பிற்குரிய விஷயம் அதில் இடம்பெற்றுள்ள படங்கள் மற்றும் ஓவியங்கள். படைப்பின் தலைப்புகளை வரைந்து இதழில் சேர்த்துள்ளனர். புதுமைப்பித்தன் நினைவுகளைப் பற்றி அவரது மனைவி கமலா அவர்கள் பொன்னி இதழில் எழுதியுள்ளார். பொன்னி இதழ் தொடங்கப்பெற்ற காலத்திலிருந்து இந்தி எதிர்ப்பு குறித்தான எழுத்துகள் தொடர்ந்து காத்திரமாக இடம்பெற்றுள்ளது. அறிஞர்களும் மக்களும் இதில் எழுதியுள்ளனர். பொன்னி இதழ் விடுதலை போராட்ட காலகட்டத்தில் வெளியான இதழ் என்பதால், அக்கால அரசியல் சூழ்நிலைகள் மற்றும் சமூக நிலைகள் படைப்புகளில் பிரதிபலிக்கின்றன.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    மார்க்சியம், பெண்ணியம் போன்ற இசங்களும் பொன்னி இதழில் இடம்பெற்றுள்ளன. இன்றைய தமிழ்நாடு 1947இல் மதராஸ் மாகாணமாக இருந்தது. 1950ல் அது மெட்ராஸ் மாநிலமாக மாறியது. இது தொடர்பான கட்டுரைகள் பொன்னியில் இடம்பெற்றுள்ளன. பொன்னியில் பார்ப்பனியத்திற்கு எதிரான கருத்துகளும் திராவிடத்தை ஆதரிக்கும் கருத்துகளும் வலுவாகத் தொடர்ந்து இடம்பெற்று வந்திருக்கின்றன. மாநில சுயாட்சி, தனித்தமிழ் போன்றவற்றைக் குறித்தும் பொன்னியில் எழுதப்பட்டுள்ளன. ஓர் இலக்கியம் கருத்துடன் சேர்ந்து காலத்திற்கு ஏற்ப அமைந்தால் மட்டுமே அது நிலைத்து நிற்கும். பொன்னியில் இடம் பெற்றுள்ள படைப்புகளும் அக்காலகட்ட சூழலுக்கு ஏற்ப அமைந்திருக்கின்றன. பொன்னி இதழ் ஒரு கலை இலக்கிய இதழாக மட்டுமின்றி புரட்சி இதழாகவே இருந்திருக்கிறது.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sl-about-text">
    1947 முதல் 1955 வரையிலான தமிழகத்தின் காலக் கண்ணாடியாகப் பொன்னி இதழ் விளங்குகிறது. தொடக்க காலத் திராவிடக் கருத்தியல்களையும், அவை பரப்பப்பெற்ற வடிவங்களையும் முறைகளையும் ஆயும் ஆய்வாளர்களுக்கு மிகச் சிறந்த களமாகப் பொன்னி இதழ்கள் அமையும். 1947க்கு பிறகான எழுத்துருக்கள், சிந்தனைகள், உரிமை முழக்கங்கள், கேலிச் சித்திரங்கள், நூலறிமுகங்கள் போன்றவற்றை அறியவும், அவற்றை ஆய்வுக்குட்படுத்தவும் பெரும் வாய்ப்பை பொன்னி இதழ்கள் ஏற்படுத்திக் கொடுக்கும்.
    </div>""", unsafe_allow_html=True)


def main():
    """
    Main application entry point.

    Orchestrates the entire Streamlit application flow including configuration,
    initialization, routing, and page rendering based on URL parameters.

    Application Flow:
        1. Configure Streamlit page settings
        2. Initialize session state variables
        3. Process URL query parameters
        4. Apply CSS styles
        5. Render navigation bar
        6. Route to appropriate page based on current_page parameter
        7. Log final page state

    Page Routing:
        - "library" → render_library_page()
        - "issues" → render_issues_page(volume_id)
        - "pdf_viewer" → render_pdf_viewer_page(volume_id, issue_num)
        - "about" → render_about_page()
        - default/other → render_home_page()
    """
    logger.info("Starting Ponni Archive application")
    configure_page()
    initialize_session_state()
    current_page, selected_volume, selected_issue = handle_query_parameters()
    st.markdown(get_app_styles(), unsafe_allow_html=True)
    render_navigation_bar()
    
    if current_page == "library":
        render_library_page()
    elif current_page == "issues":
        render_issues_page(selected_volume)
    elif current_page == "pdf_viewer":
        render_pdf_viewer_page(selected_volume, selected_issue)
    elif current_page == "tags":
        render_tags_page()
    elif current_page == "about":
        render_about_page()
    else:
        render_home_page()
    
    logger.info(f"Rendered page: {current_page}")


if __name__ == "__main__":
    main()