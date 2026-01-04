import time
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(BASE_DIR / "qdrant_data")

# Import the hybrid search function
from hybrid_search import ask_question

# PDF mapping - converts Google Drive sharing links to direct download links
PDF_LINKS = {
    "vol_1_issue_1": "https://drive.google.com/uc?export=download&id=14WpSLrcqWqRXB9sQIvmRlWS2ffPKyerC",
    "vol_1_issue_2": "https://drive.google.com/uc?export=download&id=14WpSLrcqWqRXB9sQIvmRlWS2ffPKyerC",
}

# -----------------------------------------------------------------------------
# Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ponni Archive",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Localization & State Management
# -----------------------------------------------------------------------------
# Initialize Language
if "language" not in st.session_state:
    st.session_state.language = "ta"  # Default to Tamil

# Handle Query Parameters
query_params = st.query_params
current_page = query_params.get("page", "home")
selected_volume = query_params.get("volume", None)
selected_issue = query_params.get("issue", None)

# Check if language toggle was requested
if "lang" in query_params:
    st.session_state.language = query_params["lang"]
    # clear param to avoid sticking
    current_params = st.query_params.to_dict()
    current_params.pop("lang")
    st.query_params.clear()
    st.query_params.update(current_params)
    st.rerun()

lang = st.session_state.language

TRANSLATIONS = {
    "ta": {
        "app_title": "பொன்னி களஞ்சியம்",
        "app_subtitle": "பாரதிதாசன் பரம்பரை மற்றும் திராவிட பாரம்பரியத்தை பாதுகாத்தல்",
        "nav_ask_ai": "AI-யிடம் கேளுங்கள்",
        "nav_library": "நூலகம்",
        "nav_about": "பற்றி",
        "nav_login": "உள்நுழை",
        "nav_toggle": "English",
        "hero_input_placeholder": "பொன்னி வரலாறு பற்றி கேளுங்கள்...",
        "sugg_founder": "பொன்னியை நிறுவியவர் யார்?",
        "sugg_poets": "புகழ்பெற்ற கவிஞர்கள்",
        "sugg_dravidian": "திராவிட இயக்கம்",
        "sugg_archive": "காப்பக விவரங்கள்",
        "lib_title": "பொன்னி மின் நூலகம்",
        "lib_desc": "1947–1955 வரையிலான அரிய தொகுப்புகளை ஆராயுங்கள்.",
        "lib_vol": "தொகுதி",
        "lib_back": "← நூலகத்திற்கு திரும்பு",
        "lib_back_issues": "← இதழ்களுக்கு திரும்பு",
        "sources_title": "ஆதாரங்கள்",
        "searching": "தேடுகிறது...",
        "document_type": "ஆவண வகை",
        "article": "கட்டுரை",
        "intro": "அறிமுகம்",
        "filter_author": "எழுத்தாளர்",
        "volume": "தொகுதி",
        "heading": "தலைப்பு",
        "read": "படிக்க",
        "issue": "இதழ்",
        "about_mission_text": '''திராவிட கருத்தியலைப் பட்டித்தொட்டி எங்கும் பரப்பும் முயற்சிக்குத் திராவிட கருத்தியலாளர்கள் பல்வேறு ஊடகங்களைப் கைக்கொண்டனர். அவற்றுள் இதழ்கள் குறிப்பிடத்தக்கன. குடியரசு, விடுதலை, திராவிடநாடு, திராவிடன், போர்வாள், தனியரசு, கிளர்ச்சி, குயில் போன்ற இதழ்கள் மிகப்பெரிய அளவில் அறிவு அரசியல் தளத்தில் தமிழ் மக்களிடையே பெரும் தாக்கத்தை ஏற்படுத்தின. இவ்விதழ்கள் பகுத்தறிவு, சுயமரியாதை, சமத்துவம் போன்ற கொள்கைகளை மக்களிடையே பரப்பியதுடன், சாதி, மத மூடநம்பிக்கைகளுக்கு எதிரான கருத்துகளை மிகக் காத்திரமாக முன்வைத்தன.

1900களில் வெளிவந்த இதழ்கள் சமூக மாற்றத்திற்கும் முன்னேற்றத்திற்கும் பெருந்துணையாக அமைந்துள்ளன என்பது வரலாற்று ரீதியான உண்மை. 1947 முதல் 1955 வரை இயங்கிய கலை இலக்கிய இதழ் 'பொன்னி'. பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்றது. தொடங்கப்பட்ட முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது.'''
    },
    "en": {
        "app_title": "Ponni Archive",
        "app_subtitle": "Preserving the Bharathidasan Parambarai & Dravidian Legacy",
        "nav_ask_ai": "Ask AI",
        "nav_library": "Library",
        "nav_about": "About",
        "nav_login": "Log In",
        "nav_toggle": "தமிழ்",
        "hero_input_placeholder": "Ask about Ponni history...",
        "sugg_founder": "Who founded Ponni?",
        "sugg_poets": "Famous Poets",
        "sugg_dravidian": "Dravidian Movement",
        "sugg_archive": "Archive Details",
        "lib_title": "Ponni Digital Library",
        "lib_desc": "Browse restored volumes from 1947–1955.",
        "lib_vol": "Volume",
        "lib_back": "← Back to Library",
        "lib_back_issues": "← Back to Issues",
        "sources_title": "Sources",
        "searching": "Searching...",
        "document_type": "Document Type",
        "article": "Article",
        "intro": "Introduction",
        "filter_author": "Author",
        "volume": "Volume",
        "heading": "Heading",
        "read": "Read",
        "issue": "Issue",
        "about_mission_text": """Founded in February 1947 by A.R. Periyannan and Murugu Subramanium, Ponni magazine emerged as a powerful voice for the Dravidian movement during a transformative period in Tamil Nadu's history. This literary and cultural magazine operated from 1947 to 1955, initially publishing monthly and later twice monthly from 1948 onwards.""",
        "footer_founded": "Founded: 1947",
        "footer_founder": "A.R. Periyannan & Murugu Subramanium"
    }
}

def t(key):
    return TRANSLATIONS[lang].get(key, key)

st.markdown(
    """
<style>
    /* 1. Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force white background on EVERYTHING */
    * {
        scrollbar-color: #cbd5e1 #f1f5f9;
    }
    
    html, body {
        background: #ffffff !important;
    }
    
    .stApp {
        background: #ffffff !important;
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
        min-height: 100vh;
    }
    
    /* Ensure all main containers have white background */
    section.main,
    section.main > div,
    section[data-testid="stMain"],
    div[data-testid="stAppViewContainer"],
    div[data-testid="stApp"],
    .main,
    .block-container {
        background: #ffffff !important;
    }

    /* 2. Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }
    
    .logo {
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: -0.5px;
        color: #1e3a8a;
        text-decoration: none !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-right: 3rem;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        font-size: 0.95rem;
        font-weight: 500;
        align-items: center;
        flex-grow: 1;
    }
    
    .nav-link {
        color: #64748b;
        text-decoration: none !important;
        transition: color 0.2s;
    }
    .nav-link:hover {
        color: #1e3a8a;
        text-decoration: none !important;
    }
    
    .lang-toggle {
        font-size: 0.85rem;
        color: #64748b;
        text-decoration: none !important;
        border: 1px solid #e2e8f0;
        padding: 0.25rem 0.75rem;
        border-radius: 99px;
        margin-left: 1rem;
    }
    .lang-toggle:hover {
        background: #f1f5f9;
        color: #1e3a8a;
        text-decoration: none !important;
    }

    /* 3. Hero Section */
    .hero-container {
        text-align: center;
        padding-top: 8rem;
        padding-bottom: 3rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        color: #64748b;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }

    /* 4. Chat Input Styling - CRITICAL FIXES */
    
    /* Target the parent container of chat input */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    section[data-testid="stBottom"],
    .stChatFloatingInputContainer,
    div[data-baseweb="base-input"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }
    
    /* All parent divs around chat input */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div {
        background: white !important;
        background-color: white !important;
    }
    
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px !important;
        border-radius: 2rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.5rem 1rem !important;
        z-index: 900;
    }
    
    div[data-testid="stChatInput"] input,
    div[data-testid="stChatInput"] textarea {
        background: white !important;
        color: #1e293b !important;
    }
    
    div[data-testid="stChatInput"] input::placeholder,
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    /* NUCLEAR OPTION - Force white on ALL bottom elements */
    section[data-testid="stBottom"],
    section[data-testid="stBottom"] *,
    div[class*="bottom"],
    div[class*="Bottom"],
    div[class*="floating"],
    div[class*="Floating"] {
        background: white !important;
        background-color: white !important;
    }
    
    /* Ensure footer and all possible bottom areas are white */
    footer,
    .main footer,
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"] {
        background: white !important;
        background-color: white !important;
    }
    
    /* Fix chat message text visibility */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] li {
        color: #1e293b !important;
    }
    
    div[data-testid="stMarkdownContainer"] {
        color: #1e293b !important;
    }

    /* 5. Message Styling */
    div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        color: #1e293b !important;
    }
    
    div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {
        background-color: #f8fafc;
        color: #1e293b !important;
    }
    
    div[data-testid="stChatMessage"].stChatMessage-user div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        color: #1e293b !important;
    }
    
    /* 6. Source Cards */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .source-title {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.95rem;
    }
    
    .source-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .source-preview {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.5;
        font-style: italic;
        border-left: 3px solid #3b82f6;
        padding-left: 0.75rem;
        margin-top: 0.5rem;
    }

    /* 7. Library Grid */
    .lib-grid {
        padding-top: 6rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    div[data-testid="column"] button {
        height: 100%;
        width: 100%;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
        text-align: left;
        padding: 1.5rem;
        background-color: #f1f5f9 !important;
        color: #1e3a8a !important;
    }
    
    div[data-testid="column"] button:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transform: translateY(-2px);
        background-color: #e2e8f0 !important;
    }

    /* 8. Hide elements */
    header[data-testid="stHeader"] { 
        display: none; 
    }
    
    div[data-testid="stSidebar"] { 
        display: none; 
    }
    
    /* 9. Button Styling */
    button[kind="secondary"],
    button[kind="primary"],
    .stButton > button {
        background-color: #9ca3af !important;
        color: white !important;
        border: 1px solid #4b5563 !important;
    }
    
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover,
    .stButton > button:hover {
        background-color: #6b7280 !important;
        border-color: #4b5563 !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #9ca3af !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
        height: auto !important;
        white-space: normal !important;
        text-align: left !important;
    }

    /* 10. PDF Viewer Styling */
    .pdf-container {
        width: 100%;
        height: 800px;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .pdf-container iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Navbar
# -----------------------------------------------------------------------------
target_lang = "en" if lang == "ta" else "ta"
toggle_page_param = f"&page={current_page}" if current_page else ""

st.markdown(
    f"""
    <div class="nav-container">
        <a href="/?page=home" target="_self" class="logo">
            {t('app_title')}
        </a>
        <div class="nav-links">
            <a href="/?page=home" target="_self" class="nav-link">{t('nav_ask_ai')}</a>
            <a href="/?page=library" target="_self" class="nav-link">{t('nav_library')}</a>
            <a href="/?page=about" target="_self" class="nav-link">{t('nav_about')}</a>
            <a href="/?lang={target_lang}{toggle_page_param}" target="_self" class="lang-toggle">{t('nav_toggle')}</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Render Functions
# -----------------------------------------------------------------------------
def render_home():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def handle_suggestion(prompt_text):
        st.session_state.temp_submit = prompt_text

    if not st.session_state.messages:
        st.markdown(
            f"""
            <div class="hero-container">
                <h1 class="hero-title">{t('app_title')}</h1>
                <p class="hero-subtitle">{t('app_subtitle')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_SpacerL, col_Main, col_SpacerR = st.columns([1, 2, 1])
        with col_Main:
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"📓 {t('sugg_founder')}", use_container_width=True):
                    handle_suggestion("Who founded Ponni Magazine?")
                    st.rerun()
                if st.button(f"✍️ {t('sugg_poets')}", use_container_width=True):
                    handle_suggestion("List famous poets from the Bharathidasan Parambarai.")
                    st.rerun()
            with c2:
                if st.button(f"⚖️ {t('sugg_dravidian')}", use_container_width=True):
                    handle_suggestion("How did Ponni contribute to the Dravidian movement?")
                    st.rerun()
                if st.button(f"🏛️ {t('sugg_archive')}", use_container_width=True):
                    handle_suggestion("Tell me about the digitalization of Ponni archives.")
                    st.rerun()

    else:
        st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True) 
        
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"{t('sources_title')} ({len(msg['sources'])} {t('document_type').lower()})"):
                        for idx, src in enumerate(msg["sources"], 1):
                            # Build metadata display
                            meta_parts = [f"{t('volume')}: {src.get('volume', 'N/A')}"]
                            
                            if src.get('heading'):
                                meta_parts.append(f"{t('heading')}: {src['heading']}")
                            
                            if src.get('doc_issue'):
                                meta_parts.append(f"{t('issue')}: {src['doc_issue']}")
                            
                            meta_str = " • ".join(meta_parts)
                            
                            # Full content
                            full_content = src.get('content', '').strip()
                            
                            # Word count check (minimum 100 words)
                            word_count = len(full_content.split())
                            
                            # Create unique key for this source's expander state
                            read_more_key = f"read_more_{msg_idx}_{idx}"
                            
                            # Initialize session state for this source if not exists
                            if read_more_key not in st.session_state:
                                st.session_state[read_more_key] = False
                            
                            # Display preview or full content based on state
                            if len(full_content) > 300:  # Show read more if content is long
                                preview_content = full_content[:300] + "..."
                                
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span class="source-title">📄 {t('sources_title')} {idx}</span>
                                        </div>
                                        <div class="source-meta">{meta_str} • {word_count} words</div>
                                        <div class="source-preview" id="content_{read_more_key}">
                                            {full_content if st.session_state[read_more_key] else preview_content}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Read More / Show Less button
                                button_label = "Show Less " if st.session_state[read_more_key] else "Read More "
                                if st.button(button_label, key=f"btn_{read_more_key}", use_container_width=False):
                                    st.session_state[read_more_key] = not st.session_state[read_more_key]
                                    st.rerun()
                                    
                            else:
                                # Short content - display directly
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span class="source-title">📄 {t('sources_title')} {idx}</span>
                                        </div>
                                        <div class="source-meta">{meta_str} • {word_count} words</div>
                                        <div class="source-preview">{full_content}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

    if "temp_submit" in st.session_state:
        user_input = st.session_state.temp_submit
        del st.session_state.temp_submit
    else:
        user_input = st.chat_input(t('hero_input_placeholder'))

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner(t('searching')):
            try:
                # Call the hybrid search function
                result = ask_question(user_input)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                error_msg = "மன்னிக்கவும், பிழை ஏற்பட்டது" if lang == "ta" else "Sorry, an error occurred"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{error_msg}: {str(e)}",
                    "sources": []
                })
        
        st.rerun()


def render_library():
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"## 📚 {t('lib_title')}")
    st.markdown(t('lib_desc'))
    
    volumes = [
        {"id": 1, "title": f"Ponni {t('lib_vol')} 1", "desc": "1947", "icon": "📓"},
        {"id": 2, "title": f"Ponni {t('lib_vol')} 2", "desc": "1948", "icon": "🖊️"},
        {"id": 3, "title": f"Ponni {t('lib_vol')} 3", "desc": "1949", "icon": "📜"},
        {"id": 4, "title": f"Ponni {t('lib_vol')} 4", "desc": "1950", "icon": "⚖️"},
        {"id": 5, "title": f"Ponni {t('lib_vol')} 5", "desc": "1951", "icon": "📊"},
        {"id": 6, "title": f"Ponni {t('lib_vol')} 6", "desc": "1952", "icon": "🌱"},
        {"id": 7, "title": f"Ponni {t('lib_vol')} 7", "desc": "1953", "icon": "🏭"},
        {"id": 8, "title": f"Ponni {t('lib_vol')} 8", "desc": "1954", "icon": "⚖️"},
    ]

    cols = st.columns(2)
    for i, vol in enumerate(volumes):
        col = cols[i % 2]
        with col:
            if st.button(
                f"{vol['icon']} {vol['title']}\n\n{vol['desc']}", 
                key=f"vol_{vol['id']}",
            ):
                st.query_params["page"] = "issues"
                st.query_params["volume"] = str(vol['id'])
                st.rerun()
            st.markdown("---")


def render_issues(volume_id):
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    
    if st.button(t('lib_back')):
        st.query_params["page"] = "library"
        if "volume" in st.query_params:
            del st.query_params["volume"]
        st.rerun()

    st.markdown(f"## 📑 {t('lib_vol')} {volume_id}")
    st.markdown("Browse individual issues.")

    # For now, only Volume 1 has PDFs
    if volume_id == "1":
        issues_data = [
            {"issue_num": 1, "has_pdf": True},
            {"issue_num": 2, "has_pdf": True},
        ]
    else:
        issues_data = [
            {"issue_num": i, "has_pdf": False} for i in range(1, 11)
        ]

    for issue in issues_data:
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"### {t('issue')} {issue['issue_num']}")
            with col2:
                st.markdown(f"**{t('issue')} {issue['issue_num']}**")
                if issue['has_pdf']:
                    if st.button(t('read'), key=f"issue_{volume_id}_{issue['issue_num']}"):
                        st.query_params["page"] = "pdf_viewer"
                        st.query_params["volume"] = volume_id
                        st.query_params["issue"] = str(issue['issue_num'])
                        st.rerun()
                else:
                    st.markdown("*PDF not available yet*")
            st.divider()


def render_pdf_viewer(volume_id, issue_num):
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    
    if st.button(t('lib_back_issues')):
        st.query_params["page"] = "issues"
        if "issue" in st.query_params:
            del st.query_params["issue"]
        st.rerun()
    
    st.markdown(f"## 📖 {t('lib_vol')} {volume_id} - {t('issue')} {issue_num}")
    
    # Get PDF link
    pdf_key = f"vol_{volume_id}_issue_{issue_num}"
    pdf_url = PDF_LINKS.get(pdf_key)
    
    if pdf_url:
        # Extract file ID from the URL
        if "id=" in pdf_url:
            file_id = pdf_url.split("id=")[1].split("&")[0]
        elif "/d/" in pdf_url:
            file_id = pdf_url.split("/d/")[1].split("/")[0]
        else:
            st.error("Invalid PDF URL format")
            return
        
        # Create embed URL
        embed_url = f"https://drive.google.com/file/d/{file_id}/preview"
        
        # Display PDF in iframe
        st.markdown(
            f'''
            <div class="pdf-container">
                <iframe 
                    src="{embed_url}" 
                    width="100%" 
                    height="800px"
                    style="border: 1px solid #e2e8f0; border-radius: 0.5rem;"
                    allow="autoplay">
                </iframe>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        # Add open in new tab button
        st.markdown(f"[Open PDF in new tab]({pdf_url})", unsafe_allow_html=True)
        
    else:
        st.warning(f"PDF not available for Volume {volume_id}, Issue {issue_num}")

        
def render_about():
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## {t('nav_about')}")
        st.markdown(f"{t('about_mission_text')}")

# -----------------------------------------------------------------------------
# Main Routing
# -----------------------------------------------------------------------------
if current_page == "library":
    render_library()
elif current_page == "issues":
    render_issues(selected_volume)
elif current_page == "pdf_viewer":
    render_pdf_viewer(selected_volume, selected_issue)
elif current_page == "about":
    render_about()
else:
    render_home()