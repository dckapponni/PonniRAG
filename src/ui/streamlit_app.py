

import time
import streamlit as st

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

# Check if language toggle was requested
if "lang" in query_params:
    st.session_state.language = query_params["lang"]
    # clear param to avoid sticking
    current_params = st.query_params.to_dict()
    current_params.pop("lang") # Remove lang to clean URL
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
        "about_title": "பொன்னி இதழின் மரபு (1947–1955)",
        "about_mission_title": "சீர்திருத்தத்திற்கான குரல்",
        "about_mission_text": "**1947** இல் **ஏ.ஆர். பெரியண்ணன்** மற்றும் **முருகு சுப்பிரமணியம்** ஆகியோரால் நிறுவப்பட்ட *பொன்னி*, வெறுமனே ஒரு இதழ் மட்டுமல்ல; இது **திராவிட இயக்கத்தின்** கலங்கரை விளக்கமாகத் திகழ்ந்தது. பகுத்தறிவுச் சிந்தனை மற்றும் சுயமரியாதை இயக்கத்தை இது முன்னெடுத்தது.",
        "about_poets_title": "பாரதிதாசன் பரம்பரை",
        "about_poets_text": "*பொன்னி*யின் மிகப்பெரிய பங்களிப்பு **பாரதிதாசன் பரம்பரையை** உருவாக்கியதே. இது **சுரதா**, **முடியரசன்**, **வாணிதாசன்** போன்ற **47 கவிஞர்களை** உலகுக்கு அறிமுகப்படுத்தியது.",
        "about_archive_title": "வரலாற்றை மீட்டெடுத்தல்",
        "about_archive_text": "நிறுவனர் ஏ.ஆர். பெரியண்ணன் அவர்களின் நூற்றாண்டு விழாவையொட்டி, **சென்னை கிறிஸ்தவக் கல்லூரி (MCC)** மற்றும் **ரோஜா முத்தையா ஆராய்ச்சி நூலகம் (RMRL)** ஆகிவற்றின் கூட்டு முயற்சியுடன் இந்த மின் காப்பகம் உருவாக்கப்பட்டுள்ளது.",
        "footer_founded": "தொடக்கம்: 1947",
        "footer_founder": "நிறுவனர்: ஏ.ஆர். பெரியண்ணன்",
        "filter_collection": "தொகுப்பு",
        "filter_author": "ஆசிரியர்",
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
        "about_title": "The Legacy of Ponni (1947–1955)",
        "about_mission_title": "A Voice for Reform",
        "about_mission_text": "Founded in **1947** by **A.R. Periyannan** and **Murugu Subramanium**, *Ponni* was a beacon of the **Dravidian movement**, championing rationalist thought and social reform.",
        "about_poets_title": "The Bharathidasan Parambarai",
        "about_poets_text": "One of *Ponni*'s most enduring contributions was nurturing the **Bharathidasan Parambarai**, introducing **47 poets** including legends like **Suratha**, **Mudiarasan**, and **Vanidasan**.",
        "about_archive_title": "Restoring History",
        "about_archive_text": "In honor of the centenary of Founder A.R. Periyannan, this archive was created in collaboration with **Madras Christian College (MCC)** and **Roja Muthiah Research Library (RMRL)**.",
        "footer_founded": "Founded: 1947",
        "footer_founder": "Founder: A.R. Periyannan",
        "filter_collection": "Collection",
        "filter_author": "Author",
    }
}

def t(key):
    return TRANSLATIONS[lang].get(key, key)

st.markdown(
    """
<style>
    /* 1. Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #f0f4f8 0%, #ffffff 100%);
        font-family: 'Inter', sans-serif;
        color: #1e3a8a; /* Deep Navy Blue */
    }

    /* 2. Navigation Bar */
    .nav-container {
        display: flex;
        justify_content: flex-start;
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
        text-decoration: none;
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
        text-decoration: none;
        transition: color 0.2s;
    }
    .nav-link:hover {
        color: #1e3a8a;
    }
    
    .action-btn {
        background-color: #2563eb;
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        text-decoration: none;
        font-weight: 600;
        transition: background-color 0.2s;
        margin-left: auto;
    }
    .action-btn:hover {
        background-color: #1d4ed8;
    }
    
    /* Small Toggle Link */
    .lang-toggle {
        font-size: 0.85rem;
        color: #64748b;
        text-decoration: none;
        border: 1px solid #e2e8f0;
        padding: 0.25rem 0.75rem;
        border-radius: 99px;
        margin-left: 1rem;
    }
    .lang-toggle:hover {
        background: #f1f5f9;
        color: #1e3a8a;
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

    /* 4. Chat Input Styling */
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
    
    div[data-testid="stChatInput"] input {
        background: transparent !important;
    }

    /* Input Container Override */
    section[data-testid="stBottom"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        pointer-events: none;
    }
    section[data-testid="stBottom"] > div {
        pointer-events: auto;
    }

    /* 5. Library Grid Styling */
    .lib-grid {
        padding-top: 6rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    /* We use standard Streamlit columns for the grid, but we can style the buttons inside them */
    div[data-testid="column"] button {
        height: 100%;
        width: 100%;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
        text-align: left;
        padding: 1.5rem;
    }
    div[data-testid="column"] button:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transform: translateY(-2px);
    }
    
    /* 6. Message Styling */
    div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Hide standard elements */
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stSidebar"] { display: none; }
    
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Navbar (HTML with Query Params)
# -----------------------------------------------------------------------------
# Determine target lang for toggle
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
# Logic: Mock RAG
# -----------------------------------------------------------------------------
def generate_mock_response(query):
    query = query.lower()
    if "ponni" in query or "periyannan" in query:
        return {
            "text": "Ponni Magazine (1947–1955), founded by AR Periyannan and Murugu Subramanium, was a cornerstone of the Dravidian literary movement. It nurtured the 'Bharathidasan Parambarai' poets.",
            "sources": [{"title": "Legacy_of_Ponni.pdf", "snippet": "Founded in 1947... Pillars of Dravidian Movement...", "score": 99}]
        }
    return {
        "text": "I can assist with queries about Ponni Magazine, Dravidian literature, and the 47 poets of the Bharathidasan lineage.",
        "sources": []
    }


# -----------------------------------------------------------------------------
# Render: Home (Chat)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Render: Home (Chat)
# -----------------------------------------------------------------------------
def render_home():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def handle_suggestion(prompt_text):
        st.session_state.temp_submit = prompt_text

    # --- HERO SECTION (Only if no messages) ---
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

        # Suggestions Grid
        col_SpacerL, col_Main, col_SpacerR = st.columns([1, 2, 1])
        with col_Main:
            c1, c2 = st.columns(2)
            with c1:
                # Localized Buttons
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
        # --- CHAT UI ---
        st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True) 
        
        # Display Messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"**{src['title']}** (Score: {src['score']})")


    # Input Processing
    if "temp_submit" in st.session_state:
        user_input = st.session_state.temp_submit
        del st.session_state.temp_submit
    else:
        user_input = st.chat_input(t('hero_input_placeholder'))

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Searching Archives..."):
            time.sleep(1) 
            response = generate_mock_response(user_input)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["text"],
                "sources": response["sources"]
            })
        st.rerun()


# -----------------------------------------------------------------------------
# Render: Library (Grid of Volumes)
# -----------------------------------------------------------------------------
def render_library():
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"## 📚 {t('lib_title')}")
    st.markdown(t('lib_desc'))
    
    # Grid of Dummy Volumes (Localized)
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

    # Display in grid
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
            st.markdown("---") # Spacing


# -----------------------------------------------------------------------------
# Render: Issues (List for a Volume)
# -----------------------------------------------------------------------------
def render_issues(volume_id):
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    
    if st.button(t('lib_back')):
        st.query_params["page"] = "library"
        st.rerun()

    st.markdown(f"## 📑 {t('lib_vol')} {volume_id}")
    st.markdown("Browse individual issues/chapters.") # Kept simple for now

    for i in range(1, 11):
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"### Issue {i}")
            with col2:
                st.markdown(f"**Chapter {i}**")
                if st.button(f"Read", key=f"issue_{i}"):
                    st.toast(f"Opening Issue {i}...")
            st.divider()


# -----------------------------------------------------------------------------
# Render: About
# -----------------------------------------------------------------------------
def render_about():
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"# {t('about_title')}")
        st.markdown(
            f"""
            ### 🌟 {t('about_mission_title')}
            {t('about_mission_text')}

            ### ✍️ {t('about_poets_title')}
            {t('about_poets_text')}

            ### 🏛️ {t('about_archive_title')}
            {t('about_archive_text')}
            """
        )
    
    with col2:
        st.info(
            f"""
            **Project Details**
            
            🏛️ **{t('footer_founded')}**
            👤 **{t('footer_founder')}**
            
            **Partners**:
            - Madras Christian College
            - Roja Muthiah Research Library
            """
        )

# -----------------------------------------------------------------------------
# Main Routing
# -----------------------------------------------------------------------------
if current_page == "library":
    render_library()
elif current_page == "issues":
    render_issues(selected_volume)
elif current_page == "about":
    render_about()
else:
    render_home()

