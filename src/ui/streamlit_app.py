import streamlit as st
import time

# Page Configuration
st.set_page_config(
    page_title="Tamil Nexus RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
        color: #0f172a !important; /* Force dark text */
    }
    
    /* Header Styling */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Force light theme on root blocks just in case */
    div[data-testid="stDecoration"] {
        display: none;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    
    div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    
    /* User Message Specifics */
    div[data-testid="stChatMessageContent"] > div {
        color: inherit;
    }
    
    /* Input Area Styling (Chat Input at bottom mostly) */
    div[data-testid="stChatInput"] textarea {
        border-radius: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        padding: 0.75rem;
        background-color: #f1f5f9 !important; /* Light Grey */
        color: #0f172a !important; /* Dark Text */
        caret-color: #0f172a; /* Cursor color */
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #334155 !important; /* Slate 700 - Darker placeholder */
        opacity: 1;
    }
    
    /* Force background removal for Input containers */
    div[data-testid="stTextInput"] {
        background: transparent !important;
    }

    div[data-testid="stChatInput"] {
        background: transparent !important;
    }

    /* Target the actual input elements again to be sure */
    input[type="text"], textarea {
         background-color: transparent !important; 
         /* We set specific bg colors below, but default should be clear */
    }

    /* Header Title Input Styling Override */
    div[data-testid="column"] .stTextInput input {
         font-size: 1.5rem;
         font-weight: 600;
         color: #4338ca !important;
         /* Use a much lighter background as requested */
         background: linear-gradient(to right, #ffffff, #f8fafc) !important;
         border: 1px solid rgba(226, 232, 240, 0.5); /* Very subtle border */
         padding: 0.5rem;
         margin: 0;
         height: auto;
         box-shadow: 0 1px 2px rgba(0,0,0,0.02); /* Slight lift */
    }
    
    /* Bottom Chat Pane / Sticky Footer Background */
    section[data-testid="stBottom"] {
        background-color: #f1f5f9 !important; /* Solid Light Grey (Slate 100) */
        border-top: 1px solid #cbd5e1 !important; /* Distinct Border */
        box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1); /* Shadow to separate from content */
    }
    
    /* Ensure no child elements inject dark backgrounds */
    section[data-testid="stBottom"] > div,
    section[data-testid="stBottom"] > div > div {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Ensure the textarea has the light white background we want */
    div[data-testid="stChatInput"] textarea {
         background-color: #ffffff !important;
         color: #0f172a !important;
         border-color: #e2e8f0 !important; /* Matches border of other elements */
         box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Slight elevation */
    }
    
    div[data-testid="column"] .stTextInput input:hover,
    div[data-testid="column"] .stTextInput input:focus {
         background: #ffffff !important;
         border-color: #4338ca;
         padding: 0.5rem;
    }
    

    /* Accent Color Utilities */
    .accent-text {
        color: #4f46e5;
        font-weight: 600;
    }
    
    /* Source Card Styling */
    .source-card {
        background-color: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .source-title {
        font-weight: 600;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .source-snippet {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .relevance-tag {
        display: inline-block;
        margin-top: 0.5rem;
        font-size: 0.7rem;
        padding: 0.125rem 0.375rem;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        font-weight: 500;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #4338ca !important; /* Indigo 700 */
    }
    
    /* Hide top padding */
    .main .block-container {
        padding-top: 2rem;
    }
    /* Remove borders/bg from icon buttons in sidebar header (approximated by column structure) */
    div[data-testid="column"] button[kind="secondary"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Apply gradient to history list buttons (full width ones) */
    /* Target buttons inside the sidebar that are NOT the icon buttons */
    /* Since we cannot easily distinguish by class, we rely on the container width usage or order */
    
    /* Better approach: Target specific keys if possible, or general styling */
    /* We will use a general rule for sidebar buttons and overrides for specific ones if needed, 
       but Streamlit generates random classes. 
       However, we know history items use use_container_width=True which usually adds a specific class 
       or we can target by absence of icon logic. 
       
       Let's target all secondary buttons in sidebar, and make the icon ones transparent specifically.
    */
    
    /* Sidebar Chat History Items (Inactive) - Light Grey Box */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #f1f5f9 !important; /* Slate 100 */
        border: 1px solid #e2e8f0 !important; /* Slate 200 */
        border-radius: 0.5rem !important;
        color: #475569 !important; /* Slate 600 */
        justify-content: flex-start !important;
        margin-bottom: 0.25rem !important;
        transition: all 0.2s !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }
    
    /* Sidebar Chat History Item (Active) - Indigo Theme */
    /* Target specifically the primary buttons in the sidebar */
    /* Note: Streamlit might not expose 'kind' easily to CSS, but active buttons usually get a specific class. 
       However, we passed type="primary" in the python code. 
       We need to distinguish them. 
       Often the primary button has a different class or styles. 
       Let's try to target by exclusion or a known attribute if possible. 
       Actually, Streamlit buttons usually have kind properly reflected or specific classes. 
       However, if the above generic rule overrides it, we need to be careful.
       
       Let's use the :has pseudo-selector or specific attributes if available. 
       The safest bet without inspecting is to trust that `button[kind="primary"]` logic MIGHT fail 
       if the attribute isn't there. 
       
       If type="primary" is used, the button usually has a reddish background by default. 
       We will try to override based on the generic button styles that Streamlit applies for primary.
    */
    
    /* Attempt to target primary/active button specifically */
    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid #4f46e5 !important;
        color: #4f46e5 !important;
    }
    
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: rgba(99, 102, 241, 0.2) !important;
        border-color: #4338ca !important;
        color: #4338ca !important;
    }

    /* WE MUST EXCLUDE THE ICON BUTTONS (Search, Plus, Trash) FROM THE BOX STYLING */
    /* These are usually in columns (stHorizontalBlock) or have specific distinct layouts */
    
    /* Icon Buttons (Search/Plus) in the header columns */
    div[data-testid="stHorizontalBlock"] .stButton button {
         background: transparent !important;
         border: none !important;
         padding: 0px !important;
         box-shadow: none !important;
    }
    
    /* Trash icon in the history list (it's in the second column) */
    /* We can try to target the smaller column 
       div[data-testid="column"]:nth-of-type(2) might work if we are lucky with structure, 
       but `column` is generic. 
       
       Alternative: The trash button text is small? 
       Let's rely on the fact that the trash button is just an icon "🗑".
       We can try to target buttons with specific text content if CSS allowed, but it doesn't.
       
       Workaround: The icons in the list (delete) are in a 1-unit column.
       The chat items are in a 5-unit column.
       Streamlit columns usually have width attributes or flex-basis.
    */
    
    /* Apply transparent style to delete buttons */
    /* This targets the buttons in the smaller second column of the history row */
    div[data-testid="column"] .stButton button {
        /* This is risky as it affects all columns. We need to be specific to the history grid. */
    }

    /* Popover Menu Styling: Remove Default Arrow/Caret and styles */
    div[data-testid="stPopover"] button {
        border: none !important;
        background: transparent !important;
        color: #64748b !important;
        box-shadow: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        font-size: 1.25rem !important; /* Make the dots visible */
    }
    
    div[data-testid="stPopover"] button:hover {
        background: rgba(0,0,0,0.05) !important;
        color: #0f172a !important;
    }

    /* Hide the SVG arrow inside the popover button */
    div[data-testid="stPopover"] button > div > div > svg {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Logic & Data
# -----------------------------------------------------------------------------

def generate_mock_response(query):
    """Simulates the RAG backend logic."""
    lower_query = query.lower()
    
    if any(k in lower_query for k in ['thirukkural', 'திருக்குறள்', 'அறம்']):
        return {
            "text": "திருக்குறளின் அறத்துப்பாலில் வள்ளுவர் மனித வாழ்வின் அடிப்படை நெறிகளை விளக்குகிறார். குறிப்பாக, 'அறன் வலியுறுத்தல்' அதிகாரத்தில் அறத்தின் சிறப்பை எடுத்துரைக்கிறார்.",
            "sources": [
                {
                    "title": "Thirukkural_Analysis_Vol1.pdf", 
                    "snippet": "அறத்துப்பால் - அதிகாரம் 4: அறன் வலியுறுத்தல். குறள் 31: சிறப்பு ஈனும் செல்வமும் ஈனும் அறத்தினூஉங்கு ஆக்கம் எவனோ உயிர்க்கு.", 
                    "score": 98,
                    "content": "இது திருக்குறள் ஆய்வு நூல் தொகுதி 1 ஆகும்.\n\nஅதிகாரம் 4: அறன் வலியுறுத்தல்\n\nகுறள் 31:\nசிறப்பு ஈனும் செல்வமும் ஈனும் அறத்தினூஉங்கு\nஆக்கம் எவனோ உயிர்க்கு.\n\nவிளக்கம்:\nசிறப்பையும், செல்வத்தையும் தரக்கூடிய அறத்தை விட, உயிருக்கு ஆக்கம் தரக்கூடியது வேறு என்ன இருக்கிறது? அறமே தலையாயது.\n\n(This is dummy full content for demonstration purposes.)"
                },
                {
                    "title": "Valluvar_Research_2024.docx", 
                    "snippet": "நவீன காலத்தில் திருக்குறளின் அறக்கருத்துக்களின் தாக்கம் பற்றிய ஆய்வு.", 
                    "score": 85,
                    "content": "ஆய்வுக்கட்டுரை: நவீன வாழ்வில் வள்ளுவம்.\n\nஇக்கட்டுரை இன்றைய சமூக, பொருளாதார சூழலில் திருக்குறளின் கருத்துக்கள் எவ்வாறு பொருந்துகின்றன என்பதை ஆராய்கிறது.\n\n(Full content placeholder...)"
                }
            ]
        }
    elif any(k in lower_query for k in ['economy', 'பொருளாதாரம்', 'tn', 'தமிழ்நாடு']):
        return {
            "text": "தமிழ்நாட்டின் பொருளாதாரம் இந்தியாவின் இரண்டாவது பெரிய பொருளாதாரமாகும். 2024-25 நிதியாண்டில் மாநிலத்தின் ஜிடிபி வளர்ச்சி 8% ஆக இருக்கும் என்று கணிக்கப்பட்டுள்ளது. ஆட்டோமொபைல் மற்றும் ஜவுளித் துறைகள் முக்கிய பங்காற்றுகின்றன.",
            "sources": [
                {
                    "title": "TN_Budget_2024-25.pdf", 
                    "snippet": "மாநிலத்தின் மொத்த உள்நாட்டு உற்பத்தி (GSDP) 8% வளர்ச்சியடையும் என எதிர்பார்க்கப்படுகிறது. ஏற்றுமதி 15% அதிகரித்துள்ளது.", 
                    "score": 95,
                    "content": "தமிழ்நாடு பட்ஜெட் அறிக்கை 2024-25\n\nபொருளாதாரக் கண்ணோட்டம்:\nமாநிலத்தின் பொருளாதார வளர்ச்சி 8% ஆக கணிக்கப்பட்டுள்ளது. இது தேசிய சராசரியை விட அதிகம்.\n\nமுக்கியத் துறைகள்:\n1. ஆட்டோமொபைல்\n2. ஜவுளி\n3. தகவல் தொழில்நுட்பம்\n\n(This content represents the full text of the budget document.)"
                },
                {
                    "title": "Industrial_Policy_Draft.txt", 
                    "snippet": "புதிய தொழில் கொள்கை 2024: மின்னணு வாகன உற்பத்தி மற்றும் பசுமை எரிசக்திக்கு முன்னுரிமை.", 
                    "score": 78,
                    "content": "தொழில் கொள்கை வரைவு 2024\n\nநோக்கம்:\n2030 ஆம் ஆண்டிற்குள் 1 டிரில்லியன் டாலர் பொருளாதாரத்தை அடைதல்.\n\nமுன்னுரிமைகள்:\n- EV உற்பத்தி\n- புதுப்பிக்கத்தக்க ஆற்றல்\n\n(Dummy file content...)"
                }
            ]
        }
    else:
        return {
            "text": "உங்கள் கேள்விக்கு தொடர்புடைய தகவல்கள் அறிவுத் தளத்தில் தேடப்படுகின்றன. இதோ சில பொதுவான குறிப்புகள்.",
            "sources": [
                {
                    "title": "General_Tamil_History.pdf", 
                    "snippet": "தமிழ்நாட்டின் வரலாறு மற்றும் கலாச்சார சுருக்கம்.", 
                    "score": 60,
                    "content": "தமிழ்நாடு வரலாறு\n\nபண்டைய காலம் முதல் தற்காலம் வரை தமிழர்களின் வரலாறு மிக நீண்டது. மூவேந்தர்கள் (சேர, சோழ, பாண்டிய) ஆண்ட பெருமை கொண்டது.\n\n(Full text placeholder...)"
                },
                {
                    "title": "Sangam_Literature_Intro.pdf", 
                    "snippet": "சங்க கால இலக்கியங்கள்: எட்டுத்தொகை, பத்துப்பாட்டு அறிமுகம்.", 
                    "score": 55,
                    "content": "சங்க இலக்கிய அறிமுகம்\n\nகாலம்: கி.மு. 300 - கி.பி. 300\n\nநூல்கள்:\n1. எட்டுத்தொகை\n2. பத்துப்பாட்டு\n\nஇவை தமிழர்களின் அகம் மற்றும் புறம் சார்ந்த வாழ்வியலைப் பேசுகின்றன.\n\n(Dummy content...)"
                }
            ]
        }

# Initialize Session State
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "திருக்குறள் ஆய்வு": [
            {"role": "assistant", "content": "வணக்கம்! திருக்குறள் பற்றிய உங்கள் கேள்விகளைக் கேட்கலாம்.\n\n(Hello! Ask me about Thirukkural.)"}
        ],
        "தமிழ்நாடு பொருளாதாரம்": [
             {"role": "assistant", "content": "வணக்கம்! தமிழ்நாட்டின் பொருளாதாரம் மற்றும் பட்ஜெட் பற்றிய தகவல்களைப் பகிரத் தயாராக உள்ளேன்.\n\n(Hello! I can share info about TN Economy and Budget.)"}
        ],
        "சங்க இலக்கியம்": [
             {"role": "assistant", "content": "வணக்கம்! சங்க கால இலக்கியங்கள் (எட்டுத்தொகை, பத்துப்பாட்டு) பற்றிய விவரங்களை என்னிடம் கேட்கலாம்.\n\n(Hello! Ask me about Sangam Literature.)"}
        ]
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "திருக்குறள் ஆய்வு"

def set_chat(chat_name):
    st.session_state.current_chat = chat_name

def create_new_chat():
    # Find a unique name for the new chat
    base_name = "New Chat"
    new_chat_name = base_name
    count = 1
    while new_chat_name in st.session_state.chat_sessions:
        count += 1
        new_chat_name = f"{base_name} {count}"

    st.session_state.chat_sessions[new_chat_name] = [
        {"role": "assistant", "content": "வணக்கம்! புதிய உரையாடலைத் தொடங்குவோம். (Hello! Let's start a new conversation.)"}
    ]
    st.session_state.current_chat = new_chat_name

def rename_chat():
    new_name = st.session_state.get("new_chat_title")
    old_name = st.session_state.current_chat
    
    if new_name and new_name != old_name:
        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(old_name)
        st.session_state.current_chat = new_name

def delete_chat(chat_name):
    if chat_name in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[chat_name]
        
        # If we deleted the current active chat, switch to another one
        if st.session_state.current_chat == chat_name:
            keys = list(st.session_state.chat_sessions.keys())
            if keys:
                st.session_state.current_chat = keys[-1] # Switch to most recent
            else:
                create_new_chat() # Create a new default if all empty

def toggle_search():
    st.session_state.show_search = not st.session_state.get("show_search", False)


# -----------------------------------------------------------------------------
# Sidebar Layout
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1 style='font-size: 2rem; margin-top: 0;'>⬡ Tamil Nexus</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.markdown("<span style='color: #4338ca; font-weight: 600;'>HISTORY (வரலாறு)</span>", unsafe_allow_html=True)
    with col2:
         st.button("➕", key="new_chat_btn", help="New Chat", on_click=create_new_chat)
    with col3:
        st.button("🔍", key="search_history", help="Search History", on_click=toggle_search)
    
    # Search Input Box (Conditional)
    search_query = ""
    if st.session_state.get("show_search", False):
        search_query = st.text_input("Find chat...", key="sidebar_search", label_visibility="collapsed", placeholder="Search history...")

    # Dynamic History Items
    # Reverse to show newest first
    all_chats = list(st.session_state.chat_sessions.keys())[::-1]
    
    # Filter chats if search query exists
    if search_query:
        filtered_chats = [chat for chat in all_chats if search_query.lower() in chat.lower()]
    else:
        filtered_chats = all_chats

    for item in filtered_chats:
        col_item, col_menu = st.columns([5, 1])
        with col_item:
            # Highlight active chat
            type_ = "primary" if st.session_state.current_chat == item else "secondary"
            if st.button(f"💬 {item}", key=item, use_container_width=True, type=type_, on_click=set_chat, args=(item,)):
                 pass
        with col_menu:
             # Use a popover to simulate a context menu (closest to "right click" or "options" menu in Streamlit)
             with st.popover("⋮", use_container_width=True):
                 st.button("Delete", key=f"del_{item}", on_click=delete_chat, args=(item,), type="primary", use_container_width=True)




# -----------------------------------------------------------------------------
# Main Chat Layout
# -----------------------------------------------------------------------------

# Static App Info Pane
with st.container():
    st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1rem 1.5rem; margin-bottom: 2rem; margin-top: -1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem;">ℹ️</span>
                <h3 style="margin: 0; font-size: 1rem; color: #1e293b !important;">About Tamil Nexus RAG</h3>
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                This AI-powered assistant helps you explore Tamil literature, history, and economic data. 
                It uses Retrieval-Augmented Generation to provide accurate answers based on curated documents.
                <br><br>
                <strong>Current Version:</strong> v1.0.0 &nbsp;|&nbsp; <strong>Knowledge Base:</strong> Updated Dec 2025
            </p>
        </div>
    """, unsafe_allow_html=True)

# Title Header
active_chat = st.session_state.current_chat

# Ensure the widget state matches the active chat if we just switched or renamed
if "new_chat_title" not in st.session_state or st.session_state.new_chat_title != active_chat:
    st.session_state.new_chat_title = active_chat

# Editable Title
st.text_input(
    "Chat Title", 
    value=active_chat, # This is technically redundant if key is set and matches, but good for initial load
    key="new_chat_title", 
    label_visibility="collapsed",
    on_change=rename_chat,
    help="Edit chat title and press Enter"
)


# Display Chat Messages
current_messages = st.session_state.chat_sessions[st.session_state.current_chat]

for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are sources attached to this message, display them
        if "sources" in msg:
            st.markdown("#### Sources:")
            for src in msg["sources"]:
                with st.expander(f"📄 {src['title']} ({src['score']}% Match)"):
                    st.markdown(f"**Snippet:** _{src['snippet']}_")
                    st.markdown("---")
                    st.markdown("**Full Content:**")
                    st.text(src.get('content', 'Content not available.'))

# User Input
if prompt := st.chat_input("கேள்விகளைக் கேட்கவும்... (Ask a question)"):
    # Logic to auto-rename chat if it's a default "New Chat" name
    current_chat_name = st.session_state.current_chat
    
    # Check if this chat has a default "New Chat" name and is in its initial state (1 message = just the welcome)
    is_default_name = current_chat_name.startswith("New Chat")
    is_initial_state = len(st.session_state.chat_sessions[current_chat_name]) == 1
    
    if is_default_name and is_initial_state:
        # Generate new title from prompt (first 5 words or 30 chars)
        words = prompt.split()
        if len(words) > 5:
            new_title = " ".join(words[:5]) + "..."
        else:
            new_title = prompt
            
        # Ensure title is unique (though unlikely conflict with new prompt)
        if new_title in st.session_state.chat_sessions:
             new_title = f"{new_title} ({int(time.time())})"

        # Rename in session state
        st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_chat_name)
        st.session_state.current_chat = new_title
        
        # PROCESS MESSAGE AND RESPONSE BEFORE RERUN
        # 1. Add User Message to the NEW chat history
        st.session_state.chat_sessions[new_title].append({"role": "user", "content": prompt})
        
        # 2. Generate and Add Response immediately (synchronously) so it appears on reload
        # We skip the spinner animation for this specific transition to ensure atomic update of title + content
        response_data = generate_mock_response(prompt)
        st.session_state.chat_sessions[new_title].append({
            "role": "assistant",
            "content": response_data["text"],
            "sources": response_data["sources"]
        })

        # Trigger rerun to update UI immediately with new title and history
        st.rerun()
    
    # STANDARD FLOW (No rename)
    # 1. Add User Message
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Simulate Processing
    with st.chat_message("assistant"):
        with st.spinner("சிந்திக்கிறது... (Thinking...)"):
            time.sleep(1.5) # Simulate latency
            response_data = generate_mock_response(prompt)
            
            st.markdown(response_data["text"])
            
            # Display Inline Sources
            if response_data["sources"]:
                st.markdown("#### Sources:")
                for src in response_data["sources"]:
                    with st.expander(f"📄 {src['title']} ({src['score']}% Match)"):
                        st.markdown(f"**Snippet:** _{src['snippet']}_")
                        st.markdown("---")
                        st.markdown("**Full Content:**")
                        st.text(src.get('content', 'Content not available.'))
            
    # 3. Save Assistant Message to State
    st.session_state.chat_sessions[st.session_state.current_chat].append({
        "role": "assistant", 
        "content": response_data["text"],
        "sources": response_data["sources"]
    })
