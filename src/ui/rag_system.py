import time

import streamlit as st
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

# ============================================================================
# GPU CONFIGURATION - Check and Display GPU Status
# ============================================================================

def check_gpu_availability():
    """Check GPU availability and return status info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "available": True,
            "count": gpu_count,
            "name": gpu_name,
            "memory_gb": gpu_memory,
            "device": "cuda"
        }
    else:
        return {
            "available": False,
            "device": "cpu"
        }

GPU_INFO = check_gpu_availability()

# Page Configuration
st.set_page_config(
    page_title="Tamil RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Styling
st.markdown(
    """
<style>
    /* Main Background */
    .stApp {
        background-color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
        color: #0f172a !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.3);
    }

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

    div[data-testid="stChatInput"] textarea {
        border-radius: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        padding: 0.75rem;
        background-color: #ffffff !important;
        color: #0f172a !important;
        caret-color: #0f172a;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #334155 !important;
        opacity: 1;
    }

    div[data-testid="column"] .stTextInput input {
         font-size: 1.5rem;
         font-weight: 600;
         color: #4338ca !important;
         background: linear-gradient(to right, #ffffff, #f8fafc) !important;
         border: 1px solid rgba(226, 232, 240, 0.5);
         padding: 0.5rem;
         margin: 0;
         height: auto;
         box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    section[data-testid="stBottom"] {
        background-color: #ffffff !important;
        border-top: 1px solid #cbd5e1 !important;
        box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #4338ca !important;
    }

    .main .block-container {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] .stButton button {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 0.5rem !important;
        color: #475569 !important;
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

    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid #4f46e5 !important;
        color: #4f46e5 !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton button {
         background: transparent !important;
         border: none !important;
         padding: 0px !important;
         box-shadow: none !important;
    }

    div[data-testid="stPopover"] button {
        border: none !important;
        background: transparent !important;
        color: #64748b !important;
        box-shadow: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        font-size: 1.25rem !important;
    }

    div[data-testid="stPopover"] button > div > div > svg {
        display: none !important;
    }

    /* GPU Status Badge */
    .gpu-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .gpu-available {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
    }
    .gpu-unavailable {
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Configuration
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "tamil_nexus_documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "abhinand/tamil-llama-7b-instruct-v0.2"

MODEL_CONFIG = {
    "use_4bit_quantization": True,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 3,
    "device": GPU_INFO["device"],  # Use detected device
}

@st.cache_resource
def load_query_encoder():
    """Load encoder for query embedding - GPU optimized"""
    try:
        print("Loading query encoder...")
        print(f"Target device: {MODEL_CONFIG['device']}")
        
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Move model to GPU if available
        if GPU_INFO["available"]:
            model = model.to(MODEL_CONFIG['device'])
            print(f"✓ Query encoder loaded on GPU: {GPU_INFO['name']}")
        else:
            print("⚠ Query encoder loaded on CPU (GPU not available)")
        
        return model, None
    except Exception as e:
        error_msg = f"Failed to load query encoder: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg


@st.cache_resource
def load_qdrant_client():
    """Load Qdrant client"""
    try:
        print("Connecting to Qdrant...")
        client = QdrantClient(path=QDRANT_PATH)

        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            point_count = collection_info.points_count
            print(f"✓ Connected! Found {point_count} indexed documents")

            if point_count == 0:
                warning_msg = "Collection is empty. Please run setup_embeddings.py first!"
                print(f"⚠ {warning_msg}")
                return client, warning_msg

            return client, None

        except Exception:
            error_msg = f"Collection '{COLLECTION_NAME}' not found. Please run setup_embeddings.py first!"
            print(f"✗ {error_msg}")
            return None, error_msg

    except Exception as e:
        error_msg = f"Failed to connect to Qdrant: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg


@st.cache_resource
def load_llm_model():
    """Load LLM with GPU optimization"""
    try:
        print("Loading LLM...")
        print(f"Target device: {MODEL_CONFIG['device']}")
        print(f"4-bit quantization: {MODEL_CONFIG['use_4bit_quantization']}")

        if MODEL_CONFIG["use_4bit_quantization"] and GPU_INFO["available"]:
            print("Configuring 4-bit quantization for GPU...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                quantization_config=bnb_config,
                device_map="auto",  # Automatically maps to available GPU
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use FP16 for faster inference
            )
            print(f"✓ LLM loaded with 4-bit quantization on GPU")
        elif GPU_INFO["available"]:
            print("Loading LLM without quantization on GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            print(f"✓ LLM loaded on GPU: {GPU_INFO['name']}")
        else:
            print("⚠ Loading LLM on CPU (GPU not available)")
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="cpu",
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            use_fast=False,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Print memory usage if GPU is available
        if GPU_INFO["available"]:
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        return model, tokenizer, None

    except Exception as e:
        error_msg = f"Failed to load LLM: {str(e)}"
        print(f"✗ {error_msg}")
        return None, None, error_msg


def retrieve_from_qdrant(query, query_encoder, qdrant_client, top_k=10):
    """Retrieve relevant documents using GPU-accelerated encoding"""
    try:
        prefixed_query = f"query: {query}"
        
        # Encode query on GPU if available
        with torch.no_grad():
            query_embedding = query_encoder.encode(
                [prefixed_query],
                convert_to_tensor=True,
                device=MODEL_CONFIG['device']
            )
            # Convert back to numpy for Qdrant
            query_embedding = query_embedding.cpu().numpy()[0]

        print(f"\n🔍 Query: {query[:50]}...")
        print(f"📊 Embedding shape: {query_embedding.shape}")

        try:
            search_results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding.tolist(),
                limit=top_k,
            ).points
            print(f"✓ Found {len(search_results)} results")
        except AttributeError:
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=top_k,
            )
            print(f"✓ Found {len(search_results)} results (legacy API)")

        for i, result in enumerate(search_results[:3], 1):
            preview = result.payload["content"][:60].replace("\n", " ")
            print(f"  #{i}: Score={result.score:.4f} | {preview}...")

        results = []
        for result in search_results:
            results.append({
                "content": result.payload["content"],
                "type": result.payload["type"],
                "chunk_id": result.payload["chunk_id"],
                "total_chunks": result.payload["total_chunks"],
                "metadata": result.payload["metadata"],
                "similarity_score": result.score,
            })

        return results, None

    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        print(f"✗ {error_msg}")
        return [], error_msg


def generate_response_with_llm(query, relevant_chunks, llm_model, tokenizer):
    """Generate response using GPU-accelerated LLM"""
    try:
        # Build context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk["metadata"]
            if chunk["type"] == "article":
                context_parts.append(
                    f"[மூலம் {i}] {metadata.get('article_heading', 'N/A')} "
                    f"(எழுதியவர்: {metadata.get('article_author_name', 'N/A')}):\n{chunk['content']}"
                )
            else:
                context_parts.append(
                    f"[மூலம் {i}] {metadata.get('heading', 'N/A')}:\n{chunk['content']}"
                )

        context = "\n\n".join(context_parts)

        prompt = f"""நீங்கள் ஒரு உதவிகரமான செயற்கை நுண்ணறிவு உதவியாளர். பயனர் கேட்கும் கேள்விகளுக்கு கீழே கொடுக்கப்பட்டுள்ள சூழலின் அடிப்படையில் மட்டுமே பதிலளிக்கவும். வழங்கப்பட்ட சூழலுக்கு வெளியே பதில்களை ஒருபோதும் சேர்க்க வேண்டாம். கேள்விக்கான பதில் உங்களுக்குத் தெரியாவிட்டால், தவறான தகவல்களைப் பகிர வேண்டாம். 

சூழல்:
{context}

கேள்வி: {query}

பதில்:"""

        # Tokenize and move to GPU
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

        print(f"🚀 Generating on device: {llm_model.device}")

        # Generate with GPU acceleration
        with torch.no_grad():
            if GPU_INFO["available"]:
                # Use torch.cuda.amp for mixed precision if on GPU
                with torch.cuda.amp.autocast():
                    outputs = llm_model.generate(
                        **inputs,
                        max_new_tokens=MODEL_CONFIG["max_new_tokens"],
                        temperature=MODEL_CONFIG["temperature"],
                        top_p=MODEL_CONFIG["top_p"],
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            else:
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=MODEL_CONFIG["max_new_tokens"],
                    temperature=MODEL_CONFIG["temperature"],
                    top_p=MODEL_CONFIG["top_p"],
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "பதில்:" in response:
            response = response.split("பதில்:")[-1].strip()

        return response, None

    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(f"✗ {error_msg}")
        return None, error_msg


def format_sources_for_display(relevant_chunks):
    """Format sources for UI"""
    sources = []
    for chunk in relevant_chunks:
        metadata = chunk["metadata"]
        similarity_score = int(chunk["similarity_score"] * 100)

        if chunk["type"] == "article":
            title = f"📄 {metadata.get('article_heading', 'Unknown')} - Article {metadata.get('article_no', 'N/A')}"
        else:
            title = f"📄 {metadata.get('heading', 'Unknown')} (Issue {metadata.get('doc_issue', 'N/A')})"

        snippet = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]

        sources.append({
            "type": chunk["type"],
            "title": title,
            "snippet": snippet,
            "score": similarity_score,
            "content": chunk["content"],
            "metadata": metadata,
            "chunk_info": f"Chunk {chunk['chunk_id'] + 1}/{chunk['total_chunks']}",
        })

    return sources

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "வரவேற்பு (Welcome)": [
            {
                "role": "assistant",
                "content": "வணக்கம்! உங்கள் கேள்விகளைக் கேட்கலாம்.\n\n(Hello! Ask me your questions.)",
            }
        ]
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "வரவேற்பு (Welcome)"

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if "initialization_error" not in st.session_state:
    st.session_state.initialization_error = None


if not st.session_state.models_loaded:
    print("\n" + "=" * 60)
    print("TAMIL RAG - GPU-OPTIMIZED INITIALIZATION")
    print("=" * 60)
    print(f"GPU Available: {GPU_INFO['available']}")
    if GPU_INFO["available"]:
        print(f"GPU Name: {GPU_INFO['name']}")
        print(f"GPU Memory: {GPU_INFO['memory_gb']:.2f} GB")
        print(f"Device Count: {GPU_INFO['count']}")
    print("=" * 60)

    query_encoder, encoder_error = load_query_encoder()
    if encoder_error:
        st.session_state.initialization_error = encoder_error

    qdrant_client, qdrant_error = load_qdrant_client()
    if qdrant_error:
        st.session_state.initialization_error = qdrant_error

    llm_model, tokenizer, llm_error = load_llm_model()
    if llm_error:
        st.session_state.initialization_error = llm_error

    if query_encoder and qdrant_client and llm_model and tokenizer:
        st.session_state.query_encoder = query_encoder
        st.session_state.qdrant_client = qdrant_client
        st.session_state.llm_model = llm_model
        st.session_state.tokenizer = tokenizer
        st.session_state.models_loaded = True

        print("=" * 60)
        print("✓ SYSTEM READY")
        print(f"✓ Running on: {MODEL_CONFIG['device'].upper()}")
        print("=" * 60 + "\n")
    else:
        print("=" * 60)
        print("✗ INITIALIZATION FAILED")
        print("=" * 60 + "\n")


def set_chat(chat_name):
    st.session_state.current_chat = chat_name

def create_new_chat():
    base_name = "New Chat"
    new_chat_name = base_name
    count = 1
    while new_chat_name in st.session_state.chat_sessions:
        count += 1
        new_chat_name = f"{base_name} {count}"

    st.session_state.chat_sessions[new_chat_name] = [
        {
            "role": "assistant",
            "content": "வணக்கம்! புதிய உரையாடலைத் தொடங்குவோம். (Hello! Let's start a new conversation.)",
        }
    ]
    st.session_state.current_chat = new_chat_name

def rename_chat():
    new_name = st.session_state.get("new_chat_title")
    old_name = st.session_state.current_chat

    if new_name and new_name != old_name and new_name not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(old_name)
        st.session_state.current_chat = new_name

def delete_chat(chat_name):
    if chat_name in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[chat_name]

        if st.session_state.current_chat == chat_name:
            keys = list(st.session_state.chat_sessions.keys())
            if keys:
                st.session_state.current_chat = keys[-1]
            else:
                create_new_chat()

def toggle_search():
    st.session_state.show_search = not st.session_state.get("show_search", False)

with st.sidebar:
    # GPU Status Badge
    if GPU_INFO["available"]:
        gpu_badge = f"<span class='gpu-badge gpu-available'>🚀 GPU: {GPU_INFO['name'][:20]}</span>"
    else:
        gpu_badge = "<span class='gpu-badge gpu-unavailable'>⚠️ CPU Mode</span>"
    
    st.markdown(
        f"<h1 style='font-size: 2rem; margin-top: 0;'>⬡ Ponni RAG {gpu_badge}</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.markdown(
            "<span style='color: #4338ca; font-weight: 600;'>HISTORY (வரலாறு)</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.button("➕", key="new_chat_btn", help="New Chat", on_click=create_new_chat)
    with col3:
        st.button("🔍", key="search_history", help="Search History", on_click=toggle_search)

    search_query = ""
    if st.session_state.get("show_search", False):
        search_query = st.text_input(
            "Find chat...",
            key="sidebar_search",
            label_visibility="collapsed",
            placeholder="Search history...",
        )

    all_chats = list(st.session_state.chat_sessions.keys())[::-1]
    filtered_chats = [chat for chat in all_chats if search_query.lower() in chat.lower()] if search_query else all_chats

    for item in filtered_chats:
        col_item, col_menu = st.columns([5, 1])
        with col_item:
            type_ = "primary" if st.session_state.current_chat == item else "secondary"
            st.button(
                f"💬 {item}",
                key=item,
                use_container_width=True,
                type=type_,
                on_click=set_chat,
                args=(item,),
            )
        with col_menu:
            with st.popover("⋮", use_container_width=True):
                st.button(
                    "Delete",
                    key=f"del_{item}",
                    on_click=delete_chat,
                    args=(item,),
                    type="primary",
                    use_container_width=True,
                )


if st.session_state.initialization_error:
    st.error("⚠️ System Initialization Error")
    st.error(st.session_state.initialization_error)
    st.info("**Solution:** Make sure you have run `setup_embeddings.py` first to index your documents!")
    st.code("python setup_embeddings.py", language="bash")
    st.stop()

# Info Banner with GPU Status
gpu_status = "✅ GPU Accelerated" if GPU_INFO["available"] else "⚠️ CPU Mode"
gpu_details = f"{GPU_INFO['name']} ({GPU_INFO['memory_gb']:.1f}GB)" if GPU_INFO["available"] else "No GPU detected"

with st.container():
    st.markdown(
        f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1rem 1.5rem; margin-bottom: 2rem; margin-top: -1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem;">ℹ️</span>
                <h3 style="margin: 0; font-size: 1rem; color: #1e293b !important;">About Ponni RAG</h3>
            </div>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                This AI-powered assistant helps you explore Tamil literature, history, and economic data.
                It uses Retrieval-Augmented Generation to provide accurate answers based on curated documents.
                <br><br>
                <strong>Version:</strong> v1.1.0 &nbsp;|&nbsp; <strong>Status:</strong> {gpu_status} &nbsp;|&nbsp; <strong>Hardware:</strong> {gpu_details}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

# Chat Title
active_chat = st.session_state.current_chat
if "new_chat_title" not in st.session_state or st.session_state.new_chat_title != active_chat:
    st.session_state.new_chat_title = active_chat

st.text_input(
    "Chat Title",
    value=active_chat,
    key="new_chat_title",
    label_visibility="collapsed",
    on_change=rename_chat,
    help="Edit chat title",
)

# Display Messages
current_messages = st.session_state.chat_sessions[st.session_state.current_chat]

for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.markdown("#### 📚 Sources:")
            for src in msg["sources"]:
                with st.expander(f"{src['title']} ({src['score']}% Match) - {src.get('chunk_info', '')}"):
                    st.markdown(f"**Snippet:** _{src['snippet']}_")

                    if src.get("metadata"):
                        st.markdown("**Metadata:**")
                        metadata = src["metadata"]
                        if src["type"] == "article":
                            st.write(f"- **Author:** {metadata.get('article_author_name', 'Unknown')}")
                            st.write(f"- **Article No:** {metadata.get('article_no', 'N/A')}")
                            st.write(f"- **Issue:** {metadata.get('doc_issue', 'N/A')}")
                        else:
                            st.write(f"- **Heading:** {metadata.get('heading', 'N/A')}")
                            st.write(f"- **Issue:** {metadata.get('doc_issue', 'N/A')}")

                    st.markdown("---")
                    st.markdown("**Full Content:**")
                    st.text_area(
                        "",
                        value=src.get("content", "N/A"),
                        height=200,
                        key=f"content_{hash(src['title'])}_{src.get('chunk_info', '')}",
                        disabled=True,
                    )

# Chat Input
if prompt := st.chat_input("கேள்விகளைக் கேட்கவும்... (Ask a question)"):
    current_chat_name = st.session_state.current_chat
    is_default_name = current_chat_name.startswith("New Chat") or current_chat_name == "வரவேற்பு (Welcome)"
    is_initial_state = len(st.session_state.chat_sessions[current_chat_name]) == 1
    
    # Auto-rename
    if is_default_name and is_initial_state:
        words = prompt.split()
        new_title = " ".join(words[:5]) + "..." if len(words) > 5 else prompt
            
        if new_title in st.session_state.chat_sessions:
            new_title = f"{new_title} ({int(time.time())})"

        st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_chat_name)
        st.session_state.current_chat = new_title
        current_chat_name = new_title
    
    # Add user message
    st.session_state.chat_sessions[current_chat_name].append({"role": "user", "content": prompt})
    
    # Retrieve
    with st.spinner(" தேடுகிறது... (Searching...)"):
        relevant_chunks, search_error = retrieve_from_qdrant(
            prompt,
            st.session_state.query_encoder,
            st.session_state.qdrant_client,
            top_k=MODEL_CONFIG['top_k']
        )
    
    if search_error:
        response_text = f" தேடல் பிழை / Search Error:\n\n{search_error}\n\nPlease check your Qdrant setup and try again."
        sources = []
    elif not relevant_chunks:
        response_text = "மன்னிக்கவும், தொடர்புடைய தகவல்கள் காணப்படவில்லை.\n\n(Sorry, no relevant information found.)\n\n💡 Make sure documents are indexed by running setup_embeddings.py"
        sources = []
    else:
        with st.spinner("பதில் உருவாக்குகிறது... (Generating...)"):
            response_text, gen_error = generate_response_with_llm(
                prompt,
                relevant_chunks,
                st.session_state.llm_model,
                st.session_state.tokenizer
            )
            
            if gen_error:
                response_text = f" உருவாக்கல் பிழை / Generation Error:\n\n{gen_error}\n\nPlease try again or simplify your question."
                sources = []
            else:
                sources = format_sources_for_display(relevant_chunks)
    
    # Add assistant response
    st.session_state.chat_sessions[current_chat_name].append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources
    })
    
    st.rerun()


