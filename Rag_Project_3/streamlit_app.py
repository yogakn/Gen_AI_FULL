import streamlit as st
from retriever import retrieve
from rag_pipeline import generate_answer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🤖 RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
        }
        .user-msg {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .bot-msg {
            background-color: #2E2E2E;
            color: #EAEAEA;
            border-radius: 10px;
            padding: 10px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 Smart RAG Chatbot</div>', unsafe_allow_html=True)
st.caption("Ask anything powered by Retrieval-Augmented Generation 🚀")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K Documents", 1, 10, 3)
    
    st.divider()
    st.markdown("### 🧠 About")
    st.write("""
    This chatbot uses:
    - Retrieval (Vector DB)
    - LLM for Answer Generation
    - Context-aware responses
    """)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
query = st.chat_input("💬 Ask your question...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)

    # Loading spinner
    with st.spinner("🔍 Retrieving & Generating response..."):
        contexts = retrieve(query, k=k)
        answer = generate_answer(query, contexts)

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
