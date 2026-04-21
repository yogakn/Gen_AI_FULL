import streamlit as st
from ingest import ingest
from retrieve import retrieve
from rerank import rerank
from generate import generate_answer

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot")

# -------------------------------
# Cache
# -------------------------------
@st.cache_resource
def load_rag():
    index, docs = ingest("data.pdf")
    return index, docs

with st.spinner("🔄 Loading PDF..."):
    index, docs = load_rag()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")
retrieve_k = st.sidebar.slider("Retrieve Top-K", 1, 20, 10)
rerank_k = st.sidebar.slider("Rerank Top-K", 1, 10, 3)

# -------------------------------
# Input
# -------------------------------
query = st.text_input("💬 Ask your question:")

if query:
    with st.spinner("🔍 Thinking..."):

        # Step 1: Retrieve
        retrieved = retrieve(query, index, docs, k=retrieve_k)

        # Step 2: Rerank
        reranked = rerank(query, retrieved, top_k=rerank_k)

        # Step 3: Generate (USE RERANKED ONLY)
        answer = generate_answer(query, reranked)

    # Answer
    st.subheader("🧠 Answer")
    st.write(answer)

    # Retrieved
    with st.expander("🔍 Retrieved Chunks"):
        for i, doc in enumerate(retrieved):
            st.write(f"{i+1}. {doc[:300]}...")

    # Reranked
    with st.expander("⭐ Reranked Chunks"):
        for i, doc in enumerate(reranked):
            st.write(f"{i+1}. {doc[:300]}...")

st.markdown("---")
st.markdown("✅ FAISS + Rerank + OpenAI RAG")