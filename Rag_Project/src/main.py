from ingestion import load_pdf
from chunking import chunk_documents
from embedding import get_embeddings
from vector_store import create_faiss_index
from retriever import retrieve

# -----------------------------
# Step 1: Load PDF
# -----------------------------
documents = load_pdf("../data/hr_policy.pdf")

# -----------------------------
# Step 2: Chunking
# -----------------------------
chunks = chunk_documents(documents)
texts = [chunk.page_content for chunk in chunks]

print(f"\nTotal Chunks Created: {len(texts)}")

# -----------------------------
# Step 3: Embeddings
# -----------------------------
embeddings = get_embeddings(texts)

# -----------------------------
# Step 4: FAISS Index
# -----------------------------
index = create_faiss_index(embeddings)

print("\n✅ Vector store created successfully!")

# -----------------------------
# Step 5: Ask Question (Loop)
# -----------------------------
while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query, index, chunks)

    print("\n📄 Retrieved Context:\n")
    for c in retrieved_chunks:
        print("-", c)

    # -----------------------------
    # Step 6: Generate Answer (RAG)
    # -----------------------------
    from config import client  # import here to avoid circular issues

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an HR assistant.

Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    print("\n🤖 Final Answer:\n")
    print(answer)