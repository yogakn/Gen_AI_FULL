import faiss
import numpy as np
from utils import get_embedding
from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            pass
    return text

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def ingest(pdf_path):
    print("Loading PDF...")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    embeddings = get_embedding(chunks)
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, chunks
