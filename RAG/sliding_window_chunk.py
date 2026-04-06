def sliding_window_chunk(text, chunk_size=50, overlap=10):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    text = "This is a simple example text for sliding window chunking method."
    chunks = sliding_window_chunk(text, chunk_size=25, overlap=5)

    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: {c}")
