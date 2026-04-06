def recursive_chunk(text, max_size=50):
    if len(text) <= max_size:
        return [text]

    sentences = text.split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "

    if current:
        chunks.append(current.strip())

    return chunks


if __name__ == "__main__":
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = recursive_chunk(text, max_size=40)

    for c in chunks:
        print(c)
