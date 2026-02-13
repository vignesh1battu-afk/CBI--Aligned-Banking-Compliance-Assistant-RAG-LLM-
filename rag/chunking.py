def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_documents(documents):
    chunked_docs = []
    chunk_id = 0

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            chunked_docs.append({
                "chunk_id": chunk_id,
                "domain": doc["domain"],
                "source": doc["filename"],
                "text": chunk
            })
            chunk_id += 1

    return chunked_docs
