import os
from pypdf import PdfReader

PDF_FOLDER = "data/Pdfs"


def detect_domain(filename):
    filename = filename.lower()

    if "anti-money" in filename or "laundering" in filename or "aml" in filename:
        return "AML"
    elif "gdpr" in filename or "celex" in filename:
        return "GDPR"
    elif "consumer" in filename:
        return "Consumer Protection"
    elif "derivative" in filename:
        return "Governance"
    else:
        return "General"



def load_pdfs(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)

            full_text = ""
            for page_number, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            documents.append({
                "filename": filename,
                "domain": detect_domain(filename),
                "text": full_text
            })

            print(f"Loaded: {filename} | Domain: {detect_domain(filename)}")

    return documents


if __name__ == "__main__":
    docs = load_pdfs(PDF_FOLDER)
    print(f"\nTotal documents loaded: {len(docs)}")

    # Print sample preview
    if docs:
        print("\nSample Preview:\n")
        print(docs[0]["text"][:1000])




from rag.chunking import chunk_documents

if __name__ == "__main__":
    docs = load_pdfs(PDF_FOLDER)

    chunked = chunk_documents(docs)

    print(f"\nTotal chunks created: {len(chunked)}")
    print("\nSample chunk:\n")
    print(chunked[0]["text"][:500])












from rag.chunking import chunk_documents
from rag.embedding import embed_chunks

if __name__ == "__main__":
    docs = load_pdfs(PDF_FOLDER)

    chunked = chunk_documents(docs)

    print(f"\nTotal chunks created: {len(chunked)}")

    # Only test first 2 chunks to save cost
    sample_chunks = chunked[:2]

    embedded = embed_chunks(sample_chunks)

    print("\nEmbedding vector length:")
    print(len(embedded[0]["embedding"]))





from rag.chunking import chunk_documents
from rag.embedding import embed_chunks, generate_embedding
from rag.vector_store import build_faiss_index, search_index

if __name__ == "__main__":
    docs = load_pdfs(PDF_FOLDER)

    chunked = chunk_documents(docs)

    print(f"\nTotal chunks created: {len(chunked)}")

    # Embed all chunks (do once)
    print("\nGenerating embeddings...")
    embedded_chunks = embed_chunks(chunked)

    print("Building FAISS index...")
    index = build_faiss_index(embedded_chunks)

    # Test query
    query = "When is enhanced due diligence required?"
    query_vector = generate_embedding(query)

    results = search_index(index, query_vector, embedded_chunks, top_k=3)

    print("\nTop Results:\n")
    for r in results:
        print("Domain:", r["domain"])
        print("Source:", r["source"])
        print(r["text"][:300])
        print("-" * 50)
