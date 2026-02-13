import pickle
from injestion.injest import load_pdfs, PDF_FOLDER
from rag.chunking import chunk_documents
from rag.embedding import embed_chunks

OUTPUT_PATH = "data/embeddings.pkl"

def main():
    print("Loading PDFs...")
    docs = load_pdfs(PDF_FOLDER)

    print("Chunking...")
    chunked = chunk_documents(docs)

    print(f"Total chunks: {len(chunked)}")

    print("Generating embeddings (one-time)...")
    embedded_chunks = embed_chunks(chunked)

    print("Saving embeddings...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(embedded_chunks, f)

    print("Embeddings saved successfully!")

if __name__ == "__main__":
    main()
