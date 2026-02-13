import pickle
from ml.query_classifier import QueryClassifier
from rag.vector_store import build_faiss_index, search_index
from rag.embedding import generate_embedding
from llm.response import generate_answer

EMBEDDING_PATH = "data/embeddings.pkl"

def main():
    print("Loading stored embeddings...")
    with open(EMBEDDING_PATH, "rb") as f:
        embedded_chunks = pickle.load(f)

    print("Building FAISS index...")
    index = build_faiss_index(embedded_chunks)

    print("Loading ML classifier...")
    with open("ml/classifier.pkl", "rb") as f:
        classifier = pickle.load(f)

    while True:
        query = input("\nEnter your compliance question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        # Predict domain
        predicted_domain = classifier.predict(query)
        print("Predicted Domain:", predicted_domain)

        # Filter by domain
        filtered_chunks = [
            doc for doc in embedded_chunks
            if doc["domain"] == predicted_domain
        ]

        # Build domain-specific FAISS
        temp_index = build_faiss_index(filtered_chunks)

        query_vector = generate_embedding(query)

        results = search_index(temp_index, query_vector, filtered_chunks, top_k=5)

        print("\nRetrieved Chunks:\n")
        for r in results:
            print("Domain:", r["domain"])
            print("Source:", r["source"])
            print(r["text"][:250])
            print("-" * 50)

        answer = generate_answer(query, results)

        print("\nFinal Grounded Answer:\n")
        print(answer)

if __name__ == "__main__":
    main()
