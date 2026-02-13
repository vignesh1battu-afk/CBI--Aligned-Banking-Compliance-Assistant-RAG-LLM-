import faiss
import numpy as np


def build_faiss_index(chunked_docs):
    embeddings = [chunk["embedding"] for chunk in chunked_docs]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)

    vectors = np.array(embeddings).astype("float32")

    # Normalize vectors
    faiss.normalize_L2(vectors)

    index.add(vectors)

    return index


def search_index(index, query_vector, chunked_docs, top_k=5):
    query_vector = np.array([query_vector]).astype("float32")

    # Normalize query
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunked_docs[idx])

    return results
