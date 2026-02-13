import pickle
from ml.query_classifier import QueryClassifier

EMBEDDING_PATH = "data/embeddings.pkl"
MODEL_PATH = "ml/classifier.pkl"

def main():
    print("Loading embedded chunks...")
    with open(EMBEDDING_PATH, "rb") as f:
        embedded_chunks = pickle.load(f)

    print("Training query classifier...")
    classifier = QueryClassifier()
    classifier.train(embedded_chunks)

    print("Saving classifier model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(classifier, f)

    print("Classifier saved successfully at:", MODEL_PATH)

if __name__ == "__main__":
    main()
