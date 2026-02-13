import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class QueryClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)

    def train(self, chunked_docs):
        texts = [doc["text"] for doc in chunked_docs]
        labels = [doc["domain"] for doc in chunked_docs]

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, query):
        X_query = self.vectorizer.transform([query])
        return self.model.predict(X_query)[0]
