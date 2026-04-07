from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = Path("cache/movie_embeddings.npy")

    def generate_embedding(self, text):
        if text is None:
            raise ValueError("The text doesn't exist")
        clean_text = text.strip()
        if clean_text == "":
            raise ValueError("The text is empty")
        encoded_text = self.model.encode([clean_text])
        return encoded_text[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map= {}
        movie_strings = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            movie_strings.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for document in self.documents:
            self.document_map[document["id"]] = document

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
            return self.build_embeddings(documents)
        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        scores = []
        for i, document in enumerate(self.documents):
            score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((score, document))

        sorted_docs = sorted(scores, key=lambda item: item[0], reverse=True)
        final = []

        for score, doc in sorted_docs[:limit]:
            final.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
        return final