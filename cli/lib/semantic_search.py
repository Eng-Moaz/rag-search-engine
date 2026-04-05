from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from .search_utils import load_movies


def verify_model():
    semantic_search = SemanticSsearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSsearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSsearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


class SemanticSsearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = Path("cache/movie_embeddings.npy")

    def generate_embedding(self, text):
        if text == "" or text is None:
            raise ValueError("The text is empty or doesn't exist")
        encoded_text = self.model.encode([text])
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