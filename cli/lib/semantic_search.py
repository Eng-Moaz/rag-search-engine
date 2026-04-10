import json
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = CACHE_DIR / "movie_embeddings.npy"

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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunks_path = CACHE_DIR / "chunk_embeddings.npy"
        self.metadata_path = CACHE_DIR / "chunk_metadata.json"

    def _semantic_chunking(self, text, max_size_chunk, overlap):
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text.strip())
        step_size = max_size_chunk - overlap
        chunks = []

        for i in range(0, len(sentences), step_size):
            chunk_sentences = sentences[i: i + max_size_chunk]
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)
            if i + max_size_chunk >= len(sentences):
                break
        return chunks

    def build_chunk_embeddings(self, documents, max_size_chunk=4, overlap=1):
        self.documents = documents
        self.document_map = {}
        for document in self.documents:
            self.document_map[document["id"]] = document

        all_chunks = []
        chunks_metadata = []
        for document in self.documents:
            text = document["description"]
            if text is None or text == "":
                pass
            else:
                chunks = self._semantic_chunking(text, max_size_chunk, overlap)
                all_chunks.extend(chunks)
                for i, chunk in enumerate(chunks):
                    chunks_metadata.append({"movie_idx":document["id"], "chunk_idx":i, "total_chunks":len(chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunks_metadata

        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.chunks_path, self.chunk_embeddings)
        with open(self.metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for document in self.documents:
            self.document_map[document["id"]] = document

        if self.chunks_path.exists() and self.metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunks_path)
            with open(self.metadata_path,"r") as f:
              self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
