from .semantic_search import SemanticSearch, ChunkedSemanticSearch
from .search_utils import load_movies
import re


class SemanticCliCommands:

    def _load_semantic_search(self):
        semantic_search = SemanticSearch()
        documents = load_movies()
        semantic_search.load_or_create_embeddings(documents)
        return semantic_search

    def verify_command(self):
        semantic_search = SemanticSearch()
        print(f"Model loaded: {semantic_search.model}")
        print(f"Max sequence length: {semantic_search.model.max_seq_length}")

    def embed_text_command(self, text):
        semantic_search = SemanticSearch()
        embedding = semantic_search.generate_embedding(text)
        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")

    def verify_embeddings_command(self):
        semantic_search = SemanticSearch()
        documents = load_movies()
        embeddings = semantic_search.load_or_create_embeddings(documents)
        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

    def embed_query_command(self, query):
        semantic_search = SemanticSearch()
        embedding = semantic_search.generate_embedding(query)
        print(f"Query: {query}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Shape: {embedding.shape}")

    def search_command(self, query, limit=5):
        semantic_search = self._load_semantic_search()
        results = semantic_search.search(query, limit)
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} (score: {result['score']:.4f})")
            print(f"{result['description'][:100]}")

    def chunk_command(self, text, chunk_size, overlap):
        words = text.split()
        step_size = chunk_size - overlap
        chunks = []
        for i in range(0,len(words),step_size):
           chunk_words = words[i:chunk_size+i]
           if len(chunk_words) >= overlap:
               break
           chunk = " ".join(chunk_words)
           chunks.append(chunk)

        print(f"Chunking {len(text)} characters")
        for i, chunk in enumerate(chunks):
            print(f"{i+1}. {chunk}")

    def semantic_chunk_command(self, text, max_size_chunk, overlap):
        pattern =  r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text.strip())
        step_size = max_size_chunk - overlap
        chunks = []

        for i in range(0, len(sentences), step_size):
            chunk_sentences = sentences[i: i + max_size_chunk]
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)
            if i + max_size_chunk >= len(sentences):
                break

        print(f"Semantically chunking {len(text)} characters")
        for i, chunk in enumerate(chunks):
            print(f"{i + 1}. {chunk}")

    def embed_chunks_command(self):
        movies = load_movies()
        chunked_semantic_search = ChunkedSemanticSearch()
        embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
        print(f"Generated {len(embeddings)} chunked embeddings")
