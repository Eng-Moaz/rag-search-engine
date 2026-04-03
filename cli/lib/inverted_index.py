from .keyword_search import tokenize, remove_stopwords
from .search_utils import load_movies
from collections import defaultdict, Counter
import pickle
import os
import math

BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT,"cache")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

    def __add_document(self, doc_id, text) -> None:
        tokens = remove_stopwords(tokenize(text))
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term) -> list[int]:
        ids = sorted(self.index[term.lower()])
        return ids

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"],text)
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(os.path.join(CACHE_PATH,"index.pkl"),"wb") as file:
            pickle.dump(self.index,file)
        with open(os.path.join(CACHE_PATH,"docmap.pkl"),"wb") as file:
            pickle.dump(self.docmap,file)
        with open(os.path.join(CACHE_PATH,"term_frequencies.pkl"),"wb") as file:
            pickle.dump(self.term_frequencies,file)
        with open(os.path.join(CACHE_PATH,"doc_lengths.pkl"),"wb") as file:
            pickle.dump(self.doc_lengths,file)

    def load(self) -> None:
        with open(os.path.join(CACHE_PATH,"index.pkl"),"rb") as file:
            self.index = pickle.load(file)
        with open(os.path.join(CACHE_PATH,"docmap.pkl"), "rb") as file:
            self.docmap = pickle.load(file)
        with open(os.path.join(CACHE_PATH,"term_frequencies.pkl"), "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(os.path.join(CACHE_PATH,"doc_lengths.pkl"), "rb") as file:
            self.doc_lengths = pickle.load(file)

    def retrieve(self, query: str, num_limit=5) -> list[dict]:
        indices = []
        result = []
        tokenized_query = remove_stopwords(tokenize(query))
        for query_token in tokenized_query:
            token_indices = self.get_documents(query_token)
            for token_index in token_indices:
                if token_index not in indices:
                    indices.append(token_index)
                if len(indices) == num_limit:
                    break

        for index in indices:
            result.append(self.docmap[index])
        return result

    def get_tf(self,doc_id,term) -> int:
        tokenized_term = remove_stopwords(tokenize(term))
        if len(tokenized_term) != 1:
            raise ValueError("Term must be a single token after processing.")
        return self.term_frequencies[doc_id][tokenized_term[0]]

    def get_idf(self,term) -> float:
        term = remove_stopwords(tokenize(term))[0]
        total_docs, term_docs = len(self.docmap), len(self.get_documents(term))
        idf = math.log((total_docs + 1) / (term_docs + 1))
        return idf

    def get_tf_idf(self,doc_id,term) -> float:
        return self.get_idf(term)*self.get_tf(doc_id,term)

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = remove_stopwords(tokenize(term))
        if len(tokenized_term) != 1:
            raise ValueError("Term must be a single token after processing.")
        total_docs, term_docs = len(self.docmap), len(self.get_documents(tokenized_term[0]))
        bm25 = math.log((total_docs - term_docs + 0.5) / (term_docs + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id,term)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id, term) -> float:
        bm25 = self.get_bm25_tf(doc_id,term) * self.get_bm25_idf(term)
        return bm25

    def bm25_search(self, query, limit=5) -> list[tuple]:
        tokenized_query = remove_stopwords(tokenize(query))
        scores = defaultdict(float)
        for query_token in tokenized_query:
            matched_docs = self.get_documents(query_token)
            for doc_id in matched_docs:
                scores[doc_id] += self.bm25(doc_id, query_token)
        sorted_docs = sorted(scores.items(), key = lambda item: item[1], reverse=True)
        retrieved = []
        for doc_id, score in sorted_docs[:limit]:
            retrieved.append((doc_id, self.docmap[doc_id], score))
        return retrieved
