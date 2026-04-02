from .keyword_search import tokenize, remove_stopwords
from .search_utils import load_movies
from collections import defaultdict, Counter
import pickle
import os
import math


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.bm25_k1 = 1.5

    def __add_document(self, doc_id, text):
        tokens = remove_stopwords(tokenize(text))
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        ids = sorted(self.index[term.lower()])
        return ids

    def build(self):
        movies = load_movies()
        for movie in movies:
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"],text)
            self.docmap[movie["id"]] = movie

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl","wb") as file:
            pickle.dump(self.index,file)
        with open("cache/docmap.pkl","wb") as file:
            pickle.dump(self.docmap,file)
        with open("cache/term_frequencies.pkl","wb") as file:
            pickle.dump(self.term_frequencies,file)

    def load(self):
        with open("cache/index.pkl","rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)

    def retrieve(self, query: str, num_limit=5):
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

    def get_tf(self,doc_id,term):
        tokenized_term = remove_stopwords(tokenize(term))
        if len(tokenized_term) != 1:
            raise ValueError("Term must be a single token after processing.")
        return self.term_frequencies[doc_id][tokenized_term[0]]

    def get_idf(self,term):
        term = remove_stopwords(tokenize(term))[0]
        total_docs, term_docs = len(self.docmap), len(self.get_documents(term))
        idf = math.log((total_docs + 1) / (term_docs + 1))
        return idf

    def get_tf_idf(self,doc_id,term):
        return self.get_idf(term)*self.get_tf(doc_id,term)

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = remove_stopwords(tokenize(term))
        if len(tokenized_term) != 1:
            raise ValueError("Term must be a single token after processing.")
        total_docs, term_docs = len(self.docmap), len(self.get_documents(tokenized_term[0]))
        bm25 = math.log((total_docs - term_docs + 0.5) / (term_docs + 0.5) + 1)
        return bm25
