from .keyword_search import tokenize, remove_stopwords
from .search_utils import load_movies
from collections import defaultdict
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = remove_stopwords(tokenize(text))
        for token in tokens:
            self.index[token].add(doc_id)

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

    def load(self):
        with open("cache/index.pkl","rb") as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

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