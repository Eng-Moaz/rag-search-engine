from .keyword_search import tokenize
from .search_utils import load_movies
from collections import defaultdict
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
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
