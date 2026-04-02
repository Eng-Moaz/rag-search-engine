import sys
from .inverted_index import InvertedIndex

class CliCommands:

    def _load_inverted_index(self):
        inverted_index = InvertedIndex()
        try:
            inverted_index.load()
        except FileNotFoundError:
            print("File not found, Run the build command first")
            sys.exit(1)
        return inverted_index

    def search_command(self,query):
        inverted_index = self._load_inverted_index()
        retrieved = inverted_index.retrieve(query)
        for movie in retrieved:
            print(f"{movie['title']} ({movie['id']})")

    def build_command(self):
        inverted_index = InvertedIndex()
        inverted_index.build()
        inverted_index.save()

        docs = inverted_index.get_documents("merida")
        print(f"First document for token 'merida' = {docs[0]}")

    def tf_command(self,doc_id,term):
        inverted_index = self._load_inverted_index()
        print(inverted_index.get_tf(doc_id, term))

    def idf_command(self,term):
        inverted_index = self._load_inverted_index()
        idf = inverted_index.get_idf(term)
        print(f"Inverse document frequency of '{term}': {idf:.2f}")

    def tfidf_command(self,doc_id,term):
        inverted_index = self._load_inverted_index()
        tf_idf = inverted_index.get_tf_idf(doc_id, term)
        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

    def bm25_idf_command(self,term):
        inverted_index = self._load_inverted_index()
        return inverted_index.get_bm25_idf(term)