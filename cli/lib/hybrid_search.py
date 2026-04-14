import os

from . import inverted_index
from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize(scores):
    if not scores or len(scores) == 0:
        return
    smallest, largest = min(scores), max(scores)
    if smallest == largest:
        final = [1.0] * len(scores)
    else:
        final = [(score - smallest) / (largest - smallest) for score in scores]
    return final


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(inverted_index.CACHE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        results_bm25 = self._bm25_search(query, 500*limit)
        results_semantic = self.semantic_search.search_chunks(query,500*limit)
        results_combined = {}

        bm25_scores = [score for doc_id, document, score in results_bm25]
        normalized_bm25_scores = normalize(bm25_scores)

        semantic_scores = [result["score"] for result in results_semantic]
        normalized_semantic_scores = normalize(semantic_scores)

        for i in range(limit * 500):
            doc_id, document, score = results_bm25[i]
            results_combined[doc_id] = {
                ""
            }


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

