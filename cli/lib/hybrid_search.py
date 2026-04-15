import os
from collections import defaultdict
from .search_utils import load_movies

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

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

class HybridSearch:
    def __init__(self):
        self.documents = load_movies()
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(self.documents)

        self.idx = InvertedIndex()
        if not os.path.exists(inverted_index.CACHE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        results_bm25 = self._bm25_search(query, 500 * limit)
        results_semantic = self.semantic_search.search_chunks(query, 500 * limit)

        bm25_scores = [r["score"] for r in results_bm25]
        norm_bm25 = normalize(bm25_scores) or []

        semantic_scores = [r["score"] for r in results_semantic]
        norm_semantic = normalize(semantic_scores) or []

        results_combined = defaultdict(lambda: {"BM25": 0.0, "Semantic": 0.0, "description": "", "title":""})

        for result, score in zip(results_bm25, norm_bm25):
            doc_id = result["id"]
            results_combined[doc_id]["BM25"] = score
            results_combined[doc_id]["description"] = result["document"].get("description", "")
            results_combined[doc_id]["title"] = result["document"].get("title", "")

        for result, score in zip(results_semantic, norm_semantic):
            doc_id = result["id"]
            results_combined[doc_id]["Semantic"] = score
            if not results_combined[doc_id]["description"]:
                results_combined[doc_id]["description"] = result["document"]
            if not results_combined[doc_id]["title"]:
                results_combined[doc_id]["title"] = result["title"]

        final_results = []
        for doc_id, scores in results_combined.items():
            h_score = hybrid_score(scores["BM25"], scores["Semantic"], alpha)
            final_results.append({
                "id": doc_id,
                "title": scores["title"],
                "Hybrid": h_score,
                "BM25": scores["BM25"],
                "Semantic": scores["Semantic"],
                "description": scores["description"][:100]
            })

        final_results.sort(key=lambda x: x["Hybrid"], reverse=True)
        return final_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")