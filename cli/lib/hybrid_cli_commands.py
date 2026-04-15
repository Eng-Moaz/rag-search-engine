from .hybrid_search import HybridSearch

class HybridCliCommands:
    def normalize(self, scores):
        if not scores or len(scores) == 0:
            return
        smallest, largest = min(scores), max(scores)
        if smallest == largest:
            final = [1.0] * len(scores)
        else:
            final = [(score-smallest)/(largest-smallest) for score in scores]

        for score in final:
            print(f"{score:.4f}")

    def weighted_search(self, query, alpha, limit):
        hybrid_search = HybridSearch()
        results = hybrid_search.weighted_search(query, alpha, limit)

        for i,result in enumerate(results):
            print(f"""{i+1}. {result['title']}
            Hybrid Score: {result['Hybrid']}
            BM25: {result['BM25']}, Semantic: {result['Semantic']}
            {result['description']}""")

    def rrf_search(self, query, k, limit):
        hybrid_search = HybridSearch()
        results = hybrid_search.rrf_search(query, k, limit)

        for i,result in enumerate(results):
            print(f"""{i+1}. {result['title']}
            RRF Score: {result['rrf']}
            BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
            {result['description']}""")