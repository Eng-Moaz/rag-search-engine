from .search_utils import load_golden
from .hybrid_search import  HybridSearch

K = 60

class EvaluationCliCommands:
    def __init__(self):
        self.golden_dataset = load_golden()
        self.hybrid_search = HybridSearch()

    def _precision(self, relevant, total):
        return relevant / total

    def precision_at_k(self, limit):
        print(f"k = {K}")

        for example in self.golden_dataset:
            query = example["query"]
            relevant_docs = example["relevant_docs"]
            results = self.hybrid_search.rrf_search(query,K,limit)
            retrieved = [result['title'] for result in results]
            relevant_num = 0
            for rel_doc in relevant_docs:
                if rel_doc in retrieved:
                    relevant_num += 1
            precision = self._precision(relevant_num, len(retrieved))

            print(f"""
                - Query: {query}
                    - Precision@{limit}: {precision:.4f}
                    - Retrieved: {retrieved}
                    - Relevant: {relevant_docs}
            """)

