from .hybrid_cli_commands import HybridCliCommands
from .hybrid_search import HybridSearch

class AugmentedCliCommands:
    def __init__(self):
        self.hybrid_commands = HybridCliCommands() # for models usage
        self.hybrid_search = HybridSearch()

    def augmented_gen(self, query):
        results = self.hybrid_search.rrf_search(query, limit=5, k=60)
        docs = [
            f"{i + 1}. Title: {doc.get('title', 'N/A')} | Description: {doc.get('description', 'N/A')[:200]}"
            for i, doc in enumerate(results)
        ]

        prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
                    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                    Provide a comprehensive answer that addresses the user's query.
                    
                    Query: {query}
                    
                    Documents:
                    {chr(10).join(docs)}
                    
                    Answer:"""

        model_answer = self.hybrid_commands._call_model_groq(prompt)

        print(f"""
            Search Results:
            {'\n-'.join(docs)}
            
            RAG Response:
            {model_answer}
            """)

        def