from .hybrid_search import HybridSearch
import os
from dotenv import load_dotenv
from google import genai


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

    def _call_model(self, prompt):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        return response.text

    def _enhance_spelling(self, query):
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                    Preserve punctuation and capitalization unless a change is required for a typo fix.
                    If there are no spelling errors, or if you're unsure, output the original query unchanged.
                    Output only the final query text, nothing else.
                    User query: "{query}"
                    """
        return self._call_model(prompt)

    def rrf_search(self, query, k, limit, enhance):
        hybrid_search = HybridSearch()
        if enhance == "spell":
            enhanced = self._enhance_spelling(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced}'\n")
            query = enhanced
        results = hybrid_search.rrf_search(query, k, limit)

        for i,result in enumerate(results):
            print(f"""{i+1}. {result['title']}
            RRF Score: {result['rrf']}
            BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
            {result['description']}""")

