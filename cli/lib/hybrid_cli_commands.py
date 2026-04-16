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

    def _enhance_writing(self, query):
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
        
                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep the rewritten query concise (under 10 words)
                    - It should be a Google-style search query, specific enough to yield relevant results
                    - Don't use boolean logic
                    
                    Examples:
                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
                    
                    If you cannot improve the query, output the original unchanged.
                    Output only the rewritten query text, nothing else.
                    
                    User query: "{query}"
                    """
        return self._call_model(prompt)

    def _expand(self, query):
        prompt = f"""Expand the user-provided movie search query below with related terms.

                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    Output only the additional terms; they will be appended to the original query.
                    
                    Examples:
                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"
                    
                    User query: "{query}"
                    """
        return self._call_model(prompt)

    def rrf_search(self, query, k, limit, enhance):
        hybrid_search = HybridSearch()
        match enhance:
            case "spell":
                enhanced = self._enhance_spelling(query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            case "rewrite":
                enhanced = self._enhance_writing(query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            case "expand":
                enhanced = self._enhance_writing(query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced

        results = hybrid_search.rrf_search(query, k, limit)

        for i,result in enumerate(results):
            print(f"""{i+1}. {result['title']}
            RRF Score: {result['rrf']}
            BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
            {result['description']}""")

