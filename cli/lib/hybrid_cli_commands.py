import json

from .hybrid_search import HybridSearch
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
import time


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

    def _call_model_gemini(self, prompt):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ]
            )
        )
        return response.text

    def _call_model_groq(self, prompt):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content

    def _enhance_spelling(self, query):
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                    Preserve punctuation and capitalization unless a change is required for a typo fix.
                    If there are no spelling errors, or if you're unsure, output the original query unchanged.
                    Output only the final query text, nothing else.
                    User query: "{query}"
                    """
        return self._call_model_groq(prompt)

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
        return self._call_model_groq(prompt)

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
        return self._call_model_groq(prompt)

    def _rerank_indv(self, query,doc):
        prompt = f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {doc.get("title", "")} - {doc.get("description", "")}
                    
                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness
                    
                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.
                    
                    Score:"""
        return self._call_model_groq(prompt)

    def _rerank_batch(self, query, docs):
        prompt = f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"
                    
                    Movies:
                    {docs}
                    
                    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.
                    
                    For example:
                    [75, 12, 34, 2, 1]
                    
                    Ranking:"""
        return self._call_model_groq(prompt)

    def rrf_search(self, query, k, limit, enhance, rerank):
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
                enhanced = self._expand(query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced

        if rerank == "individual":
            results = hybrid_search.rrf_search(query, k, limit*5)
            for doc in results:
                score = float(self._rerank_indv(query, doc).strip())
                doc |= {"rerank_score": score}
                time.sleep(3)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = results[:limit]

            for i, result in enumerate(results):
                print(f"""{i + 1}. {result['title']}
                Re-rank Score: {result["rerank_score"]}/10
                RRF Score: {result['rrf']}
                BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
                {result['description'][:100]}""")

        elif rerank == "batch":
            results = hybrid_search.rrf_search(query, k, limit * 5)
            doc_list_str = ""
            for i, result in enumerate(results):
                doc_list_str += f"""
                    {i+1}. ID : {result['id']}
                           Title : {result['title']}
                           Description : {result['description'][:200]}
                    \n\n
                    """
            scores = self._rerank_batch(query,doc_list_str)
            ranked_ids = json.loads(scores)

            results_by_id = {str(doc['id']): doc for doc in results}
            reranked_results = []

            for rank_pos, doc_id in enumerate(ranked_ids):
                doc_id_str = str(doc_id)
                if doc_id_str in results_by_id:
                    doc = results_by_id.pop(doc_id_str)
                    doc["rerank_score"] = rank_pos + 1
                    reranked_results.append(doc)

            results = reranked_results[:limit]

            for i, result in enumerate(results):
                print(f"""{i + 1}. {result['title']}
                Re-rank Rank: {result.get("rerank_score", 0)}
                RRF Score: {result['rrf']}
                BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
                {result['description'][:100]}""")

        else:
            results = hybrid_search.rrf_search(query, k, limit)

            for i,result in enumerate(results):
                print(f"""{i+1}. {result['title']}
                RRF Score: {result['rrf']}
                BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
                {result['description'][:100]}""")

    def individual_reranking(self,query,k,limit,rerank):
        hybrid_search = HybridSearch()
        results = hybrid_search.rrf_search(query, k, limit*5)

        if rerank == "individual":
            for doc in results:
                score = self._rerank_indv(query,doc)
                doc |= {"rerank_score":score}
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = results[:limit]

        for i,result in enumerate(results):
            print(f"""{i+1}. {result['title']}
            Re-rank Score: {result["rerank_score"]}/10
            RRF Score: {result['rrf']}
            BM25 Rank: {result['BM25']}, Semantic Rank: {result['Semantic']}
            {result['description'][:100]}""")