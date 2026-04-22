# RAG Search Engine

A search engine over a movie dataset that implements keyword-based, semantic, hybrid, and multimodal search, along with retrieval-augmented generation (RAG). Each feature is exposed through its own CLI.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (for dependency management)

## Setup

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml`:

- `nltk` for stemming and text processing
- `numpy` for vector operations
- `sentence-transformers` for embedding models and cross-encoder reranking
- `google-genai` for Gemini API access (image description, query enhancement)
- `groq` for Groq API access (query enhancement, reranking, RAG generation)
- `pillow` for image loading (multimodal search)
- `python-dotenv` for loading environment variables

### Environment Variables

Create a `.env` file in the project root with the following keys:

```
GOOGLE_API_KEY=<your Google Gemini API key>
GROQ_API_KEY=<your Groq API key>
```

These are required for query enhancement, reranking, RAG generation, and image description features.

## Data

The project expects a `data/movies.json` file containing a JSON object with a `"movies"` key. Each movie has an `id`, `title`, and `description`. A `data/stopwords.txt` file is also used for filtering common words during tokenization.

For evaluation, a `data/golden_dataset.json` file provides test cases with queries and their expected relevant documents.

Both `data/` and `cache/` directories are gitignored, so you will need to provide the data files yourself.

## Project Structure

```
cli/
  keyword_search_cli.py        # CLI entry point for keyword search
  semantic_search_cli.py       # CLI entry point for semantic search
  hybrid_search_cli.py         # CLI entry point for hybrid search
  augmented_generation_cli.py  # CLI entry point for RAG (retrieval-augmented generation)
  multimodal_search_cli.py     # CLI entry point for multimodal (image) search
  describe_image_cli.py        # CLI entry point for image-based query rewriting
  evaluation_cli.py            # CLI entry point for search evaluation metrics
  test_gemini.py               # Quick smoke test for Gemini API connectivity
  lib/
    search_utils.py            # Shared utilities (loading movies, stopwords, golden dataset)
    keyword_search.py          # Text cleaning, tokenization, stopword removal, stemming
    inverted_index.py          # Inverted index with TF, IDF, TF-IDF, and BM25
    semantic_search.py         # Embedding generation and cosine similarity search
    hybrid_search.py           # Hybrid search utilities (normalization, weighted search, RRF)
    multimodal_search.py       # CLIP-based image embedding and cross-modal search
    keyword_cli_commands.py    # Command handlers for the keyword search CLI
    semantic_cli_commands.py   # Command handlers for the semantic search CLI
    hybrid_cli_commands.py     # Command handlers for the hybrid search CLI (includes reranking, enhancement, evaluation)
    augmented_cli_commands.py  # Command handlers for the RAG CLI
    evaluation_cli_commands.py # Command handlers for the evaluation CLI
cache/                         # Serialized index and embeddings (gitignored)
data/                          # Movie dataset, stopwords, and golden dataset (gitignored)
logs/                          # Search logs in JSON format (generated at runtime)
```

## Keyword Search

The keyword search pipeline works as follows:

1. Text is cleaned (lowercased, punctuation removed), tokenized, filtered for stopwords, and stemmed using the Porter stemmer.
2. An inverted index maps each stemmed token to the set of document IDs that contain it.
3. Term frequencies and document lengths are tracked per document.
4. Retrieval can be done with basic token matching or with BM25 ranking.

### Building the index

Before searching, you need to build and cache the inverted index:

```bash
python cli/keyword_search_cli.py build
```

This reads every movie from the dataset, tokenizes the title and description, and saves the resulting index, document map, term frequencies, and document lengths as pickle files under `cache/`.

### CLI commands

```bash
# Basic token-matching search (returns first 5 matches)
python cli/keyword_search_cli.py search "brave princess"

# Term frequency of a term in a specific document
python cli/keyword_search_cli.py tf 123 "dragon"

# Inverse document frequency of a term
python cli/keyword_search_cli.py idf "dragon"

# TF-IDF score for a term in a document
python cli/keyword_search_cli.py tfidf 123 "dragon"

# BM25 IDF score for a term
python cli/keyword_search_cli.py bm25idf "dragon"

# BM25 TF score (with optional k1 and b parameters, defaults: k1=1.5, b=0.75)
python cli/keyword_search_cli.py bm25tf 123 "dragon"
python cli/keyword_search_cli.py bm25tf 123 "dragon" 1.2 0.8

# Full BM25 ranked search
python cli/keyword_search_cli.py bm25search "epic adventure"
```

## Semantic Search

The semantic search pipeline uses sentence-transformers (`all-MiniLM-L6-v2`) to encode movie titles and descriptions into dense vectors, then ranks results by cosine similarity against a query embedding.

### How it works

1. Each movie's title and description are concatenated and encoded into a 384-dimensional vector.
2. Embeddings are saved to `cache/movie_embeddings.npy`. On subsequent runs, they are loaded from cache unless the document count has changed.
3. At query time, the query is encoded with the same model and compared against all document embeddings using cosine similarity.

### CLI commands

```bash
# Verify the embedding model loaded correctly
python cli/semantic_search_cli.py verify

# Embed a piece of text and print its shape
python cli/semantic_search_cli.py embed_text "a story about robots"

# Build or load embeddings and print summary info
python cli/semantic_search_cli.py verify_embeddings

# Embed a query and print the vector shape
python cli/semantic_search_cli.py embedquery "space adventure"

# Semantic search with default top-5 results
python cli/semantic_search_cli.py search "romantic comedy set in paris"

# Semantic search with a custom result limit
python cli/semantic_search_cli.py search "romantic comedy set in paris" --limit 10

# Chunk a piece of text into fixed-size segments
python cli/semantic_search_cli.py chunk "long text goes here" --chunk-size 100 --overlap 20

# Semantically chunk a piece of text based on sequence
python cli/semantic_search_cli.py semantic_chunk "long text goes here" --max-chunk-size 4

# Embed all document chunks
python cli/semantic_search_cli.py embed_chunks

# Search for documents using chunk-level embeddings
python cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 5
```

## Hybrid Search

The hybrid search combines results from both keyword and semantic searches to provide better overall retrieval, supporting operations like normalization, weighted average, and Reciprocal Rank Fusion (RRF).

### CLI commands

```bash
# Normalize a list of scores
python cli/hybrid_search_cli.py normalize 0.5 0.8 1.2

# Weighted hybrid search (adjust keyword/semantic bias using --alpha, default 0.5)
python cli/hybrid_search_cli.py weighted-search "epic adventure" --alpha 0.7 --limit 5

# Reciprocal Rank Fusion (RRF) search (adjust ranking penalty using --k, default 60.0)
python cli/hybrid_search_cli.py rrf-search "space movie" --k 60 --limit 5
```

### Query Enhancement

RRF search supports LLM-powered query enhancement via the `--enhance` flag. Enhancement methods use the Groq API (`llama-3.3-70b-versatile`) to preprocess the query before searching.

```bash
# Spell-check the query before searching
python cli/hybrid_search_cli.py rrf-search "space moive" --enhance spell

# Rewrite the query to be more specific and searchable
python cli/hybrid_search_cli.py rrf-search "that bear movie in london" --enhance rewrite

# Expand the query with synonyms and related terms
python cli/hybrid_search_cli.py rrf-search "scary bear" --enhance expand
```

### Reranking

RRF search supports post-retrieval reranking via the `--rerank-method` flag. When reranking is enabled, more candidates are fetched (5× the limit) and then re-scored.

```bash
# Individual LLM-based reranking (scores each doc 0-10, slower but granular)
python cli/hybrid_search_cli.py rrf-search "epic adventure" --rerank-method individual --limit 5

# Batch LLM-based reranking (ranks all docs in one call, faster)
python cli/hybrid_search_cli.py rrf-search "epic adventure" --rerank-method batch --limit 5

# Cross-encoder reranking (uses cross-encoder/ms-marco-TinyBERT-L2-v2, no API needed)
python cli/hybrid_search_cli.py rrf-search "epic adventure" --rerank-method cross_encoder --limit 5
```

### LLM Evaluation

Append `--evaluate` to any RRF search to have the LLM rate each result's relevance on a 0–3 scale.

```bash
python cli/hybrid_search_cli.py rrf-search "space movie" --evaluate
```

### Logging

Every RRF search run writes a JSON log file to `logs/` with the query, enhancement method, reranking method, and all intermediate and final results.

## Augmented Generation (RAG)

The RAG CLI retrieves relevant documents using hybrid RRF search and then generates natural-language answers via the Groq API (`llama-3.3-70b-versatile`). All responses are tailored to Hoopla, a movie streaming service.

### CLI commands

```bash
# Basic RAG: retrieve documents and generate a comprehensive answer
python cli/augmented_generation_cli.py rag "what are some good space movies?"

# Summarize: synthesize information from multiple search results
python cli/augmented_generation_cli.py summarize "animated bear movies" --limit 10

# Citations: answer with inline source citations ([1], [2], etc.)
python cli/augmented_generation_cli.py citations "romantic comedy recommendations" --limit 5

# Question: casual, conversational answer to a question
python cli/augmented_generation_cli.py question "what's a good movie for kids?" --limit 5
```

## Multimodal Search

The multimodal search CLI uses a CLIP model (`clip-ViT-B-32`) to embed images and text into a shared vector space, enabling image-to-text search over the movie dataset.

### How it works

1. All movie titles and descriptions are encoded into CLIP text embeddings.
2. A query image is encoded into a CLIP image embedding.
3. Results are ranked by cosine similarity between the image embedding and all text embeddings.

### CLI commands

```bash
# Verify that an image can be embedded and print the embedding shape
python cli/multimodal_search_cli.py verify_image_embedding path/to/image.jpg

# Search for movies similar to a given image
python cli/multimodal_search_cli.py image_search path/to/image.jpg
```

## Image-Based Query Rewriting

The image description CLI uses the Gemini API (`gemini-2.5-flash`) to rewrite a text query based on visual information from an image. This synthesizes visual and textual cues to produce a more targeted movie search query.

### CLI commands

```bash
# Rewrite a query informed by an image (uses default image if --image is omitted)
python cli/describe_image_cli.py --query "movies like this"

# Rewrite a query using a specific image
python cli/describe_image_cli.py --image path/to/image.jpg --query "find movies similar to this scene"
```

## Search Evaluation

The evaluation CLI measures retrieval quality against a golden dataset (`data/golden_dataset.json`). It runs hybrid RRF search for each test query and reports precision@k, recall@k, and F1 score.

### CLI commands

```bash
# Evaluate with default k=5
python cli/evaluation_cli.py

# Evaluate with a custom k
python cli/evaluation_cli.py --limit 10
```
