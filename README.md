# RAG Search Engine

A search engine over a movie dataset that implements keyword-based and semantic search, each exposed through its own CLI.

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
- `sentence-transformers` for embedding models

## Data

The project expects a `data/movies.json` file containing a JSON object with a `"movies"` key. Each movie has an `id`, `title`, and `description`. A `data/stopwords.txt` file is also used for filtering common words during tokenization.

Both files are gitignored, so you will need to provide them yourself.

## Project Structure

```
cli/
  keyword_search_cli.py    # CLI entry point for keyword search
  semantic_search_cli.py   # CLI entry point for semantic search
  lib/
    search_utils.py        # Shared utilities (loading movies, stopwords)
    keyword_search.py      # Text cleaning, tokenization, stopword removal, stemming
    inverted_index.py      # Inverted index with TF, IDF, TF-IDF, and BM25
    semantic_search.py     # Embedding generation and cosine similarity search
    keyword_cli_commands.py        # Command handlers for the keyword search CLI
    semantic_cli_commands.py  # Command handlers for the semantic search CLI
cache/                     # Serialized index and embeddings (gitignored)
data/                      # Movie dataset and stopwords (gitignored)
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
```
