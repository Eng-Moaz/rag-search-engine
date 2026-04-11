import argparse
from lib.semantic_cli_commands import SemanticCliCommands


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies that the embedding model is loaded successfully")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a given text using an embedding model")
    embed_text_parser.add_argument("text", type=str, help="The intended text to be embedded")

    subparsers.add_parser("verify_embeddings", help="Create or load embeddings and verify them")

    embedquery = subparsers.add_parser("embedquery", help="Embeds the given query")
    embedquery.add_argument("query", type=str, help="Query to be embedded")

    search = subparsers.add_parser("search", help="Uses semantic search to retrieve documents")
    search.add_argument("query", type=str, help="Query to be searched for")
    search.add_argument("--limit", type=int, default=5, help="Limit for retrieved documents")

    chunk = subparsers.add_parser("chunk", help="Chunk the documents")
    chunk.add_argument("text", type=str, help="text to be chunked")
    chunk.add_argument("--chunk-size", type=int, default=200, help="Size of the chunked documents")
    chunk.add_argument("--overlap", type=int, default=0, help="Size of the overlapped tokens")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Semantically chunk the documents")
    semantic_chunk.add_argument("text", type=str, help="text to be chunked")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="Size of the chunked documents")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="Size of the overlapped tokens")

    subparsers.add_parser("embed_chunks", help="Embeds the chunks of the documents")

    search_chunked = subparsers.add_parser("search_chunked", help="Search for documents that are chunked semantically")
    search_chunked.add_argument("query", type=str, help="Query to be searched for")
    search_chunked.add_argument("--limit", type=int, default=5, help="Limit for retrieved documents")


    args = parser.parse_args()

    CLI = SemanticCliCommands()

    match args.command:
        case "verify":
            CLI.verify_command()

        case "embed_text":
            CLI.embed_text_command(args.text)

        case "verify_embeddings":
            CLI.verify_embeddings_command()

        case "embedquery":
            CLI.embed_query_command(args.query)

        case "search":
            CLI.search_command(args.query, args.limit)

        case "chunk":
            CLI.chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            CLI.semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            CLI.embed_chunks_command()

        case "search_chunked":
            CLI.search_chunked(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
