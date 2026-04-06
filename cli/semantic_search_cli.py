import argparse
from lib.semantic_search import (
    verify_model, embed_text, verify_embeddings, embed_query_text
    )


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="verifies that the embedding model is loaded successfully")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a given text using an embedding model")
    embed_text_parser.add_argument("text", type=str, help="The intended text to be embedded")

    subparsers.add_parser("verify_embeddings", help="create or load embeddings and verify them")

    embedquery = subparsers.add_parser("embedquery", help="embeds the given query")
    embedquery.add_argument("query", type=str, help="query to be embedded")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()