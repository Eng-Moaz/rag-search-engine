import argparse
import sys
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Builds the inverted index and save it to the disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError:
                print("File not found, Run the build command first")
                sys.exit(1)
            retrieved = inverted_index.retrieve(args.query)
            for movie in retrieved:
                print(f"{movie['title']} ({movie['id']})")

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

            docs = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()