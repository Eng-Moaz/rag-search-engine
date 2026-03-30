import argparse
from lib.keyword_search import search_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_query(args.query)
            for movie in results:
                print(movie["title"])

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()