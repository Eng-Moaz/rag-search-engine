import argparse
from lib.hybrid_cli_commands import HybridCliCommands

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("normalize", help="normalizes a given list")
    search_parser.add_argument("scores", type=float, nargs="+", help="list to be normalized")

    weighted_search = subparsers.add_parser("weighted-search", help="Performs a weighted-search")
    weighted_search.add_argument("query", type=str, help="Query to be searched for")
    weighted_search.add_argument("--alpha", type=float, help="alpha for the weighted-search formula", default=0.5)
    weighted_search.add_argument("--limit", type=int, help="limit for the returned results", default=5)


    args = parser.parse_args()

    CLI = HybridCliCommands()

    match args.command:
        case "normalize":
            CLI.normalize(args.scores)

        case "weighted-search":
            CLI.weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()