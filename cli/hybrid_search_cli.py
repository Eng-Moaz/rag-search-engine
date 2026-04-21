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

    rrf_search = subparsers.add_parser("rrf-search", help="Performs a rrf-search")
    rrf_search.add_argument("query", type=str, help="Query to be searched for")
    rrf_search.add_argument("--k", type=float, help="K parameter for rrf control", default=60)
    rrf_search.add_argument("--limit", type=int, help="limit for the returned results", default=5)
    rrf_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Technique for reranking"
        )
    rrf_search.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the results to the given query",
    )


    args = parser.parse_args()

    CLI = HybridCliCommands()

    match args.command:
        case "normalize":
            CLI.normalize(args.scores)

        case "weighted-search":
            CLI.weighted_search(args.query, args.alpha, args.limit)

        case "rrf-search":
            CLI.rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()