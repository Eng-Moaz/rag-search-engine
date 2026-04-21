import argparse
from lib.augmented_cli_commands import AugmentedCliCommands


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, default=5, help="limit for the retrieved docs")

    args = parser.parse_args()

    CLI = AugmentedCliCommands()

    match args.command:
        case "rag":
            query = args.query
            CLI.augmented_gen(query)

        case "summarize":


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
