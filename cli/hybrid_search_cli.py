import argparse
from lib.hybrid_cli_commands import HybridCliCommands

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("normalize", help="normalizes a given list")
    search_parser.add_argument("scores", type=float, nargs="+", help="list to be normalized")

    args = parser.parse_args()

    CLI = HybridCliCommands()

    match args.command:
        case "normalize":
            CLI.normalize(args.scores)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()