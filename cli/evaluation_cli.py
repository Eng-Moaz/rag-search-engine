import argparse
from lib.evaluation_cli_commands import EvaluationCliCommands

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    CLI = EvaluationCliCommands()

    args = parser.parse_args()
    limit = args.limit
    CLI.precision_at_k(limit)


if __name__ == "__main__":
    main()
