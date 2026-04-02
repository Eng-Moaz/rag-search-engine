import argparse
from lib.cli_commands import CliCommands
from lib import inverted_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index and save it to the disk")

    term_frequency = subparsers.add_parser("tf", help="term frequency")
    term_frequency.add_argument("doc_id", type=int, help="document ID")
    term_frequency.add_argument("term", type=str, help="term searched for")

    idf = subparsers.add_parser("idf", help="Inverse document frequency")
    idf.add_argument("term", type=str)

    tfidf = subparsers.add_parser("tfidf", help="TF-IDF")
    tfidf.add_argument("doc_id", type=int, help="document ID")
    tfidf.add_argument("term", type=str)

    bm25_idf = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=inverted_index.BM25_K1, help="Tunable BM25 K1 parameter")

    args = parser.parse_args()
    CLI = CliCommands()
    match args.command:
        case "search":
            CLI.search_command(args.query)

        case "build":
            CLI.build_command()

        case "tf":
            CLI.tf_command(args.doc_id,args.term)

        case "idf":
            CLI.idf_command(args.term)

        case "tfidf":
            CLI.tfidf_command(args.doc_id, args.term)

        case "bm25idf":
            bm25idf = CLI.bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = CLI.bm25_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()