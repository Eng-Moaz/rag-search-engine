import argparse
import sys
from lib.inverted_index import InvertedIndex

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

    idf = subparsers.add_parser("tfidf", help="TF-IDF")
    idf.add_argument("doc_id", type=int, help="document ID")
    idf.add_argument("term", type=str)

    args = parser.parse_args()

    def _load_inverted_index():
        inverted_index = InvertedIndex()
        try:
            inverted_index.load()
        except FileNotFoundError:
            print("File not found, Run the build command first")
            sys.exit(1)
        return inverted_index

    match args.command:
        case "search":
            inverted_index = _load_inverted_index()
            retrieved = inverted_index.retrieve(args.query)
            for movie in retrieved:
                print(f"{movie['title']} ({movie['id']})")

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

            docs = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

        case "tf":
            inverted_index = _load_inverted_index()
            print(inverted_index.get_tf(args.doc_id,args.term))

        case "idf":
            inverted_index = _load_inverted_index()
            idf = inverted_index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            inverted_index = _load_inverted_index()
            tf_idf = inverted_index.get_tf_idf(args.doc_id,args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()