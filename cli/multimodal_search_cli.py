import argparse
from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="Image description")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Embed and verify an image")
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)

if __name__ == "__main__":
    main()
