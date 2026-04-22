import argparse
import mimetypes
import os
from google.genai import types
from google import genai
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_PATH = os.path.join(PROJECT_ROOT,"data","paddignton.jpeg")

def main():
    parser = argparse.ArgumentParser(description="Image description")
    parser.add_argument("--image", type=str, default=IMAGE_PATH, help="Path to Image processed")
    parser.add_argument("--query", type=str, required=True, help="A text query to rewrite based on the image")

    args = parser.parse_args()


    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img = f.read()

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    sys_prompt = """
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary"""
    contents = [
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt
        )
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
