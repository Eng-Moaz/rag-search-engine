import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT,"data","movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT,"data","stopwords.txt")
GOLDEN_DATA_PATH = os.path.join(PROJECT_ROOT,"data","golden_dataset.json")

def load_movies() -> list[dict]:
    with open(DATA_PATH,"r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH,"r") as f:
        stopwords = f.read().splitlines()
    return stopwords

def load_golden() -> list[dict]:
    with open(GOLDEN_DATA_PATH,"r") as f:
        data = json.load(f)
    return data["test_cases"]