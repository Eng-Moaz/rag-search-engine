from .search_utils import load_movies
import string

def preprocess(query:str):
    query = query.lower()
    removing_table = str.maketrans("","",string.punctuation)
    return query.translate(removing_table)


def search_query(query:str,num_limit=5):
    movies = load_movies()
    result = []
    for movie in movies:
        if preprocess(query) in preprocess(movie["title"]):
            result.append(movie["title"])
        if len(result) == num_limit:
            break
    return result
