from .search_utils import load_movies
import string

def tokenize(query:str):
    query = query.lower()
    removing_table = str.maketrans("","",string.punctuation)
    clean_query = query.translate(removing_table)
    tokens = clean_query.split()
    return tokens


def search_query(query:str,num_limit=5):
    movies = load_movies()
    result = []
    query_tokens = tokenize(query)
    for movie in movies:
        movie_title = movie["title"]
        tokenized_movie_title = tokenize(movie_title)
        for query_token in query_tokens:
            for movie_token in tokenized_movie_title:
                if movie_title not in result and query_token in movie_token:
                    result.append(movie_title)
    return result
