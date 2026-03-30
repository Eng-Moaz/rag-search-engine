from .search_utils import load_movies, load_stopwords
import string

def clean(query:str):
    query = query.lower()
    removing_table = str.maketrans("","",string.punctuation)
    clean_query = query.translate(removing_table)
    return clean_query

def tokenize(query:str):
    clean_query = clean(query)
    tokens = clean_query.split()
    return tokens

def remove_stopwords(tokenized_text):
    stopwords = load_stopwords()
    for stopword in stopwords:
        if stopword in tokenized_text:
            tokenized_text.remove(stopword)
    return tokenized_text

def matching_movie(tokenized_query,tokenized_movie_title)-> bool:
    for query_token in tokenized_query:
        for movie_title_token in tokenized_movie_title:
            if query_token in movie_title_token:
                return True
    return False

def search_query(query:str,num_limit=5):
    movies = load_movies()
    result = []
    tokenized_query = remove_stopwords(tokenize(query))
    for movie in movies:
        tokenized_movie_title = remove_stopwords(tokenize(movie["title"]))
        if matching_movie(tokenized_query,tokenized_movie_title):
            result.append(movie)
        if len(result) == num_limit:
            break
    return result
