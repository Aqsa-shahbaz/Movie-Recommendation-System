# recommender.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers
from sklearn.model_selection import train_test_split

# Load and preprocess data
def load_and_preprocess_data():
    movies = pd.read_csv('dataset/movies.csv')
    ratings = pd.read_csv('dataset/ratings.csv')

    user_ids = ratings["userId"].unique().tolist()
    userencoded = {x: i for i, x in enumerate(user_ids)}
    user_rev = {i: x for i, x in enumerate(user_ids)}

    movie_ids = ratings['movieId'].unique().tolist()
    moviecoded = {x: i for i, x in enumerate(movie_ids)}
    movie_rev = {i: x for i, x in enumerate(movie_ids)}

    ratings['user'] = ratings['userId'].map(userencoded)
    ratings['movie'] = ratings['movieId'].map(moviecoded)

    # Normalize ratings between 0 and 1
    ratings['rating'] = (ratings['rating'] - ratings['rating'].mean()) / ratings['rating'].std()
    max_rating = max(ratings['rating'])
    min_rating = min(ratings['rating'])
    ratings['rating'] = ratings['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))

    # One-hot encode genres
    genres = set()
    for genre_list in movies['genres']:
        genres.update(genre_list.split('|'))
    genres = list(genres)

    for genre in genres:
        movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

    x = ratings[['user', 'movie']].values
    y = ratings['rating'].values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

    return movies, ratings, x_train, x_val, y_train, y_val, userencoded, user_rev, moviecoded, movie_rev, genres

def build_model(num_users, num_movies, embedding_size=100):
    user_layer = layers.Input(shape=[1])
    user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer="he_normal",
                                     embeddings_regularizer=keras.regularizers.l2(1e-6))(user_layer)
    user_vector = layers.Flatten()(user_embedding)

    movie_layer = layers.Input(shape=[1])
    movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer="he_normal",
                                      embeddings_regularizer=keras.regularizers.l2(1e-6))(movie_layer)
    movie_vector = layers.Flatten()(movie_embedding)

    prod = layers.dot(inputs=[user_vector, movie_vector], axes=1)
    dense1 = layers.Dense(200, activation='relu')(prod)
    dense2 = layers.Dense(100, activation='relu')(dense1)
    dropout = layers.Dropout(0.5)(dense2)
    dense3 = layers.Dense(1)(dropout)

    model = Model([user_layer, movie_layer], dense3)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model

def recommend_movies(model, movies, ratings, userencoded, moviecoded, user_id, selected_genres, top_n=10):
    if user_id not in userencoded:
        return {"error": f"User ID {user_id} not found."}

    user_encoder = userencoded[user_id]
    
    movies_watched = ratings[ratings['user'] == user_encoder][['movieId', 'rating']]
    movies_not_watched = movies[~movies["movieId"].isin(movies_watched['movieId'])]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(moviecoded.keys())))
    
    user_movie_array = np.hstack((
        np.array([[user_encoder]] * len(movies_not_watched)), 
        np.array([[moviecoded[x]] for x in movies_not_watched])
    ))

    predicted_ratings = model.predict([user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()

    predicted_ratings_dict = {movie_id: rating for movie_id, rating in zip(movies_not_watched, predicted_ratings)}

    # Filter movies by genres
    filtered_movies = movies[movies[selected_genres].sum(axis=1) > 0]
    filtered_movie_ids = filtered_movies['movieId'].values
    filtered_predicted_ratings = {movie_id: predicted_ratings_dict.get(movie_id, 0) for movie_id in filtered_movie_ids}

    top_movies = sorted(filtered_predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]

    recommended = [{"title": movies[movies["movieId"] == movie_id]["title"].values[0],
                    "genres": movies[movies["movieId"] == movie_id]["genres"].values[0]}
                   for movie_id, _ in top_movies]

    return recommended
