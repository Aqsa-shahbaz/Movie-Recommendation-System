from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import numpy as np
from recommender import load_and_preprocess_data, build_model, recommend_movies

app = Flask(__name__)

# Load data and model once when server starts
movies, ratings, _, _, _, _, userencoded, user_rev, moviecoded, movie_rev, genres = load_and_preprocess_data()

# Load or build the model
model = build_model(len(userencoded), len(moviecoded))
model.load_weights("movie_recommendation_model.h5")

@app.route("/")
def index():
    # Show genre selection form
    genre_list = [genre for genre in genres]
    return render_template("index.html", genres=genre_list)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = int(request.form["user_id"])
        selected_genres = request.form.getlist("genres") 
        top_n = int(request.form["top_n"])

        recommendations = recommend_movies(
            model, movies, ratings, userencoded, moviecoded,
            user_id, selected_genres, top_n
        )

        if "error" in recommendations:
            return jsonify({"error": recommendations["error"]})

        return render_template("index.html", genres=genres, recommendations=recommendations)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
