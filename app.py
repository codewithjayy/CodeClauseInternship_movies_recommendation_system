# app.py - Flask application for movie recommendation system

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)

# Load datasets (or create sample data if running for the first time)
def load_data():
    # Check if we have pre-processed data
    if os.path.exists('models/ratings_matrix.pkl') and os.path.exists('models/svd_model.pkl'):
        ratings_matrix = pickle.load(open('models/ratings_matrix.pkl', 'rb'))
        svd_model = pickle.load(open('models/svd_model.pkl', 'rb'))
        movies_df = pd.read_csv('data/movies.csv')
        return ratings_matrix, svd_model, movies_df
    
    # If first run, create sample data or load from MovieLens dataset
    try:
        # Try to load MovieLens data if available
        movies_df = pd.read_csv('data/movies.csv')
        ratings_df = pd.read_csv('data/ratings.csv')
    except:
        # Create sample data
        print("Creating sample movie and ratings data...")
        # Sample movies
        movies_data = {
            'movieId': list(range(1, 21)),
            'title': [
                'The Shawshank Redemption (1994)', 'The Godfather (1972)', 
                'Pulp Fiction (1994)', 'The Dark Knight (2008)',
                'Forrest Gump (1994)', 'The Matrix (1999)',
                'Star Wars: Episode IV (1977)', 'Inception (2010)',
                'Titanic (1997)', 'Jurassic Park (1993)',
                'The Lion King (1994)', 'Interstellar (2014)',
                'Avatar (2009)', 'The Avengers (2012)',
                'Toy Story (1995)', 'The Silence of the Lambs (1991)',
                'Fight Club (1999)', 'The Lord of the Rings (2001)',
                'Gladiator (2000)', 'The Departed (2006)'
            ],
            'genres': [
                'Drama', 'Crime|Drama', 
                'Crime|Drama', 'Action|Crime|Drama',
                'Drama|Romance', 'Action|Sci-Fi',
                'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi',
                'Drama|Romance', 'Adventure|Sci-Fi',
                'Animation|Adventure|Drama', 'Adventure|Drama|Sci-Fi',
                'Action|Adventure|Fantasy|Sci-Fi', 'Action|Adventure|Sci-Fi',
                'Animation|Adventure|Comedy', 'Crime|Drama|Thriller',
                'Drama', 'Adventure|Fantasy',
                'Action|Adventure|Drama', 'Crime|Drama|Thriller'
            ]
        }
        movies_df = pd.DataFrame(movies_data)
        
        # Sample ratings (simulate 50 users rating these movies)
        user_ids = list(range(1, 51))
        ratings_data = []
        for user_id in user_ids:
            # Each user rates a random number of movies
            num_ratings = np.random.randint(5, 15)
            movie_ids = np.random.choice(movies_df['movieId'].values, size=num_ratings, replace=False)
            for movie_id in movie_ids:
                # Ratings between 1 and 5
                rating = np.random.randint(1, 6)
                ratings_data.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating,
                    'timestamp': 1619784000 # Dummy timestamp
                })
        ratings_df = pd.DataFrame(ratings_data)
        
        # Save the created data
        os.makedirs('data', exist_ok=True)
        movies_df.to_csv('data/movies.csv', index=False)
        ratings_df.to_csv('data/ratings.csv', index=False)
    
    # Create user-item ratings matrix
    ratings_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
    
    # Train SVD model
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd_model = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd_model.fit(trainset)
    
    # Save the preprocessed data and model
    os.makedirs('models', exist_ok=True)
    pickle.dump(ratings_matrix, open('models/ratings_matrix.pkl', 'wb'))
    pickle.dump(svd_model, open('models/svd_model.pkl', 'wb'))
    
    return ratings_matrix, svd_model, movies_df

# Load data at startup
ratings_matrix, svd_model, movies_df = load_data()

# Utility functions for recommendations
def get_similar_users(user_ratings, ratings_matrix, n=5):
    """Find similar users based on rating patterns"""
    # Fill in the user's ratings in a user vector
    user_vector = pd.Series(index=ratings_matrix.columns, dtype=float)
    for movie_id, rating in user_ratings.items():
        if int(movie_id) in user_vector.index:
            user_vector[int(movie_id)] = rating
    
    # Calculate similarity between this user and all others
    user_similarity = {}
    for user_id in ratings_matrix.index:
        # Get ratings for this user
        other_user = ratings_matrix.loc[user_id].dropna()
        # Find movies rated by both
        common_movies = list(set(user_vector.dropna().index) & set(other_user.index))
        
        if len(common_movies) > 0:
            # Calculate similarity if there are common movies
            user_ratings_common = user_vector[common_movies].values
            other_ratings_common = other_user[common_movies].values
            
            # Simple correlation
            similarity = np.corrcoef(user_ratings_common, other_ratings_common)[0, 1]
            if not np.isnan(similarity):
                user_similarity[user_id] = similarity
    
    # Return the most similar users
    similar_users = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[:n]
    return similar_users

def collaborative_filtering_recommendations(user_ratings, ratings_matrix, movies_df, n=10):
    """Get recommendations using user-based collaborative filtering"""
    # Find similar users
    similar_users = get_similar_users(user_ratings, ratings_matrix)
    
    # Get ratings from similar users for movies the current user hasn't rated
    movie_scores = {}
    movie_count = {}
    
    # Movies already rated by the user
    rated_movies = set(int(movie_id) for movie_id in user_ratings.keys())
    
    for user_id, similarity in similar_users:
        user_ratings_df = ratings_matrix.loc[user_id].dropna()
        
        for movie_id, rating in user_ratings_df.items():
            if movie_id not in rated_movies:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = 0
                    movie_count[movie_id] = 0
                
                # Weight rating by similarity
                movie_scores[movie_id] += rating * similarity
                movie_count[movie_id] += abs(similarity)
    
    # Calculate weighted average scores
    weighted_scores = {movie_id: score/movie_count[movie_id] 
                       for movie_id, score in movie_scores.items() 
                       if movie_count[movie_id] > 0}
    
    # Sort by score and get top n
    recommended_movie_ids = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Get movie details
    recommendations = []
    for movie_id, score in recommended_movie_ids:
        movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        recommendations.append({
            'movieId': int(movie_id),
            'title': movie['title'],
            'genres': movie['genres'],
            'score': round(float(score), 2)
        })
    
    return recommendations

def model_based_recommendations(user_id, user_ratings, svd_model, movies_df, n=10):
    """Get recommendations using SVD matrix factorization"""
    # Get movies not rated by the user
    rated_movies = set(int(movie_id) for movie_id in user_ratings.keys())
    unrated_movies = set(movies_df['movieId']) - rated_movies
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        predicted_rating = svd_model.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    
    # Sort by predicted rating and get top n
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    # Get movie details
    recommendations = []
    for movie_id, score in top_predictions:
        movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        recommendations.append({
            'movieId': int(movie_id),
            'title': movie['title'],
            'genres': movie['genres'],
            'score': round(float(score), 2)
        })
    
    return recommendations

# Route for home page
@app.route('/')
def index():
    # Get list of all movies for the preference selection
    movies_list = movies_df[['movieId', 'title', 'genres']].to_dict('records')
    return render_template('index.html', movies=movies_list)

# Route for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user ratings from form
    user_ratings = request.json.get('ratings', {})
    
    # Check if we have enough ratings
    if len(user_ratings) < 3:
        return jsonify({'error': 'Please rate at least 3 movies for better recommendations'})
    
    # User ID for model-based recommendations (use a fixed ID for demo)
    user_id = 9999
    
    # Get recommendations using collaborative filtering
    cf_recommendations = collaborative_filtering_recommendations(user_ratings, ratings_matrix, movies_df)
    
    # Get recommendations using matrix factorization
    model_recommendations = model_based_recommendations(user_id, user_ratings, svd_model, movies_df)
    
    return jsonify({
        'collaborative_filtering': cf_recommendations,
        'model_based': model_recommendations
    })

# Run the application
if __name__ == '__main__':
    app.run(debug=True)