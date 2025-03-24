import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import numpy as np

# Load books data
def load_books(file_path = "data/processed_books.csv"):
    print("--- Loading Books Data ---")
    books = pd.read_csv(file_path)
    print(f"Books Data Loaded! Shape: {books.shape}")
    return books

# Load ratings data
def load_ratings(file_path = "data/synthetic_ratings.csv"):
    print("--- Loading Ratings Data ---")
    ratings = pd.read_csv(file_path)
    print(f"Ratings Data Loaded! Shape: {ratings.shape}")
    return ratings

# Load pre-trained models
def load_models():
    print("--- Loading Models ---")
    
    # Load content-based filtering similarity matrix
    with open("book-recommendation-system/models/book_similarity.pkl", "rb") as f:
        cb_similarity = pickle.load(f)
    print("Content-Based Filtering Model Loaded!")

    # Load collaborative filtering model
    with open("book-recommendation-system/models/svd_model.pkl", "rb") as f:
        cf_model = pickle.load(f)
    print("Collaborative Filtering Model Loaded!")
    
    return cb_similarity, cf_model

# Evaluate models using synthetic ratings
def evaluate_models(ratings, books, cb_similarity, cf_model):
    # Prepare data for collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Evaluate collaborative filtering (SVD model)
    print("--- Evaluating Collaborative Filtering Model ---")
    cf_model_predictions = cf_model.test(testset)
    cf_rmse = accuracy.rmse(cf_model_predictions)
    print(f"Collaborative Filtering RMSE: {cf_rmse}")
    
    # Evaluate content-based filtering using the similarity matrix
    print("--- Evaluating Content-Based Filtering Model ---")
    
    def get_cb_recommendations(book_title, cb_similarity, books, top_n=10):
        book_idx = books[books['title'] == book_title].index[0]
        similarity_scores = list(enumerate(cb_similarity[book_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_books_idx = [i[0] for i in similarity_scores[1:top_n+1]]
        recommended_books = books.iloc[recommended_books_idx]
        return recommended_books
    
    # Test content-based filtering by recommending books for a sample
    test_book_title = "Reckless"  # Replace with an actual book title from your data
    recommended_books_cb = get_cb_recommendations(test_book_title, cb_similarity, books)
    print(f"Top 10 Content-Based Recommendations for '{test_book_title}':")
    print(recommended_books_cb[['title', 'authors']])

if __name__ == "__main__":
    # Load the data and models
    books = load_books()
    ratings = load_ratings()
    cb_similarity, cf_model = load_models()

    # Evaluate models
    evaluate_models(ratings, books, cb_similarity, cf_model)