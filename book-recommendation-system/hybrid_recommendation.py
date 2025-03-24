import pickle
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import cosine_similarity
from content_based_filtering import load_books_data, compute_similarity

# Load Data
books_df = load_books_data()
ratings_df = pd.read_csv("data/processed_ratings.csv")

# Load Collaborative Filtering Model (SVD)
def load_svd_model():
    with open("models/svd_model.pkl", "rb") as f:
        return pickle.load(f)

svd_model = load_svd_model()

# Load Content-Based Similarity Matrix
with open("models/book_similarity.pkl", "rb") as f:
    content_similarity = pickle.load(f)

def get_collaborative_recommendations(user_id, num_recs=10):
    """Generate book recommendations for a user using the collaborative filtering model."""
    book_ids = books_df["book_id"].unique()
    predictions = [(book_id, svd_model.predict(user_id, book_id).est) for book_id in book_ids]
    predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating
    return [book_id for book_id, _ in predictions[:num_recs]]

def get_content_recommendations(book_id, num_recs=5):
    """Get content-based recommendations based on similarity scores."""
    # Sort the similarity scores for the given book_id
    similar_books = sorted(enumerate(content_similarity[book_id]), key=lambda x: x[1], reverse=True)

    # Filter out the book_id itself from recommendations
    similar_books = [book for book in similar_books if book[0] != book_id]

    # Get the top N recommendations
    recommendations = similar_books[:num_recs]
    
    return recommendations

def hybrid_recommend(user_id, book_id, alpha=0.5, num_recs=5):
    # Get content-based recommendations
    content_recs = get_content_recommendations(book_id, num_recs) if book_id else []

    # Generate collaborative filtering recommendations using the SVD model
    collaborative_recs = []
    if book_id:
        # Generate collaborative recommendations by predicting ratings
        for _, row in books_df.iterrows():
            book_id_collab = row['book_id']
            predicted_rating = svd_model.predict(user_id, book_id_collab).est
            collaborative_recs.append((book_id_collab, predicted_rating))

    # Combine recommendations using the alpha weight
    combined_recs = []
    for content_book, content_sim in content_recs:
        for collab_book, collab_sim in collaborative_recs:
            if content_book == collab_book:
                combined_score = alpha * content_sim + (1 - alpha) * collab_sim
                combined_recs.append((content_book, combined_score))

    # Sort the combined recommendations by the score
    combined_recs = sorted(combined_recs, key=lambda x: x[1], reverse=True)

    # Return top N recommendations
    final_recs = combined_recs[:num_recs]

    # Fetch book information for the top recommendations
    recommended_books = []
    for book, _ in final_recs:
        book_info = books_df[books_df["book_id"] == book]  # Corrected here (no subscript needed)
        recommended_books.append(book_info)

    return recommended_books

# Example Usage
if __name__ == "__main__":
    user_id = 314  # Example user
    book_id = 1    # Example book
    recommendations = hybrid_recommend(user_id, book_id, alpha=0.5)
    print("\n📚 Hybrid Recommendations:")
    for i, book in enumerate(recommendations, 1):
        title = book["title"].values[0] if not book.empty else "Unknown Title"  # Corrected here
        print(f"{i}. {title}")