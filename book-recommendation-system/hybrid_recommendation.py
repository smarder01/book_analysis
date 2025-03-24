import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load preprocessed books data
def load_books_data():
    print("\n--- Loading Books Data ---")
    try:
        books = pd.read_csv("data/processed_books.csv")
        print(f"Books Data Loaded! Shape: {books.shape}")
        return books
    except FileNotFoundError:
        print("Error: Processed books file not found.")
        exit()

# Load collaborative filtering model
def load_cf_model():
    print("\n--- Loading Collaborative Filtering Model ---")
    try:
        with open("book-recommendation-system/models/svd_model.pkl", "rb") as f:
            cf_model = pickle.load(f)
        print("Collaborative Filtering Model Loaded!")
        return cf_model
    except FileNotFoundError:
        print("Error: Collaborative Filtering model file not found.")
        exit()

# Load content-based similarity matrix
def load_cb_similarity():
    print("\n--- Loading Content-Based Filtering Similarity Matrix ---")
    try:
        with open("book-recommendation-system/models/book_similarity.pkl", "rb") as f:
            cb_similarity = pickle.load(f)
        print("Content-Based Filtering Similarity Matrix Loaded!")
        return cb_similarity
    except FileNotFoundError:
        print("Error: Content-Based Filtering similarity matrix file not found.")
        exit()

# Get top-N similar books based on content-based filtering
def get_cb_similar_books(book_title, books, cb_similarity, top_n=10):
    print(f"\nFinding Books Similar to: {book_title} (Content-Based Filtering)")

    # Ensure book title exists
    if book_title not in books["title"].values:
        print(f"Error: Book '{book_title}' not found in dataset.")
        return []

    # Find the index of the given book
    book_idx = books.index[books["title"] == book_title].tolist()[0]

    # Get similarity scores for the book and sort them
    similarity_scores = list(enumerate(cb_similarity[book_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar books
    top_books = []
    for i, (index, score) in enumerate(similarity_scores[1:top_n + 1], 1):  # Skip self-match
        book_info = books.iloc[index]
        top_books.append((book_info["title"], book_info["authors"], round(score, 4)))

    return top_books

# Get top-N similar books based on collaborative filtering
def get_cf_similar_books(book_id, cf_model, top_n=10):
    print(f"\nFinding Books Similar to: {book_id} (Collaborative Filtering)")

    # Get the books that have been rated by users
    books_rated_by_users = cf_model.trainset.all_items()

    # Get predictions for the book from the CF model
    predictions = []
    for other_book_id in books_rated_by_users:
        prediction = cf_model.predict(book_id, other_book_id)
        predictions.append((other_book_id, prediction.est))

    # Sort predictions based on estimated ratings
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Get top N recommended books
    top_books = []
    for i, (book_id, rating) in enumerate(predictions[:top_n]):
        top_books.append((book_id, round(rating, 4)))

    return top_books

# Hybrid recommendation system (50% CF, 50% CBF)
def hybrid_recommendation(book_title, books, cb_similarity, cf_model, top_n=10):
    # Get book_id for the given book title
    book_id = books[books["title"] == book_title].index[0]

    # Get top-N similar books from content-based filtering
    cb_books = get_cb_similar_books(book_title, books, cb_similarity, top_n)

    # Get top-N similar books from collaborative filtering
    cf_books = get_cf_similar_books(book_id, cf_model, top_n)

    # Combine the results
    hybrid_books = {}
    
    for title, author, similarity in cb_books:
        hybrid_books[title] = {"author": author, "cb_similarity": similarity, "cf_similarity": 0}

    for book_id, cf_rating in cf_books:
        book_title = books.iloc[book_id]["title"]
        if book_title in hybrid_books:
            hybrid_books[book_title]["cf_similarity"] = cf_rating
        else:
            hybrid_books[book_title] = {"author": books.iloc[book_id]["authors"], "cb_similarity": 0, "cf_similarity": cf_rating}

    # Combine CF and CBF similarities
    hybrid_books_with_scores = []
    for title, info in hybrid_books.items():
        combined_score = 0.5 * info["cb_similarity"] + 0.5 * info["cf_similarity"]
        hybrid_books_with_scores.append(
            {"title": title, "author": info["author"], "score": combined_score}
        )

    # Sort books based on the combined score
    return sorted(hybrid_books_with_scores, key=lambda x: x["score"], reverse=True)[:top_n]

# Main script execution
if __name__ == "__main__":
    # Load necessary data
    books = load_books_data()
    cf_model = load_cf_model()
    cb_similarity = load_cb_similarity()

    # Example: Get hybrid recommendations for a given book title
    book_title = "Reckless"
    hybrid_books = hybrid_recommendation(book_title, books, cb_similarity, cf_model)

    # Display the results
    print("\n📚 Hybrid Book Recommendations:")
    for i, book in enumerate(hybrid_books, 1):
        title = book["title"]
        author = book["author"]
        combined_score = book["score"]
        print(f"{i}. {title} by {author} (Combined Score: {combined_score:.4f})")