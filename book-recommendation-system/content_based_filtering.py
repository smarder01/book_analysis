import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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

# Preprocess genres column
def preprocess_genres(books):
    if "genres" not in books.columns:
        print("Error: 'genres' column not found in dataset.")
        return books

    # Fill missing genres and replace commas with spaces
    books["genres"] = books["genres"].fillna("").str.replace(",", " ")

    print("\nSample Book Genres (After Processing):")
    print(books["genres"].head(10))  # Show first 10 processed genres

    return books

# Compute cosine similarity between books
def compute_similarity(books):
    print("\n--- Computing Book Similarity ---")

    # Preprocess genres
    books = preprocess_genres(books)

    # Combine multiple features into one "metadata" column
    books["metadata"] = (
        books["title"].fillna("") + " " +
        books["authors"].fillna("") + " " +
        books["genres"].fillna("") + " " +
        books["description"].fillna("")  # Using description for more detail
    )

    # Convert metadata into TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(books["metadata"])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print("--- Similarity Computation Complete ---")
    return books, cosine_sim

# Get top similar books for a given title
def get_similar_books(book_title, books, cosine_sim, top_n=10):
    print(f"\nFinding Books Similar to: {book_title}")

    # Ensure book title exists
    if book_title not in books["title"].values:
        print(f"Error: Book '{book_title}' not found in dataset.")
        return []

    # Find the index of the given book
    book_idx = books.index[books["title"] == book_title].tolist()[0]

    # Get similarity scores for the book and sort them
    similarity_scores = list(enumerate(cosine_sim[book_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar books
    top_books = []
    for i, (index, score) in enumerate(similarity_scores[1:top_n + 1], 1):  # Skip self-match
        book_info = books.iloc[index]
        top_books.append((book_info["title"], book_info["authors"], round(score, 4)))

    return top_books

# Main script execution
if __name__ == "__main__":
    books = load_books_data()
    books, cosine_sim = compute_similarity(books)

    # Example: Find books similar to "The Hunger Games"
    book_title = "Reckless"
    similar_books = get_similar_books(book_title, books, cosine_sim)
    
    # Save similarity matrix
    with open("book-recommendation-system/models/book_similarity.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)

    # Print results
    if similar_books:
        print("\n📚 Similar Books:")
        for i, (title, author, similarity) in enumerate(similar_books, 1):
            print(f"{i}. {title} by {author} (Similarity: {similarity})")