import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    """Load book dataset with error handling."""
    logging.info("--- Loading Datasets ---")
    file_path = "data/goodreads_top100_from1980to2023_final.csv"

    if os.path.exists(file_path):
        books = pd.read_csv(file_path)
    else:
        logging.error(f"File {file_path} not found. Please check the file path.")
        return None

    return books

def clean_books(books):
    """Clean and process book metadata."""
    books = books[[
        "isbn", "title", "series_title", "series_release_number", "authors", "publisher", "language", 
        "description", "num_pages", "format", "genres", "publication_date", "rating_score", 
        "num_ratings", "num_reviews", "current_readers", "want_to_read", "price", "url"
    ]]

    # Rename columns for consistency
    books.rename(columns={"isbn": "book_id", "rating_score": "average_rating", "num_ratings": "ratings_count"}, inplace=True)

    # Convert to numeric & handle missing values
    books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce")
    books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce").fillna(0).astype(int)
    books["num_reviews"] = pd.to_numeric(books["num_reviews"], errors="coerce").fillna(0).astype(int)
    books["current_readers"] = pd.to_numeric(books["current_readers"], errors="coerce").fillna(0).astype(int)
    books["want_to_read"] = pd.to_numeric(books["want_to_read"], errors="coerce").fillna(0).astype(int)
    books["price"] = pd.to_numeric(books["price"], errors="coerce").fillna(0).astype(float)

    # Handle missing authors (if any)
    books["authors"].fillna("Unknown", inplace=True)

    # Convert genres to category for memory optimization
    books["genres"] = books["genres"].astype("category")

    logging.info("Clean books complete!")
    return books

def save_processed_data(books):
    """Save preprocessed data to CSV files."""
    books.to_csv("data/processed_books.csv", index=False)
    logging.info("Saving preprocessed data complete!")

if __name__ == "__main__":
    books = load_data()
    if books is not None:
        books = clean_books(books)
        save_processed_data(books)
        logging.info("Data preprocessing complete!")