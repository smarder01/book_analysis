import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    """Load book datasets with error handling."""
    logging.info("--- Loading Datasets ---")
    file_paths = {
        "ratings": "data/ratings.csv",
        "books": "data/books.csv",
        "book_tags": "data/book_tags.csv",
        "tags": "data/tags.csv"
    }

    data = {}
    for name, path in file_paths.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            logging.error(f"File {path} not found. Please check the file path.")
            return None

    return data["ratings"], data["books"], data["book_tags"], data["tags"]

def merge_tags(book_tags, tags):
    """Merge book_tags with their tag names and combine them per book."""
    book_tags = book_tags.merge(tags, on="tag_id", how="left")
    book_tags = book_tags.groupby("goodreads_book_id")["tag_name"].apply(lambda x: " | ".join(x)).reset_index()
    logging.info("Merge tags complete!")
    return book_tags

def clean_books(books):
    """Clean and process book metadata."""
    books = books[["id", "title", "authors", "original_publication_year", "average_rating", "ratings_count", "image_url"]]
    books.rename(columns={"id": "book_id", "original_publication_year": "year"}, inplace=True)

    # Convert to numeric & handle missing values
    books["year"] = pd.to_numeric(books["year"], errors="coerce").fillna(0).astype(int)
    books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce")
    books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce").fillna(0).astype(int)

    # Optimize memory usage
    books["authors"] = books["authors"].astype("category")

    logging.info("Clean books complete!")
    return books

def merge_books_with_tags(books, book_tags):
    """Merge books with tags for content-based recommendations."""
    books = books.merge(book_tags, left_on="book_id", right_on="goodreads_book_id", how="left")
    books["tag_name"] = books["tag_name"].fillna("")
    books.drop(columns=["goodreads_book_id"], inplace=True)  # Drop unnecessary column
    logging.info("Merge books with tags complete!")
    return books

def clean_ratings(ratings):
    """Clean and filter ratings data to remove noise."""
    ratings.drop_duplicates(subset=["user_id", "book_id"], inplace=True)

    # Remove users with <5 ratings
    user_counts = ratings["user_id"].value_counts()
    ratings = ratings[ratings["user_id"].isin(user_counts[user_counts >= 5].index)]

    # Remove books with <50 ratings
    book_counts = ratings["book_id"].value_counts()
    ratings = ratings[ratings["book_id"].isin(book_counts[book_counts >= 50].index)]

    # Optimize memory
    ratings["user_id"] = ratings["user_id"].astype("int32")
    ratings["book_id"] = ratings["book_id"].astype("int32")
    ratings["rating"] = ratings["rating"].astype("float32")

    logging.info("Clean ratings complete!")
    return ratings

def save_processed_data(books, ratings):
    """Save preprocessed data to CSV files."""
    books.to_csv("data/processed_books.csv", index=False)
    ratings.to_csv("data/processed_ratings.csv", index=False)
    logging.info("Saving preprocessed data complete!")

if __name__ == "__main__":
    data = load_data()
    if data:
        ratings, books, book_tags, tags = data
        book_tags = merge_tags(book_tags, tags)
        books = clean_books(books)
        books = merge_books_with_tags(books, book_tags)
        ratings = clean_ratings(ratings)
        save_processed_data(books, ratings)
        logging.info("Data preprocessing complete!")