import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pickle
import os

# Load book data
books = pd.read_csv("data/goodreads_top100_from1980to2023_final.csv")

# Generate synthetic user ratings
num_users = 10000  # Adjust as needed
num_samples = 50000  # Total number of ratings to generate

# Ensure ISBNs are used as book IDs
book_ids = books["isbn"].dropna().unique()
user_ids = np.random.randint(1, num_users + 1, size=num_samples)
book_sample = np.random.choice(book_ids, size=num_samples)
ratings = np.clip(np.random.normal(books["rating_score"].mean(), 1, size=num_samples), 1, 5)  # Random ratings around avg score

# Create synthetic ratings DataFrame
ratings_df = pd.DataFrame({"user_id": user_ids, "book_id": book_sample, "rating": ratings.round(1)})

# Save the generated ratings dataset
os.makedirs("data", exist_ok=True)
ratings_df.to_csv("data/synthetic_ratings.csv", index=False)
print(f"Synthetic ratings dataset created with shape: {ratings_df.shape}")

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["user_id", "book_id", "rating"]], reader)

# Train the SVD model
svd = SVD()
cross_validate(svd, data, cv=5, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)

# Save the trained model
os.makedirs("models", exist_ok=True)
with open("book-recommendation-system/models/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)
print("Collaborative Filtering Model Saved!")