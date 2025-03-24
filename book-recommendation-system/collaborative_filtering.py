import pandas as pd 
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pickle

def load_ratings(file_path = "data/processed_ratings.csv"):
    # load the processed ratings dataset
    print("--- Loading Ratings Data ---")
    ratings = pd.read_csv(file_path)
    return ratings

def prepare_surprise_data(ratings):
    """
    Convert the ratings DataFrame into a Surprise Dataset.
    """
    print("🔍 Checking Ratings Data Before Conversion...")
    print(f"Type of 'ratings': {type(ratings)}")  # Should be <class 'pandas.DataFrame'>
    print(f"Ratings Data Shape: {ratings.shape}")  # Should not be (0, 0)
    print("Columns in Ratings:", ratings.columns.tolist())  # Check column names
    
    # Ensure it's a DataFrame
    if not isinstance(ratings, pd.DataFrame):
        raise TypeError("❌ Expected ratings to be a DataFrame, got something else.")

    # Ensure required columns exist
    required_columns = ["user_id", "book_id", "rating"]
    missing_columns = [col for col in required_columns if col not in ratings.columns]
    if missing_columns:
        raise KeyError(f"❌ Missing required columns: {missing_columns}")

    reader = Reader(rating_scale=(1, 5))  # Assuming ratings are between 1-5

    print("✅ Preparing Surprise Dataset...")
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    print("✅ Surprise Dataset Ready!")
    
    return data

def train_svd_model(data):
    # Train an SVD model using surprise
    print("--- Training SVD Model ---")
    svd = SVD()
    cross_validate(svd, data, cv = 5, verbose = True)
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    print("SVD Model Training Complete!")
    return svd

import os
import pickle

def save_model(model, file_path):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    
    # Save the model
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved successfully at {file_path}")

if __name__ == "__main__":
    ratings = load_ratings()
    data = prepare_surprise_data(ratings)
    svd_model = train_svd_model(data)
    save_model(svd_model, 'models/svd_model.pkl')
    print("Collaborative Filtering Model Ready!")