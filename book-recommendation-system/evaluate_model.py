import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

# load the trained model from the saved path
def load_model(file_path = "models/svd_model.pkl"):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    print("Model Loaded Successfully!")
    return model

# load processed ratings data
def load_data(file_path = "data/processed_ratings.csv"):
    print("--- Loading Ratings Data ---")
    ratings = pd.read_csv(file_path)
    print(f"Ratings Data Loaded! Shape: {ratings.shape}")
    return ratings

# prepare surprise dataset for evaluation
def prepare_suprise_data(ratings):
    print("--- Preparing Surprise Data ---")
    reader = Reader(rating_scale = (1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    return data

# evaluate model using RMSE and MAE
def evaluate_model(model, data):
    trainset, testset = train_test_split(data, test_size = 0.2)

    print("--- Making Predictions ---")
    preds = model.test(testset)

    rmse = accuracy.rmse(preds)
    mae = accuracy.mae(preds)

    print("Model Evaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    svd_model = load_model()
    ratings = load_data()
    data = prepare_suprise_data(ratings)

    evaluate_model(svd_model, data)