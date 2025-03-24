# Bookish Bytes: Recommendation & Rating Prediction Magic 📖✨🔮
Book Recommendation and Rating Prediction Projects

Welcome to the **Book Recommendation and Rating Prediction** repository! This collection of projects focuses on machine learning techniques to recommend books and predict book ratings. The repository contains two primary projects:

1. **Personalized Book Recommendation System**: Recommend books based on a user's past reading history.
2. **Predict a Book's Rating Before It Gets Reviews**: Predict a book's rating based on features like title, author, genre, and description.

## Project Goals

- **Personalized Book Recommendation System**: 
  - Recommend books to users based on their past reading history using collaborative and content-based filtering methods.
  - Build a simple recommendation engine and extend it with deep learning approaches.

- **Predict a Book's Rating**: 
  - Predict a book’s rating by using features such as the title, author, genre, and description.
  - Implement regression models and explore the use of LSTM models for sequential text data.

## Dataset

The datasets used in these projects are from **Kaggle**:

- **Ultimate Book Collection: Top 100 Books up to 2023**: For both the recommendation system and rating prediction.

### Basic Information About the Data

- **isbn**: ISBN codes of the book.
- **title**: Titles of the book.
- **series_title**: Titles of the series to which some books belong.
- **series_release_number**: Release numbers of the series for some books.
- **authors**: Authors of the book.
- **publisher**: Publishers of the book.
- **language**: Language in which the book is written.
- **description**: Descriptions of the book.
- **num_pages**: Number of pages of the book.
- **format**: Formats of the book (paperback, e-book, etc.).
- **genres**: Genres to which the book belongs.
- **publication_date**: Publication dates of the book.
- **rating_score**: Rating score of the book.
- **num_ratings**: Number of ratings received by the book.
- **num_reviews**: Number of reviews of the book.
- **current_readers**: Current number of readers of the book.
- **want_to_read**: Number of people interested in reading the book.
- **price**: Prices of the book.
- **url**: URLs of the book.

## Tech Stack

- **Recommendation System**:
  - Pandas for data manipulation
  - Scikit-learn for traditional models
  - Surprise library for collaborative filtering
  - TensorFlow for deep learning models

- **Rating Prediction**:
  - Pandas for data manipulation
  - Scikit-learn for regression models
  - LSTM (TensorFlow/Keras) for sequence-based prediction models

## Project Structure
```bash
book-recommendation-rating-prediction/
│
├── data/
│   ├── goodreads_top100_from1980to2023_final.csv
│   ├── processed_books.csv
│
├── book-recommendation-system/
│   ├── preprocess.py
│   ├── collaborative_filtering.py
│   ├── content_based_filtering.py
│   ├── hybrid_recommendation.py
│   ├── evaluate_model.py
│   ├── streamlit_app.py
│   └── README.md
│
├── rating-prediction/
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── README.md
│
├── requirements.txt
└── README.md
