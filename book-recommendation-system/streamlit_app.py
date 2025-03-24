import streamlit as st
import pandas as pd
import pickle
from hybrid_recommendation import hybrid_recommendation  # Assuming you saved this function in a separate file

# Load data and models
@st.cache(allow_output_mutation=True)
def load_data():
    # Load the necessary data
    books = pd.read_csv('data/processed_books.csv')  # Adjust path as necessary
    return books

@st.cache(allow_output_mutation=True)
def load_models():
    # Load content-based filtering similarity matrix
    with open("book-recommendation-system/models/book_similarity.pkl", "rb") as f:
        cb_similarity = pickle.load(f)
    
    # Load collaborative filtering model
    with open("book-recommendation-system/models/svd_model.pkl", "rb") as f:
        cf_model = pickle.load(f)

    return cb_similarity, cf_model

# Streamlit UI
st.title("Book Recommendation System")

# Instructions
st.write("""
    This app recommends books based on a hybrid recommendation system, combining content-based filtering 
    and collaborative filtering. Enter a book title, and we'll show you similar books!
""")

# Load data and models
books = load_data()
cb_similarity, cf_model = load_models()

# User input for book title
book_title = st.text_input("Enter a Book Title", "")

if book_title:
    # Get recommendations
    st.write(f"Recommendations for: **{book_title}**")
    hybrid_books = hybrid_recommendation(book_title, books, cb_similarity, cf_model)
    
    # Display recommendations
    for i, book in enumerate(hybrid_books):
        st.write(f"{i+1}. {book['title']} by {book['authors']} (Rating: {book['rating_score']})")