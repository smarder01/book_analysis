import streamlit as st
import pandas as pd
import pickle
from content_based_filtering import load_books_data
from hybrid_recommendation import hybrid_recommend

# load books data
books_df = load_books_data()

# streamlit app title
st.title("Book Reccomendation System")

# user input
# select a book
st.subheader("Select a Book for Recommendations:")
book_options = books_df["title"].tolist()
book_id = st.selectbox("Choose a Book:", book_options)

# alpha slider: adjust the influence of collaborative vs content-based filtering
st.subheader("Adjust the Weight Between Recommendation Methods")
st.write("""
Content-based filtering recommends books similar to the one you've selected, based on its features like genre, author, or keywords.
Collaborative filtering recommends books based on the preferences of other users who have similar tastes to yours.

Use the slider to adjust how much you want the recommendations to rely on each method:
- A higher weight for content-based filtering will prioritize books with similar themes or genres.
- A higher weight for collaborative filtering will suggest books liked by users who share your preferences.
""")
alpha = st.slider("Adjust the weight between content-based and collaborative recommendations:", 0.0, 1.0, 0.5)

# button to get recs
if st.button("Get Recommendations"):
    # get book id for the selected book
    book_id_selected = books_df[books_df["title"] == book_id].iloc[0]["book_id"]

    # get hybrid recs
    recommendations = hybrid_recommend(user_id = None, book_id = book_id_selected, alpha = alpha)

    # display recs
    st.subheader("Recommended Books:")
    for i, book_info in enumerate(recommendations, 1):
        book_title = book_info["title"].values[0] if not book_info.empty else "Unknown Title"
        st.write(f"{i}. {book_title}")