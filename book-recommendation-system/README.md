# BookBuddy: Your AI Reading Companion

## Objective
The goal of this project is to build a personalized book recommendation system that suggests books based on a user's past reading history. By analyzing user preferences and applying machine learning techniques, this system aims to recommend books tailored to each individual's unique reading tastes.

## Project Overview
This project uses a combination of content-based filtering, collaborative filtering, and deep learning techniques to recommend books. By leveraging user data from Goodreads, the system is designed to offer personalized book suggestions based on what users have liked in the past. Additionally, a Streamlit web app will be built to allow users to input books they enjoy and receive recommendations in real-time.

Key Features:
- **Content-based filtering**: Recommends books based on similarity to books the user has already liked.
- **Collaborative filtering**: Uses data from other users with similar reading habits to provide suggestions.
- **Matrix factorization**: Identifies latent factors in user and book interactions for improved recommendations.
- **Web app**: A simple and interactive Streamlit web app to get book recommendations instantly.

## Dataset
The dataset used for this project is from the **Goodreads Books** dataset available on Kaggle. This dataset contains information about books, including their titles, authors, genres, and user ratings. The dataset is rich with features that enable us to implement various recommendation techniques.

### Dataset includes:
- Book information (title, author, genres)
- User ratings and reviews
- User activity data (books read, ratings given)

## Tech Stack
This project uses a combination of the following technologies:
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For machine learning algorithms and data preprocessing.
- **Surprise**: A Python library for building collaborative filtering models (e.g., matrix factorization).
- **TensorFlow**: For deep learning techniques to enhance the recommendation system.
- **Streamlit**: For creating an interactive web app where users can input their book preferences and receive real-time recommendations.

## How to Run
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/book-recommendation-system.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Future Enhancements
- Improve recommendation accuracy by fine-tuning models.
- Add more advanced features such as user profiles and book summaries.
- Explore other recommendation algorithms and hybrid methods.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.