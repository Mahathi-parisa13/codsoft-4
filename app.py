import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite movie")

# ---------------------------
# Sample Movie Dataset
# ---------------------------
movies_dict = {
    "title": [
        "Avatar",
        "Titanic",
        "The Avengers",
        "Iron Man",
        "Batman Begins",
        "The Dark Knight",
        "Interstellar",
        "Inception",
        "Doctor Strange",
        "Spider-Man"
    ],
    
    "genre": [
        "Action Adventure Sci-Fi",
        "Romance Drama",
        "Action Superhero",
        "Action Sci-Fi",
        "Action Crime",
        "Action Crime Drama",
        "Sci-Fi Space Drama",
        "Sci-Fi Thriller",
        "Fantasy Action",
        "Action Superhero"
    ]
}

movies = pd.DataFrame(movies_dict)

# ---------------------------
# Convert Text to Features
# ---------------------------
cv = CountVectorizer()

vectors = cv.fit_transform(movies["genre"]).toarray()

# ---------------------------
# Similarity Matrix
# ---------------------------
similarity = cosine_similarity(vectors)

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# ---------------------------
# User Input
# ---------------------------
selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

# ---------------------------
# Show Recommendations
# ---------------------------
if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("✅", movie)
