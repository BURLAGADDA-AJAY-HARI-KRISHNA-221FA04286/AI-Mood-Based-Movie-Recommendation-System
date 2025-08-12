import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Extract genres list (if needed later)
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g["name"] for g in genres]
    except:
        return []

movies["genres_list"] = movies["genres"].apply(extract_genres)
movies["overview"] = movies["overview"].fillna("")

# TF-IDF fit on all overviews
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["overview"])

# Recommend movies from text description
def recommend_by_feeling(user_text, top_n=10):
    user_vector = tfidf.transform([user_text])
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    return movies.iloc[top_indices][["title", "vote_average", "popularity"]]

# Streamlit UI
st.title("🎬 AI Mood-Based Movie Recommender")
st.write("💬 Describe how you feel or what you want to watch, and I'll recommend movies.")

user_feeling = st.text_area("Example: 'I feel adventurous and want an epic journey'")

if st.button("Recommend Movies"):
    recs = recommend_by_feeling(user_feeling)
    if not recs.empty:
        st.write("### Recommended Movies:")
        for _, row in recs.iterrows():
            st.write(f"🎬 **{row['title']}** — ⭐ {row['vote_average']}")
    else:
        st.write("No matching movies found.")
