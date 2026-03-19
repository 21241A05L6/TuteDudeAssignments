import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -----------------------------
# DATASET (simple built-in)
# -----------------------------
data = {
    "title": [
        "Avengers","Iron Man","Thor","Captain America","Hulk",
        "Batman","Superman","Flash","Wonder Woman","Joker"
    ],
    "description": [
        "superhero team saves world",
        "genius builds iron suit hero",
        "god of thunder hero story",
        "soldier becomes super hero",
        "scientist turns into monster",
        "dark knight fights crime",
        "alien with super powers",
        "fastest man alive hero",
        "amazon warrior princess hero",
        "villain chaos mastermind"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# PREPROCESSING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["clean_text"] = df["description"].apply(clean_text)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

# -----------------------------
# SIMILARITY
# -----------------------------
similarity = cosine_similarity(tfidf_matrix)

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(title, top_n=5):
    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df.iloc[i[0]]["title"] for i in scores]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Select Movie", df["title"])

if st.button("Recommend"):
    recs = recommend(movie)
    st.write("Recommended Movies:")
    for r in recs:
        st.write("👉", r)
