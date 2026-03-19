# Assignment 20 — Recommendation System

## Dataset
Custom movie dataset (sample for demo)

---

## Steps Implemented

### Data Preprocessing
- Lowercasing
- Removing special characters

### Vectorization
- TF-IDF (max_features=50, ngrams)

### Similarity
- Cosine similarity used

### Recommendation
- Top N similar items returned

### App
- Built using Streamlit
- Dropdown + button UI

---

## Files
- app.py
- requirements.txt
- README.md

---

## Run Locally

pip install -r requirements.txt

streamlit run app.py

---

## Deployment (Render)

### Build Command
pip install -r requirements.txt

### Start Command
streamlit run app.py --server.port 10000 --server.address 0.0.0.0

---

## GitHub Steps

git init
git add .
git commit -m "assignment20"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main

---

## Notes
- Content-based recommendation only
- Simple dataset used for demo
