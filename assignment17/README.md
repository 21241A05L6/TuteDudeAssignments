# Assignment 17 — NLP Preprocessing Pipeline

## Dataset
Custom dataset of 20 text samples (reviews, messages).

---

## Tasks Covered

### Basic Cleaning
- Lowercasing
- Removing punctuation
- Removing numbers
- Removing extra spaces

### Advanced Cleaning
- Remove URLs
- Remove emails
- Remove HTML tags
- Remove special characters

### Stopwords
- Removed using NLTK

### Tokenization
- Word tokenization
- Sentence tokenization

### Stemming
- Porter Stemmer applied

### Lemmatization
- WordNet Lemmatizer applied

### NLP Pipeline
Combined function for:
- Cleaning
- Stopword removal
- Tokenization
- Lemmatization

---

## Output
- processed_text.csv

---

## Libraries Used
- pandas
- nltk
- re

---

## How to Run

pip install pandas nltk

python assignment17.py

---

## Notes
- No ML models used (as per restriction)
- Focus only on text preprocessing
