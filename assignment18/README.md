# Assignment 18 — Text Vectorization Techniques

## Dataset
20 cleaned text samples (same as previous assignment).

---

## Techniques Implemented

### One-Hot Encoding
- Manual implementation
- Using CountVectorizer(binary=True)

### Bag of Words
- CountVectorizer used
- Vocabulary size and vectors displayed

### Word Frequency
- Top frequent words
- Least frequent words

### N-Grams
- Unigrams, Bigrams, Trigrams
- Combined n-grams (1,2)

### TF-IDF
- TfidfVectorizer used
- Matrix shape and vocabulary shown

### Comparison
- High vs Low TF-IDF words

### Parameters
- max_features
- min_df
- max_df

---

## Output
- vectorized_data.csv

---

## Libraries Used
- pandas
- numpy
- scikit-learn

---

## How to Run

pip install pandas numpy scikit-learn

python assignment18.py

---

## Notes
- No ML models used
- Focus only on feature extraction
- Clean and simple implementation
