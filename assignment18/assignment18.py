import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -----------------------------
# DATASET (Same style as Ass17)
# -----------------------------
data = [
"hello this is awesome",
"i love this product",
"visit now",
"email me please",
"this is cool",
"bad service never coming back",
"what a great day",
"numbers should be removed",
"extra spaces here",
"html bold text",
"you are amazing",
"great work done",
"terrible experience",
"best purchase ever",
"not worth money",
"highly recommended",
"so boring dull",
"amazing product great quality",
"will buy again",
"worst app ever"
]

df = pd.DataFrame({"text": data})

# -----------------------------
# PART 1 — ONE HOT ENCODING
# -----------------------------
print("\n===== TASK 1: MANUAL ONE HOT =====")

sentences = df["text"].head(5).tolist()

# vocabulary
vocab = sorted(set(" ".join(sentences).split()))
print("Vocabulary:", vocab)

# one-hot
one_hot = []
for s in sentences:
    words = s.split()
    vector = [1 if word in words else 0 for word in vocab]
    one_hot.append(vector)

print("One Hot Matrix:")
for i, vec in enumerate(one_hot):
    print(sentences[i], "->", vec)

# -----------------------------
# TASK 2 — SKLEARN ONE HOT
# -----------------------------
print("\n===== TASK 2: SKLEARN ONE HOT =====")

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(sentences)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Matrix:\n", X.toarray())

# -----------------------------
# PART 2 — BAG OF WORDS
# -----------------------------
print("\n===== TASK 3: BAG OF WORDS =====")

bow = CountVectorizer()
X_bow = bow.fit_transform(df["text"])

print("Vocabulary Size:", len(bow.vocabulary_))
print("Sample Vectors:\n", X_bow.toarray()[:3])

# -----------------------------
# TASK 4 — WORD FREQUENCY
# -----------------------------
print("\n===== TASK 4: WORD FREQUENCY =====")

word_counts = np.sum(X_bow.toarray(), axis=0)
words = bow.get_feature_names_out()

freq_df = pd.DataFrame({"word": words, "count": word_counts})
freq_df = freq_df.sort_values(by="count", ascending=False)

print("Top 10 words:\n", freq_df.head(10))
print("Least frequent words:\n", freq_df.tail(5))

print("\nExplanation: BoW counts how many times each word appears in documents.")

# -----------------------------
# PART 3 — NGRAMS
# -----------------------------
print("\n===== TASK 5: NGRAMS =====")

uni = CountVectorizer(ngram_range=(1,1))
bi = CountVectorizer(ngram_range=(2,2))
tri = CountVectorizer(ngram_range=(3,3))

X_uni = uni.fit_transform(df["text"])
X_bi = bi.fit_transform(df["text"])
X_tri = tri.fit_transform(df["text"])

print("Unigram vocab size:", len(uni.vocabulary_))
print("Bigram vocab size:", len(bi.vocabulary_))
print("Trigram vocab size:", len(tri.vocabulary_))

print("Sample unigram:", X_uni.toarray()[:2])

# -----------------------------
# TASK 6 — COMBINED NGRAM
# -----------------------------
print("\n===== TASK 6: COMBINED NGRAM =====")

combo = CountVectorizer(ngram_range=(1,2))
X_combo = combo.fit_transform(df["text"])

print("Combined vocab size:", len(combo.vocabulary_))

print("\nObservation: Bigrams add context compared to single words.")

# -----------------------------
# PART 4 — TF-IDF
# -----------------------------
print("\n===== TASK 7: TF-IDF =====")

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df["text"])

print("Vocabulary:", tfidf.get_feature_names_out()[:10])
print("Shape:", X_tfidf.shape)

# -----------------------------
# TASK 8 — BOW vs TFIDF
# -----------------------------
print("\n===== TASK 8: COMPARISON =====")

tfidf_vals = np.mean(X_tfidf.toarray(), axis=0)

tfidf_df = pd.DataFrame({
    "word": tfidf.get_feature_names_out(),
    "tfidf": tfidf_vals
}).sort_values(by="tfidf", ascending=False)

print("High TF-IDF:\n", tfidf_df.head(5))
print("Low TF-IDF:\n", tfidf_df.tail(5))

print("\nExplanation: TF-IDF reduces importance of common words.")

# -----------------------------
# PART 5 — PARAMETERS
# -----------------------------
print("\n===== TASK 9: PARAMETERS =====")

vec1 = CountVectorizer(max_features=10)
vec2 = CountVectorizer(min_df=2)
vec3 = CountVectorizer(max_df=0.8)

print("Max features vocab:", len(vec1.fit(df["text"]).vocabulary_))
print("Min_df vocab:", len(vec2.fit(df["text"]).vocabulary_))
print("Max_df vocab:", len(vec3.fit(df["text"]).vocabulary_))

# -----------------------------
# TASK 10 — CONCEPTS
# -----------------------------
print("\n===== TASK 10: ANSWERS =====")

print("1. One-hot uses binary presence, BoW uses counts.")
print("2. N-grams increase dimensionality.")
print("3. Use TF-IDF when common words dominate.")
print("4. Limitation: ignores word meaning/context.")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("vectorized_data.csv", index=False)
print("\nSaved vectorized_data.csv")
