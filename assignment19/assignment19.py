import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# -----------------------------
# DATASET (30+ sentences)
# -----------------------------
data = [
"this product is amazing",
"i love this item",
"this is a great product",
"bad quality product",
"not worth money",
"highly recommended item",
"this product is good",
"very bad experience",
"i will buy again",
"worst product ever",
"great quality and design",
"poor performance overall",
"excellent item loved it",
"not satisfied with product",
"amazing experience overall",
"terrible service and quality",
"very happy with purchase",
"best item ever bought",
"waste of money",
"good value for money",
"nice product quality",
"awful experience",
"superb item",
"great performance",
"bad product quality",
"i like this",
"i hate this",
"very good item",
"not good not bad",
"excellent quality"
]

df = pd.DataFrame({"text": data})

# -----------------------------
# TOKENIZATION
# -----------------------------
sentences = [text.split() for text in df["text"]]

# -----------------------------
# PART 1 — CONCEPTUAL
# -----------------------------
print("\n===== TASK 1: CONCEPTS =====")
print("1. Word embeddings are dense vector representations of words.")
print("2. One-hot and BoW fail because they ignore meaning and context.")
print("3. Embeddings capture semantic relationships.")

# -----------------------------
# PART 2 — WORD2VEC OVERVIEW
# -----------------------------
print("\n===== TASK 2: DEFINITIONS =====")
print("Vocabulary: set of unique words")
print("Context window: number of surrounding words")
print("Embedding dimension: size of vector")

# -----------------------------
# TASK 3 — CBOW vs SKIPGRAM
# -----------------------------
print("\n===== TASK 3: CBOW vs SKIPGRAM =====")
print("CBOW predicts word from context, faster.")
print("Skip-gram predicts context from word, better for rare words.")

# -----------------------------
# TASK 4 — NN INTUITION
# -----------------------------
print("\n===== TASK 4: NN INTUITION =====")
print("Input -> Hidden (embedding) -> Output")
print("Weights become word embeddings.")

# -----------------------------
# PART 3 — TRAINING
# -----------------------------
print("\n===== TASK 6: CBOW MODEL =====")

cbow_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0
)

print("Vocabulary size:", len(cbow_model.wv))
print("Vector for 'product':\n", cbow_model.wv['product'][:5])

# -----------------------------
# TASK 7 — SKIPGRAM
# -----------------------------
print("\n===== TASK 7: SKIPGRAM MODEL =====")

sg_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1
)

print("Vocabulary size:", len(sg_model.wv))

# -----------------------------
# TASK 8 — SIMILARITY
# -----------------------------
print("\n===== TASK 8: SIMILARITY =====")

print("Words similar to 'good':")
print(cbow_model.wv.most_similar('good'))

try:
    print("\nAnalogy (good - bad + excellent):")
    print(cbow_model.wv.most_similar(positive=['excellent','bad'], negative=['good']))
except:
    print("Analogy not strong due to small dataset")

# -----------------------------
# PART 4 — OBSERVATIONS
# -----------------------------
print("\n===== TASK 10: ANSWERS =====")

print("1. CBOW is faster, Skip-gram better for rare words.")
print("2. Word2Vec captures meaning, TF-IDF does not.")
print("3. Needs large data, struggles with rare contexts.")
print("4. Context still matters → leads to transformers.")

# -----------------------------
# SAVE
# -----------------------------
df.to_csv("word2vec_data.csv", index=False)
print("\nSaved word2vec_data.csv")
