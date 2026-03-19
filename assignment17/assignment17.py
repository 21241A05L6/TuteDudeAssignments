import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# CREATE DATASET (20 TEXTS)
# -----------------------------
data = [
"Hello!!! This is AWESOME 😊",
"I loooove this product!!!",
"Visit https://example.com now",
"Email me at test@gmail.com",
"This is soooo coool!!!",
"Bad service :( never coming back",
"OMG!!! what a great day",
"Numbers 12345 should be removed",
"Extra   spaces   here",
"HTML <b>bold</b> text",
"U r amazing bro",
"gr8 work done",
"Terrible experience!!!",
"Best purchase ever",
"Not worth the money",
"Highly recommended!!!",
"So boring and dull",
"Amazing product with great quality",
"Will buy again!!!",
"Worst app ever"
]

df = pd.DataFrame({"text": data})

# -----------------------------
# TASK 1 — UNDERSTANDING
# -----------------------------
print("FIRST 5 TEXTS:")
print(df.head())

df["length"] = df["text"].apply(len)
print("\nTEXT LENGTHS:")
print(df["length"].head())

# -----------------------------
# TASK 2 — BASIC CLEANING
# -----------------------------
def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text_basic"] = df["text"].apply(basic_clean)

print("\nBASIC CLEANED:")
print(df[["text","clean_text_basic"]].head())

# -----------------------------
# TASK 3 — ADVANCED CLEANING
# -----------------------------
def advanced_clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df["clean_text_advanced"] = df["text"].apply(advanced_clean)

# -----------------------------
# TASK 4 — REMOVE STOPWORDS
# -----------------------------
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])

df["text_no_stopwords"] = df["clean_text_advanced"].apply(remove_stopwords)

# -----------------------------
# TASK 5 — SLANG & REPEATED
# -----------------------------
slang_dict = {"u":"you","gr8":"great"}

def slang_replace(text):
    words = text.split()
    words = [slang_dict.get(w,w) for w in words]
    return " ".join(words)

df["text_slang_fixed"] = df["text_no_stopwords"].apply(slang_replace)

# -----------------------------
# TASK 6 — TOKENIZATION
# -----------------------------
print("\nTOKENIZATION:")
for i in range(3):
    print(word_tokenize(df["text"][i]))

# -----------------------------
# TASK 7 — STEMMING
# -----------------------------
stemmer = PorterStemmer()

def stem_text(text):
    return " ".join([stemmer.stem(w) for w in text.split()])

df["stemmed"] = df["text_slang_fixed"].apply(stem_text)

# -----------------------------
# TASK 8 — LEMMATIZATION
# -----------------------------
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])

df["lemmatized"] = df["text_slang_fixed"].apply(lemmatize_text)

# -----------------------------
# TASK 9 — FINAL PIPELINE
# -----------------------------
def nlp_preprocess(text):
    text = basic_clean(text)
    text = advanced_clean(text)
    text = remove_stopwords(text)
    text = slang_replace(text)
    text = lemmatize_text(text)
    return text

df["final_clean_text"] = df["text"].apply(nlp_preprocess)

print("\nFINAL CLEANED TEXT:")
print(df[["text","final_clean_text"]].head())

# -----------------------------
# TASK 10 — INSIGHTS
# -----------------------------
print("\nINSIGHTS:")
print("1. Basic cleaning removes punctuation and case differences.")
print("2. Advanced cleaning removes URLs, emails, HTML.")
print("3. Stopwords removal reduces noise.")
print("4. Lemmatization keeps meaningful root words.")
print("5. Preprocessing is essential for NLP models.")

df.to_csv("processed_text.csv", index=False)
print("\nSaved processed_text.csv")
