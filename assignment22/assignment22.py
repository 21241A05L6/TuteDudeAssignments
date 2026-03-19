import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma

# -----------------------------
# LOAD & SPLIT (Reuse Ass21)
# -----------------------------
loader = TextLoader("data/sample.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)

print("\nChunks:", len(chunks))

# -----------------------------
# PART 1 — OPENAI EMBEDDINGS
# -----------------------------
print("\n===== TASK 1: OPENAI =====")

try:
    openai_embed = OpenAIEmbeddings()
    vectors = openai_embed.embed_documents([c.page_content for c in chunks])
    print("Vector length:", len(vectors[0]))
    print("Sample:", vectors[0][:5])
except:
    print("OpenAI key not set (optional)")

# -----------------------------
# TASK 2 — HUGGINGFACE
# -----------------------------
print("\n===== TASK 2: HUGGINGFACE =====")

hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
hf_vectors = hf_embed.embed_documents([c.page_content for c in chunks])

print("Vector length:", len(hf_vectors[0]))

# -----------------------------
# TASK 3 — COMPARISON
# -----------------------------
print("\n===== TASK 3: COMPARISON =====")
print("OpenAI → better quality, paid API")
print("HuggingFace → free, local, fast")

# -----------------------------
# PART 2 — SIMILARITY SEARCH
# -----------------------------
print("\n===== TASK 4: SEARCH FUNCTION =====")

def search(query, embeddings, texts, top_k=3):
    q_vec = embeddings.embed_query(query)
    scores = []
    for i, t in enumerate(texts):
        sim = sum([a*b for a,b in zip(q_vec, embeddings.embed_query(t))])
        scores.append((i, sim))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [texts[i[0]] for i in scores[:top_k]]

texts = [c.page_content for c in chunks]

print(search("What are embeddings?", hf_embed, texts))

# -----------------------------
# TASK 5 — LANGCHAIN SEARCH
# -----------------------------
print("\n===== TASK 5: LANGCHAIN SEARCH =====")

db = FAISS.from_documents(chunks, hf_embed)
results = db.similarity_search("vector search", k=2)

for r in results:
    print(r.page_content)

# -----------------------------
# PART 3 — OLLAMA (OPTIONAL)
# -----------------------------
print("\n===== TASK 6: OLLAMA =====")
print("Use Ollama locally with nomic-embed-text model (optional).")

# -----------------------------
# PART 4 — FAISS
# -----------------------------
print("\n===== TASK 7: FAISS =====")

db.save_local("faiss_index")
loaded = FAISS.load_local("faiss_index", hf_embed)

res = loaded.similarity_search("LLM applications")
print(res[0].page_content)

# -----------------------------
# TASK 8 — CHROMA
# -----------------------------
print("\n===== TASK 8: CHROMA =====")

chroma = Chroma.from_documents(chunks, hf_embed, persist_directory="chroma_db")
chroma.persist()

res = chroma.similarity_search("embeddings")
print(res[0].page_content)

# -----------------------------
# TASK 9 — COMPARISON
# -----------------------------
print("\n===== TASK 9: FAISS vs CHROMA =====")
print("FAISS → fast, in-memory")
print("Chroma → persistent storage")

# -----------------------------
# PART 5 — PIPELINE
# -----------------------------
print("\n===== TASK 10: PIPELINE =====")

def pipeline(query, embed_type="hf", store_type="faiss"):
    if embed_type == "hf":
        emb = hf_embed
    else:
        emb = OpenAIEmbeddings()

    if store_type == "faiss":
        store = FAISS.from_documents(chunks, emb)
    else:
        store = Chroma.from_documents(chunks, emb)

    return store.similarity_search(query, k=2)

print(pipeline("What is RAG?"))

# -----------------------------
# TASK 11 — ANSWERS
# -----------------------------
print("\n===== TASK 11: ANSWERS =====")
print("1. Embeddings convert text to vectors for meaning.")
print("2. Vector DB enables fast similarity search.")
print("3. This pipeline is core of RAG systems.")
