import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader, YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# -----------------------------
# SET API KEY
# -----------------------------
# export OPENAI_API_KEY="your_key"

llm = ChatOpenAI(temperature=0)

# -----------------------------
# TASK 1 — BASIC PROMPT
# -----------------------------
print("\n===== TASK 1 =====")
response = llm.predict("What is RAG in simple terms?")
print(response)

# -----------------------------
# TASK 2 — WIKIPEDIA RETRIEVER
# -----------------------------
print("\n===== TASK 2 =====")
wiki_docs = WikipediaLoader(query="Artificial Intelligence").load()
print("Docs:", len(wiki_docs))
print(wiki_docs[0].page_content[:200])

# -----------------------------
# TASK 3 — VECTOR STORE
# -----------------------------
print("\n===== TASK 3 =====")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(wiki_docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()
results = retriever.get_relevant_documents("What is AI?")
print(results[0].page_content[:200])

# -----------------------------
# PART 3 — ADVANCED RETRIEVERS
# -----------------------------

# TASK 4 — MMR
print("\n===== TASK 4: MMR =====")
mmr_retriever = db.as_retriever(search_type="mmr")
mmr_results = mmr_retriever.get_relevant_documents("AI applications")
print("MMR:", mmr_results[0].page_content[:200])

# TASK 5 — MULTI QUERY
print("\n===== TASK 5: MULTI QUERY =====")
multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
multi_results = multi_retriever.get_relevant_documents("AI uses")
print("MultiQuery:", multi_results[0].page_content[:200])

# TASK 6 — CONTEXTUAL COMPRESSION
print("\n===== TASK 6: COMPRESSION =====")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

comp_results = compression_retriever.get_relevant_documents("AI benefits")
print("Compressed:", comp_results[0].page_content[:200])

# -----------------------------
# PART 4 — YOUTUBE RAG
# -----------------------------
print("\n===== TASK 7: YOUTUBE LOAD =====")

video_url = "https://www.youtube.com/watch?v=aircAruvnKk"

yt_loader = YoutubeLoader.from_youtube_url(video_url)
yt_docs = yt_loader.load()

yt_chunks = splitter.split_documents(yt_docs)

yt_db = FAISS.from_documents(yt_chunks, embeddings)
yt_retriever = yt_db.as_retriever()

# -----------------------------
# TASK 9 — RAG CHATBOT
# -----------------------------
print("\n===== TASK 9: YOUTUBE RAG =====")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=yt_retriever
)

query = "What is neural network?"
answer = qa.run(query)

print("Q:", query)
print("A:", answer)

# -----------------------------
# TASK 10 — TESTING
# -----------------------------
print("\n===== TASK 10 =====")
questions = [
    "What is deep learning?",
    "How do neural networks work?",
    "What is training data?"
]

for q in questions:
    print("\nQ:", q)
    print("A:", qa.run(q))

# -----------------------------
# TASK 11 — ANSWERS
# -----------------------------
print("\n===== TASK 11 =====")
print("1. RAG uses external data, prompting does not.")
print("2. Vector stores enable fast retrieval.")
print("3. MMR improves diversity.")
print("4. Multi-query improves recall.")
print("5. Compression removes irrelevant data.")
