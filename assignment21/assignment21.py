import os
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# -----------------------------
# TASK 1 — TEXT LOADER
# -----------------------------
print("\n===== TASK 1: TEXT LOADER =====")
text_loader = TextLoader("data/sample.txt")
docs = text_loader.load()

print("Docs loaded:", len(docs))
print("Content preview:", docs[0].page_content)
print("Metadata:", docs[0].metadata)

# -----------------------------
# TASK 2 — CSV LOADER
# -----------------------------
print("\n===== TASK 2: CSV LOADER =====")
csv_loader = CSVLoader(file_path="data/sample.csv")
csv_docs = csv_loader.load()

print("Sample row:", csv_docs[0].page_content)

# -----------------------------
# TASK 3 — PDF LOADER
# -----------------------------
print("\n===== TASK 3: PDF LOADER =====")
try:
    pdf_loader = PyPDFLoader("data/sample.pdf")
    pdf_docs = pdf_loader.load()
    print("Pages:", len(pdf_docs))
    print("Sample:", pdf_docs[0].page_content[:200])
except:
    print("No PDF found (optional task)")

# -----------------------------
# TASK 4 — DIRECTORY LOADER
# -----------------------------
print("\n===== TASK 4: DIRECTORY LOADER =====")
dir_loader = DirectoryLoader("data/")
all_docs = dir_loader.load()
print("Total docs loaded:", len(all_docs))

# -----------------------------
# TASK 5 — WEB LOADER
# -----------------------------
print("\n===== TASK 5: WEB LOADER =====")
try:
    web_loader = WebBaseLoader("https://example.com")
    web_docs = web_loader.load()
    print("Web content:", web_docs[0].page_content[:200])
except:
    print("Web loading skipped")

# -----------------------------
# TASK 6 — CONCEPTS
# -----------------------------
print("\n===== TASK 6: CONCEPTS =====")
print("Large docs exceed token limits.")
print("Chunking improves retrieval and efficiency.")

# -----------------------------
# TASK 7 — CHARACTER SPLITTER
# -----------------------------
print("\n===== TASK 7: CHARACTER SPLITTER =====")
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(docs)

print("Chunks:", len(chunks))
print("Sample chunk:", chunks[0].page_content)

# -----------------------------
# TASK 8 — RECURSIVE SPLITTER
# -----------------------------
print("\n===== TASK 8: RECURSIVE SPLITTER =====")
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
rec_chunks = recursive_splitter.split_documents(docs)

print("Chunks:", len(rec_chunks))

# -----------------------------
# TASK 9 — STRUCTURE SPLIT
# -----------------------------
print("\n===== TASK 9: STRUCTURE SPLIT =====")
print("Recursive splitter preserves structure better.")

# -----------------------------
# TASK 10 — SEMANTIC SPLIT
# -----------------------------
print("\n===== TASK 10: SEMANTIC =====")
print("Semantic splitting uses embeddings to split meaningful chunks.")

# -----------------------------
# TASK 11 — PIPELINE
# -----------------------------
print("\n===== TASK 11: PIPELINE =====")

def load_and_split_documents(path):
    loader = DirectoryLoader(path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    return splitter.split_documents(docs)

pipeline_chunks = load_and_split_documents("data/")
print("Pipeline chunks:", len(pipeline_chunks))

# -----------------------------
# TASK 12 — ANSWERS
# -----------------------------
print("\n===== TASK 12: ANSWERS =====")
print("1. TextLoader for txt, CSVLoader for csv, PyPDFLoader for pdf, WebBaseLoader for web.")
print("2. Small text → Character splitter, Large PDFs → Recursive splitter, Web → Recursive.")
print("3. Overlap keeps context between chunks.")
