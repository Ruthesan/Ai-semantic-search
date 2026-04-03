import streamlit as st
# from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    USE_EMBEDDINGS = True
except:
    USE_EMBEDDINGS = False
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pypdf import PdfReader
import os

# Optional OpenAI (will fallback if quota error)
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    USE_OPENAI = True
except:
    USE_OPENAI = False


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI System", layout="wide")

# -------------------------------
# LOGIN SYSTEM
# -------------------------------
users = {
    "admin": "1234",
    "user": "password"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_model()

# -------------------------------
# MEMORY
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# VECTOR DATABASE
# -------------------------------
class VectorDatabase:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        if not docs:
            return

        new_embeddings = embedding_model.encode(docs)

        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))

        self.documents.extend(docs)

    def search(self, query, top_k=3):
        if len(self.documents) == 0:
            return []

        query_embedding = embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[i], similarities[i]) for i in top_indices]


# -------------------------------
# TEXT PROCESSING
# -------------------------------
def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# -------------------------------
# INITIALIZE DB
# -------------------------------
@st.cache_resource
def init_db():
    return VectorDatabase()

db = init_db()

# -------------------------------
# FILE UPLOAD SYSTEM
# -------------------------------
st.sidebar.title("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload TXT or PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    new_docs = []

    for file in uploaded_files:
        if file.type == "text/plain":
            text = file.read().decode("utf-8")

        elif file.type == "application/pdf":
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

        chunks = chunk_text(text)
        new_docs.extend(chunks)

    db.add_documents(new_docs)
    st.sidebar.success("Documents added!")

# -------------------------------
# LLM RESPONSE
# -------------------------------
def generate_answer(query, context_chunks):
    context = "\n\n".join([doc for doc, _ in context_chunks])

    # Fallback if no OpenAI
    if not USE_OPENAI:
        return f"[MOCK ANSWER]\nQuery: {query}\nContext: {context[:200]}..."

    try:
        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"[ERROR / FALLBACK]\n{str(e)}\n\nUsing context:\n{context[:200]}..."


# -------------------------------
# MAIN UI
# -------------------------------
st.title("🔍 AI Semantic Search + Chat System")

query = st.text_input("Ask a question:")

if query:
    results = db.search(query)

    if not results:
        st.warning("No documents found. Upload files first.")
    else:
        answer = generate_answer(query, results)

        # Save memory
        st.session_state.history.append((query, answer))

        # Show answer
        st.subheader("🤖 Answer")
        st.write(answer)

        # Show sources
        st.subheader("📚 Sources")
        for doc, score in results:
            st.info(f"{doc[:200]}...\nScore: {score:.4f}")

# -------------------------------
# HISTORY
# -------------------------------
st.sidebar.title("🧠 History")

for q, a in reversed(st.session_state.history):
    st.sidebar.write(f"Q: {q}")
    st.sidebar.write(f"A: {a[:100]}...")
    st.sidebar.markdown("---")
