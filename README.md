# 🔍 AI Document Search & Chat System (RAG-Based)

An AI-powered application that allows users to upload documents, perform semantic search, and get intelligent answers using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 🔎 Semantic search using embeddings (Sentence Transformers)
* 📂 Upload and analyze TXT & PDF documents
* 🤖 Context-aware AI responses (RAG pipeline)
* 🧠 Chat memory (conversation history)
* 🔐 Simple user authentication
* 🌐 Deployed web app using Streamlit
* ⚡ Works with or without OpenAI API (fallback mode)

---

## 🧠 How It Works

1. Documents are uploaded and split into chunks
2. Each chunk is converted into vector embeddings
3. User query is also embedded
4. System finds the most similar chunks (semantic search)
5. Retrieved context is sent to the AI model
6. AI generates a relevant answer

---

## 🏗️ Tech Stack

* Python
* Streamlit
* Sentence Transformers
* Scikit-learn (cosine similarity)
* NumPy
* PyPDF
* OpenAI API (optional)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run ai_system_app.py
```

---

## 🌍 Live Demo

👉 Add your deployed link here (Streamlit Cloud)

---

## 🔐 Environment Variables

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

---

## 📂 Supported Files

* .txt
* .pdf

---

## 💡 Example Use Cases

* Chat with personal notes
* Search through documents
* AI-powered knowledge assistant
* Research assistant

---

## 🚀 Future Improvements

* FAISS for faster search (large datasets)
* User accounts with database
* Multi-document collections
* Better UI/UX

---

## 👨‍💻 Author

Your Name

---

## ⭐ If you found this useful

Give the repo a star ⭐
