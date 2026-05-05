# 🚀 RAG Book Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that allows users to upload PDFs and ask intelligent questions based on document content.

---

## 📌 Features

- 📄 Upload custom PDF documents  
- 🔍 Semantic search using embeddings  
- 🧠 Context-aware answers using LLM  
- ⚡ Fast retrieval using Chroma vector database  
- 🎯 Answers strictly based on document context  
- 🌐 Interactive UI with Streamlit  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Mistral AI  
- **Embeddings:** HuggingFace (sentence-transformers/all-MiniLM-L6-v2)  
- **Vector DB:** Chroma  
- **Framework:** LangChain  

---

## ⚙️ How It Works

1. Upload PDF  
2. Split text into chunks  
3. Convert chunks → embeddings  
4. Store embeddings in Chroma DB  
5. Retrieve relevant chunks  
6. Generate answer using LLM  

---

## 📂 Project Structure

.
├── app.py                  # Streamlit UI  
├── create_database.py      # PDF → embeddings → Chroma  
├── main.py                 # RAG pipeline logic  
├── document_loaders/       # PDF loading logic  
├── retrievers/             # Retrieval strategies  
├── vector_store/           # Database handling  
├── requirements.txt  

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

MISTRAL_API_KEY=your_mistral_api_key  




## ▶️ Run Locally

```bash
git clone https://github.com/GovindNair07/Rag-Implementation-Project.git
cd Rag-Implementation-Project

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

streamlit run app.py