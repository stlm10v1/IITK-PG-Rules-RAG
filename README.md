# 🎓 📜 IITK PG Rules & FAQ Assistant | Advanced RAG System

This project performs advanced **Retrieval-Augmented Generation (RAG)** to answer complex policy questions regarding the IIT Kanpur Postgraduate (PG) Manual and FAQs. The primary goal is to build a highly resilient and accurate AI assistant utilizing **Hierarchical Routing** (FAQ → PG Manual → Web Search) and a **Multi-Model Fallback Architecture** (Gemini 1.5 Flash, Gemini 1.5 Flash-8B, Groq Llama 3).

---

## 📍 Project Highlights

* Developed an end-to-end NLP query expansion layer mapping colloquial student slang to official academic terms.
* Built a robust multi-model fallback chain in LangChain to ensure zero downtime during API rate limits.
* Integrated zero-hallucination safeguards with exact page-level PDF citations for every generated answer.
* Deployed the final hierarchical RAG pipeline as an interactive, frosted-glass **Streamlit web app**.

---

## 🚀 Streamlit Web App

Try the model live here:
👉 [IITK PG Rules Assistant Web App](https://iitk-rag-assistant.streamlit.app/)

---

## ⚙️ Technologies Used

* Python
* LangChain
* Google Gemini APIs & Groq (Llama 3)
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* ChromaDB (Vector Database)
* DuckDuckGo Web Search API
* Streamlit

---

## 📁 Files Included

* `RAG_VScode.ipynb` – Pipeline development, chunking logic, and LLM-as-a-Judge evaluation metrics
* `app.py` – Streamlit app code for deployment and UI rendering
* `PGRulesFAQ.pdf` & `PG-Manual.pdf` – Source documents for the vector database context
* `9227948.jpg` – Background image for the web interface
* `requirements.txt` – Python dependencies

---

## 🧠 Learning Outcomes

* Deep understanding of Retrieval-Augmented Generation (RAG) architecture and semantic vector search.
* Comparison and integration of multiple Large Language Models (LLMs) to create fault-tolerant systems.
* End-to-end deployment of generative AI models and data pipelines using Streamlit Cloud.
* Practical experience with prompt engineering, text normalization, and handling API rate limits gracefully.

---

## 👨‍💻 Author

**Vikram Anand**  
M.Tech, Department of Management Sciences (DoMS)  
IIT Kanpur
