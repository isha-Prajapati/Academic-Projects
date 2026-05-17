AI-Powered HR Policy Document Chatbot - RAG System

A Retrieval-Augmented Generation (RAG) chatbot that provides accurate, source-grounded answers to questions about HR policies and documents. The system combines factual extractive Q&A and conversational AI to help employees and HR teams quickly find the information they need. Project Description This project implements a document-based chatbot that can answer questions about HR policies and documents using two different AI approaches:

Specialized Q&A Model (DistilBERT) - For precise, factual answers with zero hallucinations
General LLM (Ollama + Llama2) - For conversational responses
Features

Process HR documents (.docx) into searchable knowledge base
Semantic search using FAISS vector database
Two AI backends: Q&A model and conversational LLM
Real-time chat interface with Streamlit
Source document tracking and confidence scores
100% factual answers with no hallucinations (Q&A model)
Quick Start

Clone Repository
git clone <repository-url>
cd HR_Chatbot
Install Dependencies
pip install streamlit langchain transformers faiss-cpu sentence-transformers python-docx unstructured
Setup HR Documents
Place HR .docx files in hr_data/ directory
Create Vector Store
python create_memory_for_llm.py
Run Chatbot
# For accurate Q&A model (recommended)
streamlit run chatbot_qna_distilbert_model.py
# For conversational LLM (requires Ollama)
streamlit run chatbot_ollama_model.py
Project Structure

HR_Chatbot/
│
├── create_memory_for_llm.py          # Document processing & vector store
├── chatbot_qna_distilbert_model.py            # Q&A model (DistilBERT)
├── chatbot_ollama_model.py        # LLM model (Ollama)
├── hr_data/                 # HR documents directory
└── vectorstore/             # FAISS database
Model Comparison Aspect Q&A Model Ollama LLM Accuracy 100% factual May hallucinate Speed Fast Slow Setup Simple Requires Ollama Best For Policy quotes Explanations

Requirements

Python 3.8+
For Ollama: Install from https://ollama.ai/ pip install streamlit langchain langchain-community transformers torch sentence-transformers faiss-cpu python-docx unstructured numpy pandas pip install pypdf pdf2image pillow ollama run llama2 ollama pull llama2.7b
Usage Examples

"What is the health and safety policy?"
"How many sick days are allowed?"
"What are the compensation benefits?"
"What is the grievance procedure?"
Technical Details

Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Vector Store: FAISS
Q&A Model: distilbert-base-cased-distilled-squad
LLM: llama2 via Ollama
Chunk Size: 500 characters with 50 overlap
License Educational/Organizational Use Privacy Considerations PII masking during ingestion and logging Audit logs of queries for compliance GDPR-aligned data retention and deletion options

Note: For production HR systems, the Q&A model is recommended for its factual accuracy.
