from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader

# Step 1: Load raw DOCX(s)
DATA_PATH = "hr_data/"

def load_docx_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    documents = loader.load()
    return documents

documents = load_docx_files(DATA_PATH)


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)


# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = get_embedding_model()


# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
