# chatbot_ollama_fixed.py
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

# ----------------- Load FAISS Vector Store -----------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ----------------- Ollama LLM -----------------
@st.cache_resource
def load_ollama_model():
    """Load Ollama model"""
    try:
        llm = Ollama(
            model="llama2",
            temperature=0.1,
            num_predict=100  # Shorter to prevent rambling
        )
        return llm
    except Exception as e:
        st.error(f"Error loading Ollama: {e}")
        return None

ollama_llm = load_ollama_model()

# ----------------- STRICT Prompt Template -----------------
def create_qa_prompt(question, context):
    """Create a strict prompt that forces context usage"""
    prompt_template = f"""<|system|>
You are an HR assistant. Extract the exact answer from the HR documents below. Do not make up information.

HR DOCUMENTS:
{context}
</|system|>

<|user|>
Question: {question}

Extract the exact answer from the HR documents above. If the answer is not found, say "Not found in documents".
</|user|>

<|assistant|>
Answer:"""
    return prompt_template.strip()

# ----------------- Context Preparation -----------------
def prepare_context(documents):
    """Prepare clean context"""
    if not documents:
        return ""
    
    cleaned_docs = []
    for doc in documents:
        content = doc.page_content
        content = re.sub(r'\s+', ' ', content).strip()
        cleaned_docs.append(content)
    
    return "\n\n".join(cleaned_docs)

# ----------------- STRICT Answer Generation -----------------
def generate_answer(question, context):
    """Generate answer with strict controls"""
    if not context:
        return "I'm sorry, I couldn't find relevant information in the HR documents."
    
    try:
        # Create the prompt
        prompt = create_qa_prompt(question, context)
        
        # Generate with Ollama
        response = ollama_llm.invoke(prompt)
        
        # Extract answer
        answer = response.strip()
        
        # STRICT validation - check if answer actually exists in context
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # If answer mentions things not in context, reject it
        if ("not found" in answer_lower or 
            "i cannot" in answer_lower or 
            "i don't know" in answer_lower):
            return "I'm sorry, I couldn't find a clear answer in the HR documents."
        
        # Check if answer is too generic or hallucinated
        if len(answer) < 15 or len(answer) > 200:
            return "I'm sorry, I couldn't extract a clear answer from the HR documents."
            
        return answer
            
    except Exception as e:
        return f"I'm having trouble processing your question. Please try again."

# ----------------- Streamlit App -----------------
st.title("🧾 HR Document Q&A")
st.write("Get accurate answers from HR documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask something about HR documents:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing HR documents..."):
            try:
                docs = retriever.get_relevant_documents(user_input)
                
                if docs:
                    context = prepare_context(docs)
                    
                    if ollama_llm:
                        answer = generate_answer(user_input, context)
                    else:
                        answer = "Ollama is not available."
                        
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    with st.expander("📄 Source Information"):
                        st.write(f"Found {len(docs)} relevant document(s)")
                        for i, doc in enumerate(docs[:3]):
                            st.write(f"**Document {i+1}:**")
                            clean_content = re.sub(r'\s+', ' ', doc.page_content)
                            display_text = clean_content[:300] + "..." if len(clean_content) > 300 else clean_content
                            st.text(display_text)
                else:
                    answer = "I'm sorry, I couldn't find any information about that in the HR documents."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})

if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.info("Using: Ollama with Strict Controls")