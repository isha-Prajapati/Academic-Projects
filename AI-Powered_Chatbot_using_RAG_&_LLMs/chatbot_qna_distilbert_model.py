# chatbot_qa_fixed.py
import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

# ----------------- Load FAISS Vector Store -----------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ----------------- Improved Q&A Pipeline -----------------
@st.cache_resource
def load_qa_model():
    """Load the Q&A model with better settings"""
    try:
        # Using a more capable Q&A model
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",  # Reliable and fast
            tokenizer="distilbert-base-cased-distilled-squad"
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

qa_pipeline = load_qa_model()

# # ----------------- Prompt Template for Better Answers -----------------
# def create_qa_prompt(question, context):
#     """Create a structured prompt for better Q&A performance"""
#     prompt_template = f"""
# Based on the following HR document context, answer the question clearly and concisely.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# INSTRUCTIONS:
# - Answer directly based only on the context provided
# - Keep the answer to 2-3 sentences maximum
# - If the answer cannot be found in the context, say "I'm sorry, this information is not available in the HR documents."
# - Be specific and factual

# ANSWER:
# """
#     return prompt_template.strip()

# ----------------- Optimized Context Preparation -----------------
def prepare_context(documents):
    """Prepare context in a way that helps the Q&A model perform better"""
    if not documents:
        return ""
    
    # Clean and combine documents
    cleaned_docs = []
    for doc in documents:
        content = doc.page_content
        
        # Remove excessive whitespace but keep structure
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove markdown symbols that might confuse the model
        content = re.sub(r'[#*_\-]{2,}', '', content)
        
        cleaned_docs.append(content)
    
    # Combine with separators
    context = "\n\n".join(cleaned_docs)
    return context

# ----------------- Improved Answer Generation -----------------
def generate_answer(question, context):
    """Generate answer using Q&A pipeline with optimized parameters"""
    if not context:
        return "I'm sorry, I couldn't find relevant information in the HR documents."
    
    try:
        # Run the Q&A model with correct parameters
        result = qa_pipeline({
            'question': question,
            'context': context,
            'max_answer_len': 500,  # Correct parameter name
            'handle_impossible_answer': True
        })
        
        answer = result['answer'].strip()
        confidence = result['score']
        
        # Show confidence in sidebar for debugging
        st.sidebar.metric("Model Confidence", f"{confidence:.3f}")
        
        # More lenient confidence threshold with better validation
        if confidence > 0.01 and len(answer) > 5:
            # Clean up the answer
            if not answer.endswith(('.', '!', '?')):
                answer += '.'
            return answer
        else:
            return "I'm sorry, I couldn't extract a clear answer from the HR documents for this question."
            
    except Exception as e:
        # If there's an error with parameters, try without them
        try:
            result = qa_pipeline({
                'question': question,
                'context': context
            })
            
            answer = result['answer'].strip()
            confidence = result['score']
            
            if confidence > 0.01 and len(answer) > 5:
                if not answer.endswith(('.', '!', '?')):
                    answer += '.'
                return answer
            else:
                return "I'm sorry, I couldn't extract a clear answer from the HR documents for this question."
                
        except Exception as e2:
            return f"I'm having trouble processing your question right now. Please try again."

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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents..."):
            try:
                # Get relevant documents
                docs = retriever.get_relevant_documents(user_input)
                
                if docs:
                    # Prepare optimized context
                    context = prepare_context(docs)
                    
                    if qa_pipeline:
                        answer = generate_answer(user_input, context)
                    else:
                        answer = "Q&A model is not available. Please check the model setup."
                        
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show source info
                    with st.expander("📄 Source Information"):
                        st.write(f"Found {len(docs)} relevant document(s)")
                        for i, doc in enumerate(docs[:3]):
                            st.write(f"**Document {i+1}:**")
                            clean_content = re.sub(r'\s+', ' ', doc.page_content)
                            display_text = clean_content[:400] + "..." if len(clean_content) > 400 else clean_content
                            st.text(display_text)
                else:
                    answer = "I'm sorry, I couldn't find any information about that in the HR documents."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error while processing your question."})

# Clear conversation
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Model info
st.sidebar.info("Using: DistilBERT Q&A Model")