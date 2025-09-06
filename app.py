# x.py
# Streamlit version of the HIV/AIDS Assistant for local machine

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables (for API keys, etc.)
load_dotenv()

st.set_page_config(page_title="Uganda HIV/AIDS Assistant", layout="wide")

# --- 1. Load or Create the Vector Store with Caching ---
PDF_PATH = "Consolidated-HIV-and-AIDS-Guidelines-2022.pdf"
FAISS_INDEX_PATH = "faiss_index"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():  
    hf_embed = load_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH, 
                hf_embed, 
                allow_dangerous_deserialization=True
            )
        except Exception:
            st.warning("Error loading FAISS index. Rebuilding index.")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    text_chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(text_chunks, hf_embed)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

@st.cache_resource
def load_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    if not GROQ_API_KEY:
        st.error("❌ ERROR: GROQ_API_KEY not found. Please set it in your .env file.")
        return None
    return ChatGroq(
        model_name="llama-3.1-70b-versatile",  # ✅ Recommended stable Groq model
        temperature=0,
        api_key=GROQ_API_KEY
    )

# --- 2. Core Functions ---
def retrieve_relevant_chunks(query, vectorstore):
    if vectorstore:
        return vectorstore.similarity_search(query, k=4)
    return []

def answer_query(query, llm, vectorstore):
    if not llm:
        return "Error: Language Model unavailable."
    if not vectorstore:
        return "Error: Vector store unavailable."

    docs = retrieve_relevant_chunks(query, vectorstore)
    if not docs:
        return "I could not find relevant information in the document."

    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        "You are an HIV/AIDS assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip() if response else "Error: Empty response."
    except Exception as e:
        return f"Error invoking LLM: {e}"

# --- 3. Streamlit UI ---
def main():
    st.title("🇺🇬 Uganda HIV/AIDS Assistant Chatbot")
    st.write("Built by **Alfred Lutaaya** | Based on *Consolidated HIV and AIDS Guidelines 2022*.")

    vectorstore = load_vectorstore()
    llm = load_llm()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Enter your question:")

    if st.button("Ask") and user_input:
        answer = answer_query(user_input, llm, vectorstore)
        st.session_state.chat_history.append((user_input, answer))

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()
