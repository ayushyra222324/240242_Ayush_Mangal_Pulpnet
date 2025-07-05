import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# ---------------- Load chunks from documents.txt ----------------

@st.cache_data
def load_chunks():
    with open("documents.txt", "r", encoding="utf-8") as f:
        return [chunk.strip() for chunk in f.read().split("\n---\n") if chunk.strip()]

chunks = load_chunks()

if not chunks:
    st.error("No chunks found in documents.txt. Please check the format.")
    st.stop()

# ---------------- Load Models & FAISS Index ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("question-answering", model="deepset/roberta-large-squad2")
    index = faiss.read_index("iitk_index.faiss")
    return embedder, qa_model, index

embedder, qa_pipeline, index = load_models()

# ---------------- UI ----------------
st.title(" PULPNET - IIT Kanpur Chatbot")
st.markdown("Ask me anything about **IITK** (faculty, students, clubs, academics, etc.)")

user_question = st.text_input(" Your Question")

if user_question:
    with st.spinner(" Searching..."):

        # Embed the question
        question_embedding = embedder.encode([user_question])
        D, I = index.search(np.array(question_embedding), k=3)

        # Retrieve top matching text chunks
        top_context = " ".join([chunks[i] for i in I[0] if i < len(chunks)])

        if not top_context.strip():
            st.warning(" No relevant information found.")
        else:
            # Ask QA model
            result = qa_pipeline(question=user_question, context=top_context)
            st.success(f" Answer: {result['answer']}")
