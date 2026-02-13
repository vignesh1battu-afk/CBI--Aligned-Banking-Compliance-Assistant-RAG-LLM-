import sys
import os
import streamlit as st

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="CBI Compliance Assistant",
    page_icon="🏦",
    layout="wide"
)

# -----------------------
# Styling
# -----------------------
st.markdown("""
<style>
.stApp { background-color: #f4f6f9; }
.main-title { font-size: 32px; font-weight: 700; color: #1f2937; }
.subtitle { font-size: 16px; color: #4b5563; }
.answer-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}
.source-box {
    background-color: #f9fafb;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header (Loads Immediately)
# -----------------------
st.markdown('<div class="main-title">🏦 CBI-Aligned Banking Compliance Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered regulatory assistant with ML routing and grounded retrieval.</div>', unsafe_allow_html=True)
st.write("")

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_loaded" not in st.session_state:
    st.session_state.system_loaded = False

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("⚙ Controls")
if st.sidebar.button("🗑 Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# -----------------------
# Display Chat History
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------
# Chat Input
# -----------------------
query = st.chat_input("Enter compliance question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        # Lazy import heavy modules ONLY when needed
        if not st.session_state.system_loaded:
            with st.spinner("Initializing AI system..."):
                import pickle
                from rag.vector_store import build_faiss_index, search_index
                from rag.embedding import generate_embedding
                from llm.response import generate_answer

                with open("data/embeddings.pkl", "rb") as f:
                    embedded_chunks = pickle.load(f)

                index = build_faiss_index(embedded_chunks)

                with open("ml/classifier.pkl", "rb") as f:
                    classifier = pickle.load(f)

                st.session_state.embedded_chunks = embedded_chunks
                st.session_state.index = index
                st.session_state.classifier = classifier
                st.session_state.search_index = search_index
                st.session_state.generate_embedding = generate_embedding
                st.session_state.generate_answer = generate_answer

                st.session_state.system_loaded = True

        embedded_chunks = st.session_state.embedded_chunks
        index = st.session_state.index
        classifier = st.session_state.classifier
        search_index = st.session_state.search_index
        generate_embedding = st.session_state.generate_embedding
        generate_answer = st.session_state.generate_answer

        with st.spinner("Analyzing regulatory documents..."):
            predicted_domain = classifier.predict(query)
            query_vector = generate_embedding(query)

            results = search_index(
                index,
                query_vector,
                embedded_chunks,
                top_k=5
            )

            answer = generate_answer(query, results)

        st.markdown(f"""
        <div class="answer-box">
        <b>📌 Predicted Domain:</b> {predicted_domain}
        <br><br>
        {answer}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔎 View Retrieved Sources"):
            for i, r in enumerate(results):
                st.markdown(f"""
                <div class="source-box">
                <b>Source {i+1}:</b> {r['source']} ({r['domain']})<br><br>
                {r['text'][:500]}
                </div>
                <br>
                """, unsafe_allow_html=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"**📌 Predicted Domain:** {predicted_domain}\n\n{answer}"
            }
        )
