import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load local .env file (only works locally)
load_dotenv()

# Try to get key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# If not found (e.g., on Streamlit Cloud), use Streamlit secrets
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file (local) "
            "or in Streamlit Cloud secrets."
        )

client = OpenAI(api_key=api_key)


def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def embed_chunks(chunked_docs):
    for chunk in chunked_docs:
        chunk["embedding"] = generate_embedding(chunk["text"])
    return chunked_docs
