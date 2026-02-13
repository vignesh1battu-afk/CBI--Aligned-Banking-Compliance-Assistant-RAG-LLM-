import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load local .env file (for local development only)
load_dotenv()

# Try environment variable first (local)
api_key = os.getenv("OPENAI_API_KEY")

# If not found, try Streamlit Cloud secrets
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file (local) "
            "or in Streamlit Cloud secrets."
        )

client = OpenAI(api_key=api_key)


def generate_answer(query, retrieved_chunks):
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[Source {i+1} | {chunk['domain']} | {chunk['source']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a banking compliance assistant aligned with regulatory documents.

Answer the question strictly using the provided context.
Cite the source number in brackets like [Source 1] when referencing information.
Do not hallucinate.
If information is not found, say:
"Information not found in provided regulatory documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800
    )

    return response.choices[0].message.content
