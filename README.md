🏦 CBI-Aligned Banking Compliance Assistant

AI-Powered Regulatory Assistant using ML Routing + RAG + LLM

🚀 Overview

The CBI-Aligned Banking Compliance Assistant is a production-grade AI system designed to answer regulatory compliance queries using official banking and financial regulation documents.

The system combines:

1) Retrieval-Augmented Generation (RAG)

2) Machine Learning Query Classification

3) FAISS Vector Search

4)  OpenAI Embeddings + LLM

5) Streamlit Production Deployment

It provides grounded, citation-based answers strictly derived from regulatory documents such as:

Consumer Protection Code

GDPR Regulation

Anti-Money Laundering Guidance

Governance Guidelines

   Architecture
1️⃣ Document Ingestion

Extracts text from regulatory PDFs

Tags each document by domain (AML, GDPR, Consumer Protection, Governance)

Splits documents into overlapping chunks

Generates embeddings using text-embedding-3-small

Stores embeddings in FAISS index

2️⃣ ML Query Classification Layer

Trained classifier predicts query domain

Routes query intelligently before retrieval

Improves search precision

3️⃣ Retrieval Layer (RAG)

Query embedding generated

FAISS similarity search retrieves top relevant chunks

Retrieved context passed to LLM

4️⃣ Grounded LLM Response

GPT model generates answer strictly from retrieved context

Includes citation references (e.g., [Source 1])

Prevents hallucination

Deployed on Streamlit Community Cloud

🔗 Live App:
https://vignesh1battu-afk-cbi--aligned-banking-compliance--uiapp-xuiqkt.streamlit.app/

⚙️ Tech Stack

Python 3.13

Streamlit

FAISS

Scikit-learn

OpenAI API

NumPy / Pandas

🔐 Security

No API keys stored in repository

Uses Streamlit secrets.toml

GitHub secret scanning protection enabled

Production deployment secured

📌 Example Query

Question:

What are the requirements for telephone contact with a consumer?

Response:

Conditions for existing customers

Conditions for non-existing customers

Time restrictions

Proper citation references

📊 Key Features

ML-powered domain routing

Grounded answers only from regulatory documents

Citation-based transparency

Clean professional UI

Deployment-ready architecture

Lazy loading optimization for performance

🎯 Use Case

Designed for:

Banking compliance teams

Risk management departments

Regulatory audit preparation

Financial services institutions