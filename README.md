# RAG-Powered Project Intelligence Assistant

An AI-powered search tool that lets you query 100+ real-world projects using plain English.

## How it works
1. Parses 200 PowerPoint slides and extracts project data
2. Converts project text into vectors using Sentence Transformers
3. Stores vectors in FAISS for fast similarity search
4. Retrieves top matching projects and generates answers using LLaMA3 (Groq)

## Tech Stack
- FAISS — vector similarity search
- Sentence Transformers — text embeddings
- Groq (LLaMA3) — answer generation
- Streamlit — web interface

## Setup
pip install -r requirements.txt
streamlit run app.py
