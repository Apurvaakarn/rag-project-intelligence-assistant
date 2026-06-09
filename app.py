import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
from groq import Groq


# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Project Repository Assistant",
    page_icon="🚀",
    layout="wide"
)


# --------------------------------
# TECHNOLOGY EXTRACTOR
# --------------------------------
def extract_tech_stack(text):
    technologies = [
        "Python", "Java", "Kafka", "Spark", "TensorFlow", "PyTorch",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Redis",
        "MongoDB", "PostgreSQL", "Elasticsearch", "Airflow", "Snowflake",
        "Flink", "Tableau", "Power BI", "XGBoost", "NLP", "LLM", "OpenAI"
    ]
    found = []
    for tech in technologies:
        if tech.lower() in text.lower():
            found.append(tech)
    return found


# --------------------------------
# DOMAIN EXTRACTOR
# --------------------------------
def extract_domain(text):
    domains = [
        "FinTech", "Healthcare", "E-Commerce", "Retail", "Education",
        "Manufacturing", "Logistics", "B2B SaaS", "Insurance", "Telecom", "Banking"
    ]
    for domain in domains:
        if domain.lower() in text.lower():
            return domain
    return "Not Specified"


# --------------------------------
# TEAM SIZE EXTRACTOR
# --------------------------------
def extract_team_size(text):
    match = re.search(r"Team Size\s*:?\s*(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "N/A"


# --------------------------------
# DURATION EXTRACTOR
# --------------------------------
def extract_duration(text):
    match = re.search(r"(\d+\s*(weeks|months))", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "N/A"


# --------------------------------
# PROBLEM + SOLUTION EXTRACTOR
# --------------------------------
def extract_problem_solution(text):
    problem = "Not Found"
    solution = "Not Found"
    text_upper = text.upper()

    if "PROBLEM STATEMENT" in text_upper:
        start = text_upper.find("PROBLEM STATEMENT")
        end = text_upper.find("SOLUTION")
        if end > start:
            problem = text[start:end]

    if "SOLUTION" in text_upper:
        start = text_upper.find("SOLUTION")
        end = len(text)
        if "PROJECT DETAILS" in text_upper:
            end = text_upper.find("PROJECT DETAILS")
        solution = text[start:end]

    return problem[:300], solution[:300]


# --------------------------------
# GROQ ANSWER GENERATOR
# --------------------------------
def generate_llm_answer(query, top_projects):
    # Build context from retrieved projects
    context_parts = []
    for i, proj in enumerate(top_projects, 1):
        text = proj["text"]
        slide = proj["slide"]
        team = extract_team_size(text)
        duration = extract_duration(text)
        domain = extract_domain(text)
        techs = extract_tech_stack(text)
        problem, solution = extract_problem_solution(text)

        context_parts.append(
            f"--- Project {i} (Slide {slide}) ---\n"
            f"Domain: {domain}\n"
            f"Team Size: {team} | Duration: {duration}\n"
            f"Technologies: {', '.join(techs) if techs else 'N/A'}\n"
            f"Problem: {problem}\n"
            f"Solution: {solution}\n"
            f"Full text: {text[:400]}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a Project Intelligence Assistant. "
        "Answer the user's question using ONLY the project data provided. "
        "For every project you mention, always cite it as (Slide N). "
        "Be concise and structured. Use bullet points. "
        "If the data does not fully answer the question, say so honestly."
    )

    user_message = (
        f"Here are the most relevant projects:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer with slide number references:"
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


# --------------------------------
# SIDEBAR
# --------------------------------
with st.sidebar:
    st.header("📊 Project Statistics")
    st.metric("Projects Indexed", "201")
    st.metric("Technologies", "80+")
    st.metric("Domains", "25+")
    st.metric("AI Search", "Enabled")

    st.markdown("---")
    st.subheader("🔎 Filters")

    selected_domain = st.selectbox(
        "Domain",
        [
            "All", "FinTech", "Healthcare", "E-Commerce", "Retail",
            "Education", "Manufacturing", "Logistics", "B2B SaaS",
            "Insurance", "Telecom", "Banking"
        ]
    )


# --------------------------------
# TITLE
# --------------------------------
st.title("🚀 Project Repository Assistant")
st.write("Search across 201 real-world projects using AI-powered semantic search.")
st.info("Try: Fraud Detection, Healthcare, Machine Learning, Recommendation Systems, Data Engineering, NLP, Computer Vision")
st.write("Enter a project domain, technology, or use case.")


# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# --------------------------------
# LOAD DATA
# --------------------------------
with open("slides.json", "r", encoding="utf-8") as f:
    slides = json.load(f)

index = faiss.read_index("slides.index")


# --------------------------------
# SEARCH INPUT
# --------------------------------
query = st.text_input(
    "Ask about projects",
    placeholder="e.g. fraud detection, healthcare analytics, recommendation engine..."
)

search_button = st.button("🔍 Search")


# --------------------------------
# SEARCH LOGIC
# --------------------------------
if search_button and query:

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32),
        5
    )

    # Collect top matching projects
    top_projects = []
    for rank, idx in enumerate(indices[0]):
        project_domain = extract_domain(slides[idx]["text"])
        if selected_domain != "All" and project_domain != selected_domain:
            continue
        top_projects.append({
            "rank": rank,
            "idx": idx,
            "slide": slides[idx]["slide"],
            "text": slides[idx]["text"],
            "score": 1 / (1 + distances[0][rank])
        })

    if not top_projects:
        st.warning("No projects matched the selected domain filter. Try 'All'.")
        st.stop()

    # ── GROQ AI ANSWER ────────────────────────────────────────
    st.subheader("🤖 AI-Generated Answer")

    with st.spinner("Generating answer using Groq..."):
        try:
            llm_answer = generate_llm_answer(query, top_projects)
            st.success(llm_answer)
            st.caption(
                f"Answer generated from top {len(top_projects)} matched projects. "
                f"Slide numbers cited inline."
            )
        except Exception as e:
            st.error(f"Could not generate AI answer: {e}")

    st.divider()

    # ── PROJECT RESULT CARDS ──────────────────────────────────
    st.subheader("🎯 Top Matching Projects")
    st.success(f"Found {len(top_projects)} relevant projects")
    st.caption(f"Search Query: {query}")

    for proj in top_projects:

        rank = proj["rank"]
        idx = proj["idx"]
        score = proj["score"]

        if rank == 0:
            st.balloons()
            st.success("🏆 BEST MATCH FOUND")

        with st.container():

            match_percent = int(score * 100)
            st.success(f"🤖 Match Confidence: {match_percent}%")
            st.progress(match_percent / 100)

            if rank == 0:
                st.markdown(f"## 🏆 Best Match - Project #{slides[idx]['slide']}")
            else:
                st.markdown(f"## 📁 Project #{slides[idx]['slide']}")

            domain = extract_domain(slides[idx]["text"])
            team_size = extract_team_size(slides[idx]["text"])
            duration = extract_duration(slides[idx]["text"])
            tech_stack = extract_tech_stack(slides[idx]["text"])
            problem, solution = extract_problem_solution(slides[idx]["text"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"🏢 Domain: {domain}")
            with col2:
                st.write(f"👥 Team Size: {team_size}")
            with col3:
                st.write(f"⏱ Duration: {duration}")

            if tech_stack:
                st.write("🛠 Technologies: " + " | ".join(tech_stack))

            st.markdown("### 🚨 Problem")
            st.write(problem)

            st.markdown("### ✅ Solution")
            st.write(solution)

            st.markdown("### 📄 Project Overview")
            st.info(slides[idx]["text"][:600])

            st.divider()


# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")
st.write("Built using FAISS + Sentence Transformers + Groq (LLaMA3) + Streamlit")