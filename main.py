import os
import json
import requests
import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Advanced Research System", layout="wide")

# ==========================================================
# STYLING
# ==========================================================
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
.navbar { font-size: 36px; font-weight: bold; text-align: center; padding: 20px; }
.section { font-size: 22px; margin-top: 20px; }
.card { background: white; color: black; padding: 20px; border-radius: 10px; }
button { background-color: #2563eb !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="navbar">Advanced Multi-Agent Research System</div>',
            unsafe_allow_html=True)

# ==========================================================
# ENV
# ==========================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENROUTER_API_KEY or not SERPAPI_API_KEY:
    st.error("❌ API keys missing. Add them in .env file")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL = "meta-llama/llama-3-8b-instruct"

# Load embedding model once (cached)


@st.cache_resource
def load_embedding():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding()

# ==========================================================
# UTIL FUNCTIONS
# ==========================================================


def call_llm(system, user, temp=0.5):
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temp
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"


@st.cache_data(show_spinner=False)
def serpapi_search(query):
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 5
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        return [r.get("snippet", "") for r in data.get("organic_results", [])]

    except:
        return []


# ==========================================================
# MEMORY (RAG)
# ==========================================================

class Memory:
    def __init__(self):
        self.docs = []
        self.vecs = []
        self.index = None

    def add(self, texts):
        for text in texts:
            chunks = text.split('. ')
            for chunk in chunks:
                if len(chunk) > 30:
                    vec = embedding_model.encode(chunk)
                    vec = vec / np.linalg.norm(vec)

                    self.docs.append(chunk)
                    self.vecs.append(vec)

        if self.vecs:
            dim = len(self.vecs[0])
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(np.array(self.vecs))

    def search(self, query, k=5):
        if not self.index:
            return []

        qv = embedding_model.encode(query)
        qv = qv / np.linalg.norm(qv)

        _, I = self.index.search(np.array([qv]), k)
        return [self.docs[i] for i in I[0]]


# ==========================================================
# AGENTS
# ==========================================================

class Planner:
    def run(self, topic):
        prompt = f"Generate 3 concise search queries for: {topic}"
        res = call_llm("You are a research planner.", prompt)
        return [q.strip("- ") for q in res.split("\n") if q.strip()]


class Critic:
    def run(self, report):
        return call_llm(
            "You are a strict academic reviewer.",
            f"Find logical issues and missing points:\n{report}",
            0.3
        )


class Improver:
    def run(self, report, critique):
        return call_llm(
            "You are an expert academic writer.",
            f"Improve the report using critique:\nReport:\n{report}\n\nCritique:\n{critique}"
        )


# ==========================================================
# PIPELINE
# ==========================================================

def run_pipeline(topic):
    planner = Planner()
    critic = Critic()
    improver = Improver()
    memory = Memory()

    queries = planner.run(topic)

    # Parallel search
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(serpapi_search, queries))

    summaries = []
    for r in results:
        if r:
            summary = call_llm("Summarize key insights", str(r))
            summaries.append(summary)

    memory.add(summaries)
    context = memory.search(topic)

    draft = call_llm(
        "Write structured research report with headings", "\n".join(context))
    critique = critic.run(draft)
    final = improver.run(draft, critique)

    return final, memory


# ==========================================================
# UI
# ==========================================================

page = st.radio(
    "Navigation",  # FIXED label issue
    ["📊 Research", "💬 Assistant"],
    horizontal=True,
    label_visibility="collapsed"
)

# -------------------------
# RESEARCH PAGE
# -------------------------
if page == "📊 Research":

    st.markdown("## 🔍 Enter Research Topic")
    topic = st.text_input(
        "Topic", placeholder="e.g., Artificial Intelligence in Healthcare")

    if st.button("🚀 Generate Report"):
        if len(topic) < 3:
            st.warning("Please enter a valid topic")
        else:
            with st.spinner("Generating research report..."):
                final, mem = run_pipeline(topic)

                st.session_state.mem = mem
                st.session_state.report = final

            st.success("✅ Report Generated")

            st.markdown("### 📄 Final Report")
            st.markdown(
                f'<div class="card">{final}</div>', unsafe_allow_html=True)

            st.download_button("⬇ Download Report", final,
                               file_name="report.txt")


# -------------------------
# ASSISTANT PAGE (RAG CHAT)
# -------------------------
elif page == "💬 Assistant":

    if "mem" not in st.session_state:
        st.info("Generate report first")
    else:
        st.markdown("## 🤖 Ask Questions")

        q = st.text_input("Ask something about the report")

        if st.button("Ask"):
            context = st.session_state.mem.search(q)

            if context:
                ans = call_llm(
                    "Answer only from given context.",
                    "\n".join(context) + f"\nQuestion: {q}"
                )
            else:
                ans = "No relevant information found."

            st.markdown(
                f'<div class="card">{ans}</div>', unsafe_allow_html=True)
