import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from utils import search, call_llm, create_pdf
from memory import Memory
from agents import planner, critic, improver, writer, summarizer
from t5_model import t5_summarize


def run_pipeline(topic):

    memory = Memory()
    progress = st.progress(0)

    # QUERY
    st.markdown("##  Query")
    st.write(topic)
    progress.progress(10)

    # PLAN (Hidden from UI)
    plan = planner(topic)
    queries = [p["query"] for p in plan]
    progress.progress(20)

    # SEARCH
    st.markdown("##  Search Results")

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(search, queries))

    flat = [i for sub in results for i in sub]

    for r in flat[:5]:
        st.markdown(
            f'<div class="card">{r}</div>',
            unsafe_allow_html=True
        )

    progress.progress(40)

    # MEMORY
    memory.add(flat)

    context = memory.search(topic)
    context_text = "\n".join(context)

    # SUMMARY
    st.markdown("##  Summary")

    summary = t5_summarize(context_text)

    st.markdown(
        f'<div class="card">{summary}</div>',
        unsafe_allow_html=True
    )

    progress.progress(55)

    # DRAFT REPORT
    st.markdown("##  Draft Report")

    draft = writer(summary)

    st.markdown(
        f'<div class="card">{draft}</div>',
        unsafe_allow_html=True
    )

    progress.progress(70)

    # CRITIC
    st.markdown("##  Critique")

    critique_text = critic(draft)

    st.markdown(
        f'<div class="card">{critique_text}</div>',
        unsafe_allow_html=True
    )

    progress.progress(80)

    # IMPROVED REPORT
    st.markdown("##  Corrected Report")

    corrected = improver(draft, critique_text)

    st.markdown(
        f'<div class="card">{corrected}</div>',
        unsafe_allow_html=True
    )

    progress.progress(90)

    # FINAL REPORT
    st.markdown("##  Final Report")

    final = corrected

    st.markdown(
        f'<div class="card">{final}</div>',
        unsafe_allow_html=True
    )

    # DOWNLOAD BUTTONS
    st.download_button(
        "⬇ Download TXT",
        final,
        "report.txt"
    )

    pdf = create_pdf(final, "report.pdf")

    with open(pdf, "rb") as f:
        st.download_button(
            "⬇ Download PDF",
            f,
            "report.pdf"
        )

    progress.progress(95)

    # FINAL SUMMARY
    st.markdown("##  Final Summary")

    final_summary = t5_summarize(final[:1000])

    st.markdown(
        f'<div class="card">{final_summary}</div>',
        unsafe_allow_html=True
    )

    # CONFIDENCE SCORE
    st.markdown("##  Confidence Score")

    confidence = call_llm(
        "Score 0-100",
        final
    )

    st.markdown(
        f'<div class="card">{confidence}</div>',
        unsafe_allow_html=True
    )

    progress.progress(100)

    return memory
