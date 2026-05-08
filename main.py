import streamlit as st
from pipeline import run_pipeline

st.set_page_config(page_title="Multi-Agent Research System", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
.card { background: white; color: black; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title(" AI-Powered Research Automation System")

page = st.radio("", [" Research", " Assistant"], horizontal=True)

if page == " Research":
    topic = st.text_input("Enter Topic")

    if st.button(" Generate"):
        if len(topic) < 3:
            st.warning("Enter valid topic")
        else:
            mem = run_pipeline(topic)
            st.session_state.mem = mem

elif page == " Assistant":
    if "mem" not in st.session_state:
        st.warning("Generate report first")
    else:
        q = st.text_input("Ask question")

        if st.button("Ask"):
            ctx = st.session_state.mem.search(q)
            from utils import call_llm

            ans = call_llm("Answer", "\n".join(ctx) + "\nQ:" + q)

            st.markdown(
                f'<div class="card">{ans}</div>', unsafe_allow_html=True)
