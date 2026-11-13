import json
from pathlib import Path

import streamlit as st

from ingest import DB_DIR, main as ingest_main
from pipeline import Navigator

st.set_page_config(page_title="Scheme Navigator", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Government Scheme Eligibility Navigator")
st.caption("Describe your situation in plain words. I search official scheme documents and tell "
           "you what you may qualify for, with documents and steps.")

# Build the index on first run.
if not Path(DB_DIR).exists():
    with st.spinner("Indexing scheme documents (first run only)..."):
        ingest_main()


@st.cache_resource
def get_navigator():
    return Navigator()


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.profile = {}

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Answer language", ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Marathi"])
    ask_first = st.checkbox("Ask for missing details first", value=True)
    st.header("Your profile")
    st.json(st.session_state.profile or {"info": "nothing collected yet"})
    if st.button("Reset conversation"):
        st.session_state.messages, st.session_state.profile = [], {}
        st.rerun()
    st.caption("Informational only. Confirm on the official portal before applying.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"[{s['n']}] **{s['scheme']}** - {s['section']} ({s['source']})")
        if m.get("trace"):
            with st.expander("Agent trace"):
                st.code("\n".join(m["trace"]))

if prompt := st.chat_input("e.g. I'm a 45-year-old farmer in Karnataka with 2 acres, income 1.5 lakh"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching scheme documents..."):
            try:
                out = get_navigator().handle(prompt, st.session_state.profile, language, ask_first)
            except Exception as e:  # surface config errors (missing API key etc.) clearly
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.session_state.profile = out["profile"]
        st.markdown(out["answer"])
        if out["sources"]:
            with st.expander("Sources"):
                for s in out["sources"]:
                    st.markdown(f"[{s['n']}] **{s['scheme']}** - {s['section']} ({s['source']})")
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"], "trace": out["trace"]}
    )
    st.rerun()  # refresh the sidebar profile