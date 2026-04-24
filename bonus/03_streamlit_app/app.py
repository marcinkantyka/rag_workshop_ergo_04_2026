"""
DKV RAG Assistant — Streamlit frontend.

Run from the project root:
    streamlit run bonus/03_streamlit_app/app.py

Requires: NETLIGHT_API_KEY and NETLIGHT_BASE_URL in .env
          ChromaDB populated (run notebooks 01–03 first)

Exercises:
  - render_sources()  — display retrieved chunks with source labels
  - load_pipeline()   — wire ChromaDB + LLM client + RAGPipeline together
  - chat input handler — call the pipeline and show the answer + sources
"""

import os
import sys
import time
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

try:
    from src.pipeline import RAGPipeline, get_llm_client, FAST_MODEL, SMART_MODEL
except (ImportError, NotImplementedError):
    from solutions.src.pipeline import RAGPipeline, get_llm_client, FAST_MODEL, SMART_MODEL

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DKV RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Exercise A — render_sources ───────────────────────────────────────────────
# This function makes retrieved context visible to the user — a key RAG transparency feature.

def render_sources(contexts: list, sources: list, latency: float | None = None) -> None:
    """Show retrieved chunks in a collapsible expander with source labels."""
    # TODO:
    # 1. Build a label string: "N source(s) retrieved" + " · Xs" if latency is given.
    # 2. Open a st.expander(label).
    # 3. Inside, iterate over zip(contexts, sources) with enumerate starting at 1.
    #    For each chunk: show the source as bold markdown "[i] source_name",
    #    then show up to 350 chars of text with st.text().
    #    Add a st.divider() between entries (not after the last one).
    raise NotImplementedError


# ── Exercise B — load_pipeline ────────────────────────────────────────────────
# @st.cache_resource means this only runs once per (model, k, min_sim) combination.

@st.cache_resource
def load_pipeline(model: str, k: int, min_sim: float) -> RAGPipeline:
    """Connect to ChromaDB and return a configured RAGPipeline."""
    # TODO:
    # 1. Create a chromadb.PersistentClient pointing at ROOT / "chroma_db".
    # 2. Get or create the "workshop_rag" collection with metadata={"hnsw:space": "cosine"}.
    # 3. Call get_llm_client() to get the OpenAI-compatible client.
    # 4. Return a RAGPipeline with the given model, k, and min_sim.
    raise NotImplementedError


# ── Sidebar — pipeline settings ───────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Pipeline settings")

    model_choice = st.selectbox(
        "LLM model",
        options=["haiku (fast)", "sonnet (smart)"],
        index=0,
    )
    selected_model = FAST_MODEL if "haiku" in model_choice else SMART_MODEL

    top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=10, value=5)
    min_similarity = st.slider("Min similarity threshold", min_value=0.0, max_value=0.9,
                                value=0.0, step=0.05)
    show_chunks = st.toggle("Show retrieved chunks", value=True)

    st.divider()
    st.caption(f"Model: `{selected_model}`")
    st.caption(f"top_k={top_k}, min_sim={min_similarity:.2f}")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Pipeline ──────────────────────────────────────────────────────────────────

pipeline = load_pipeline(selected_model, top_k, min_similarity)

# ── Chat UI ───────────────────────────────────────────────────────────────────

st.title("🏥 DKV RAG Assistant")
st.caption("Ask questions about DKV Belgium insurance products. Powered by RAG + Claude.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_chunks and msg.get("contexts"):
            render_sources(msg["contexts"], msg["sources"], msg.get("latency"))

# ── Exercise C — chat input handler ──────────────────────────────────────────
# This is where a user question becomes a RAG result.

if question := st.chat_input("Ask a question about DKV Belgium insurance..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        # TODO:
        # 1. Inside a st.spinner("Retrieving context and generating answer..."):
        #    a. Record t0 = time.perf_counter()
        #    b. Call result = pipeline.ask(question)
        #    c. Compute latency = time.perf_counter() - t0
        #
        # 2. Display result["answer"] with st.write().
        #
        # 3. If show_chunks and result has "contexts", call render_sources(...).
        #
        # 4. Add thumbs-up and thumbs-down buttons side by side using st.columns.
        #    Use a unique key per message (e.g. f"up_{len(st.session_state.messages)}").
        pass

    # TODO: Append the assistant message dict to st.session_state.messages.
    # Include: role, content (answer), contexts, sources, latency.
