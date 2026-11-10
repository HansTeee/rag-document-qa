"""
Streamlit chat interface for the RAG pipeline.

Run locally:   streamlit run app.py

All the RAG logic lives in src/ - this file only does UI: caching, session
state, upload handling and rendering. If you find yourself writing retrieval
or prompt code here, it belongs in src/ instead.
"""

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config  # noqa: E402
from generate import GeneratorError, get_generator  # noqa: E402
from ingest import build_vectorstore, get_embeddings, load_pdfs, split_documents  # noqa: E402
from rag import answer_question  # noqa: E402

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="centered")


# ===========================================================================
# Caching - get this right or the app is unusable
# ===========================================================================
# Streamlit re-runs this ENTIRE script from top to bottom on every
# interaction (every message, every button click). Without caching, the
# embedding model (~90MB) would be reloaded from disk on every message.
#
# @st.cache_resource keeps ONE copy of an object for the whole server
# process, shared by every visitor. That's right for things that are
# expensive to create and identical for everyone (models, API clients, the
# read-only sample index). It is WRONG for per-visitor data - see the
# "Session state" section for why uploads are NOT cached this way.

@st.cache_resource(show_spinner="Loading the embedding model (first visit only)...")
def cached_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Loading the language model (first use only)...")
def cached_generator(name: str):
    # If this raises (e.g. GeneratorError for a missing key), Streamlit does
    # NOT cache the failure, so it will be retried on the next run.
    return get_generator(name)


@st.cache_resource(show_spinner="Indexing the sample document...")
def cached_sample_store():
    """The bundled sample PDF, indexed in memory once per server process.

    WHY in memory and not a committed chroma_db/: Hugging Face Spaces have an
    ephemeral filesystem, and indexing this 2-page PDF takes under a second.
    Re-indexing at startup avoids committing a binary database that's tied to
    one exact chromadb version.
    """
    cached_embeddings()
    chunks = split_documents(load_pdfs([config.SAMPLE_PDF]))
    store = build_vectorstore(chunks, collection_name="sample_document")
    return store, len(chunks)


# ===========================================================================
# Session state - per-visitor memory
# ===========================================================================
# Ordinary Python variables are re-created on every rerun, so a plain
# `messages = []` would reset the conversation every time you sent a message.
# st.session_state is a dictionary that survives reruns, and it is separate
# for every browser tab - so it is also where each visitor's uploads live.
# (Putting uploads in cache_resource would let visitor B query visitor A's
# documents, because cache_resource is shared by everyone.)

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role", "content", "sources", "notice"}
if "user_store" not in st.session_state:
    st.session_state.user_store = None      # Chroma store built from this visitor's uploads
    st.session_state.user_files = []
    st.session_state.user_chunks = 0
if "session_id" not in st.session_state:
    # In-memory Chroma stores in one process share a backend, so each visitor
    # needs a unique collection name or their uploads would mix.
    st.session_state.session_id = uuid.uuid4().hex[:12]


def process_uploads(uploaded_files) -> None:
    """Uploaded files -> temp folder -> chunks -> this visitor's own store."""
    # Uploaded files arrive as in-memory bytes, but PyPDFLoader wants a file
    # path. TemporaryDirectory gives us a folder that is deleted automatically
    # when the `with` block ends - even if an error happens - so uploads are
    # never left lying on the server's disk.
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for f in uploaded_files:
            # Path(...).name strips any folder parts from the uploaded name, so
            # a malicious name like "..\\..\\x.pdf" can't escape the temp dir.
            path = Path(tmp) / Path(f.name).name
            path.write_bytes(f.getbuffer())
            paths.append(path)

        with st.status("Processing documents...", expanded=True) as status:
            st.write("Reading PDF text...")
            pages = load_pdfs(paths)
            chunks = split_documents([p for p in pages if p.page_content.strip()])
            if not chunks:
                status.update(label="No text found", state="error")
                st.error("No extractable text found. Scanned (image-only) PDFs need OCR, which this app doesn't do.")
                return

            st.write(f"{len(pages)} pages -> {len(chunks)} chunks. Embedding...")
            bar = st.progress(0.0)
            # Drop the previous upload's collection so memory doesn't pile up.
            if st.session_state.user_store is not None:
                st.session_state.user_store.delete_collection()
            store = build_vectorstore(
                chunks,
                collection_name=f"user_{st.session_state.session_id}",
                progress_callback=lambda done, total: bar.progress(done / total, f"{done}/{total} chunks"),
            )
            status.update(label=f"Indexed {len(chunks)} chunks from {len(paths)} file(s)", state="complete")

    st.session_state.user_store = store
    st.session_state.user_files = [p.name for p in paths]
    st.session_state.user_chunks = len(chunks)


def use_sample_document() -> None:
    if st.session_state.user_store is not None:
        st.session_state.user_store.delete_collection()
    st.session_state.user_store = None
    st.session_state.user_files = []
    st.session_state.user_chunks = 0


# ===========================================================================
# Layout
# ===========================================================================
st.title("📚 RAG Document Q&A")

with st.expander("About this app", expanded=False):
    st.markdown(
        "- It answers **only from the documents loaded** - the bundled sample guide, or PDFs you upload.\n"
        "- When the documents don't contain the answer, it is instructed to **say it doesn't know** "
        "rather than guess.\n"
        "- It **can still make mistakes**. Every answer shows the source passages it was based on - "
        "check them.\n"
        "- Uploaded files are processed in memory and **not stored**: they are gone when you close the "
        "tab or the Space restarts.\n"
        "- With Gemini selected, your question and the retrieved passages are sent to Google. "
        "Choose *Local* to keep everything on the server."
    )

sample_store, sample_chunks = cached_sample_store()

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not uploaded, use_container_width=True):
        process_uploads(uploaded)

    if st.session_state.user_store is not None:
        st.success(f"Using your upload(s): {st.session_state.user_chunks} chunks indexed")
        for name in st.session_state.user_files:
            st.caption(f"• {name}")
        st.button("Switch back to the sample document", on_click=use_sample_document)
    else:
        st.info(f"Using the bundled sample guide: {sample_chunks} chunks indexed. "
                "Upload PDFs above to ask about your own.")

    st.header("Answer generator")
    choice = st.radio(
        "Generator",
        ["Gemini (best answers)", "Local flan-t5 (private, weaker)"],
        label_visibility="collapsed",
    )
    generator_name = "gemini" if choice.startswith("Gemini") else "local"

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

active_store = st.session_state.user_store or sample_store
active_chunks = st.session_state.user_chunks or sample_chunks


def render_sources(sources: list[dict]) -> None:
    """The most important UI element: lets the user VERIFY instead of trust."""
    with st.expander(f"Sources ({len(sources)} retrieved passages)"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**{i}. {s['file']} — page {s['page']}** · similarity {s['similarity']:.3f}")
            st.text(s["text"])


# Re-draw the conversation so far (remember: the whole script re-runs).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("notice"):
            st.caption(msg["notice"])
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


def answer_with_fallback(question: str, wanted: str):
    """Answer with the chosen generator; fall back to local if Gemini can't.

    Returns (result, notice). WHY a fallback: someone who forks the repo
    without a key - or a visitor arriving after the shared daily quota is
    used up - should still get a working demo, with an honest notice.
    """
    notice = None
    if wanted == "gemini":
        try:
            return answer_question(question, cached_generator("gemini"), vectorstore=active_store), None
        except GeneratorError as e:
            notice = f"⚠️ Gemini unavailable ({e}) Answered with the local model instead."
    return answer_question(question, cached_generator("local"), vectorstore=active_store), notice


question = st.chat_input("Ask a question about the documents")
if question is not None:
    question = question.strip()

if question:  # empty / whitespace-only input is simply ignored
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if active_chunks == 0:
            reply = {"role": "assistant", "content": "No documents are indexed yet - upload a PDF first."}
            st.markdown(reply["content"])
        else:
            with st.spinner("Searching the documents and writing an answer..."):
                result, notice = answer_with_fallback(question, generator_name)
            sources = [
                {"file": c.source_file, "page": c.page_number, "similarity": c.similarity, "text": c.text}
                for c in result.sources
            ]
            reply = {"role": "assistant", "content": result.answer, "sources": sources, "notice": notice}
            if notice:
                st.caption(notice)
            st.markdown(result.answer)
            render_sources(sources)
    st.session_state.messages.append(reply)
