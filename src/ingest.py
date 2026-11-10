"""
Ingestion: PDF -> pages -> chunks -> embeddings -> ChromaDB.

This is the INDEXING half of RAG. It runs once per document (not once per
question), which is why it's allowed to be slow-ish: we pay the embedding cost
up front so that answering questions later is fast.

Run it from the project root:
    python src\\ingest.py
"""

import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

# LangChain 1.x import locations (verified against the installed versions).
# Most online tutorials use the old 0.x paths like `langchain.document_loaders`,
# which no longer exist in 1.x.
#
# NOTE: importing from langchain_community prints a DeprecationWarning saying
# the package is "being sunset". PyPDFLoader still works and has no official
# standalone replacement package yet; we pin the version in requirements.txt
# so it can't silently break. If it ever goes away, the fallback is to read
# pages with `pypdf.PdfReader` directly and build Document objects ourselves.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# WHY this sys.path line: it lets `python src\ingest.py` (run as a script) and
# `from src.ingest import ...` (imported by tests/app) both find config.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1: Load PDFs
# ---------------------------------------------------------------------------
def load_pdfs(pdf_paths: list[Path]) -> list[Document]:
    """Load each PDF into one Document per page, keeping page numbers.

    WHY one Document per page: PyPDFLoader does this by default, and it means
    every chunk we create later inherits a page number. "Which page did this
    come from?" is the first thing a user asks, and the UI will cite it.
    """
    all_pages = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            # PyPDFLoader stores a 0-based "page" index (page 1 -> 0). Humans
            # count from 1, so we store a 1-based copy for citations, plus the
            # bare file name (the full "source" path is long and, on a server,
            # leaks the temp-folder layout).
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
            page.metadata["source_file"] = Path(path).name
        all_pages.extend(pages)
    return all_pages


# ---------------------------------------------------------------------------
# Step 2: Split into chunks
# ---------------------------------------------------------------------------
def split_documents(pages: list[Document]) -> list[Document]:
    """Split pages into overlapping ~1000-character chunks.

    WHY split at all: an embedding is ONE vector summarising a piece of text.
    A whole page covers several topics, so its vector is a blurry average and
    matches no specific question well. Smaller chunks = sharper vectors.

    WHY "Recursive": it tries to split on paragraph breaks first ("\\n\\n"),
    then lines, then sentences/spaces, and only cuts mid-word as a last resort.
    So chunks tend to end at natural boundaries instead of mid-sentence.

    WHY OVERLAP (chunk_overlap=200):
    Imagine a fact that straddles a chunk boundary - "...the filter used a
    passband of | 0.7 Hz to 3.5 Hz...". Without overlap, it's cut in half:
    the first chunk has the question's context but not the number, the second
    has the number but not the context, so it's retrievable from NEITHER.
    With overlap, the last ~200 characters of each chunk are repeated at the
    start of the next one, so content near a boundary appears whole in at
    least one neighbour. 200/1000 (20%) is a common starting ratio, not a law:
    more overlap = fewer split facts but more duplicate storage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # add_start_index records where in the page each chunk began - handy
        # for debugging "why was this chunk cut here?"
        add_start_index=True,
    )
    # split_documents copies each page's metadata (file, page_number) onto
    # every chunk made from it - that's how citations survive splitting.
    return splitter.split_documents(pages)


# ---------------------------------------------------------------------------
# Step 3: Embedding model
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformers model (once per process).

    WHY lru_cache: loading the model takes a few seconds. Both ingestion and
    retrieval need it, and without caching each call would reload it.

    WHY normalize_embeddings=True: it scales every vector to length 1. Then
    cosine similarity is just a dot product, and scores are comparable across
    chunks regardless of how long each chunk's text was.

    WHY device="cpu": free Hugging Face Spaces have no GPU, so we develop on
    the same hardware we deploy on - no surprises later.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Step 4: Store in ChromaDB
# ---------------------------------------------------------------------------
def build_vectorstore(
    chunks: list[Document],
    persist_directory: Path | None = None,
    collection_name: str = config.COLLECTION_NAME,
    progress_callback=None,
    batch_size: int = 16,
) -> Chroma:
    """Embed the chunks and store them in Chroma.

    persist_directory=None -> in-memory store (gone when the process exits).
    persist_directory=path -> written to disk so retrieval can reopen it later.

    collection_name: in-memory stores in ONE Python process share a single
    Chroma backend, so two stores with the same collection name see each
    other's data (verified in Phase 3). The web app gives every visitor a
    unique name so nobody can query someone else's uploads.

    progress_callback(done, total): called after each batch, so the UI can
    show a progress bar instead of looking frozen.
    """
    if persist_directory is not None and Path(persist_directory).exists():
        # WHY delete first: Chroma APPENDS. Re-running ingestion without this
        # would store every chunk twice, and the same text would come back
        # as both result #1 and result #2 for every query.
        shutil.rmtree(persist_directory)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory) if persist_directory else None,
        # WHY cosine: Chroma's default metric is squared L2 distance, whose
        # numbers are hard to read. Cosine distance is 0 for identical meaning
        # and grows as meanings diverge, so "similarity = 1 - distance" is an
        # intuitive 0..1-ish number.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )

    # WHY batches instead of one big call: the result is identical, but
    # batching lets us report progress between batches.
    for start in range(0, len(chunks), batch_size):
        store.add_documents(chunks[start:start + batch_size])
        if progress_callback:
            progress_callback(min(start + batch_size, len(chunks)), len(chunks))
    return store


def folder_size_mb(path: Path) -> float:
    """Total size of every file under `path`, in megabytes."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main() -> None:
    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.DATA_DIR}. Put some there and re-run.")
        sys.exit(1)

    pages = load_pdfs(pdf_paths)
    chunks = split_documents(pages)

    # Load the model BEFORE starting the timer, so "embedding time" measures
    # embedding the chunks, not downloading/loading the model.
    get_embeddings()
    start = time.perf_counter()
    build_vectorstore(chunks, persist_directory=config.CHROMA_DIR)
    embed_seconds = time.perf_counter() - start

    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
    print("\n=== Ingestion summary ===")
    print(f"PDFs:                  {len(pdf_paths)}")
    for p in pdf_paths:
        n = sum(1 for pg in pages if pg.metadata["source_file"] == p.name)
        print(f"  - {p.name} ({n} pages)")
    print(f"Total pages:           {len(pages)}")
    print(f"Total chunks:          {len(chunks)}")
    print(f"Average chunk length:  {avg_len:.0f} characters")
    print(f"Embed + store time:    {embed_seconds:.2f} s")
    print(f"ChromaDB size on disk: {folder_size_mb(config.CHROMA_DIR):.2f} MB  ({config.CHROMA_DIR})")


if __name__ == "__main__":
    main()
