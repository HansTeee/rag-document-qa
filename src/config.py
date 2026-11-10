"""
Central configuration for the RAG pipeline.

WHY a config file: every "magic number" (chunk size, model name, paths) lives
in one place. When you later experiment with chunk_size, you change it here
and ingest.py / retrieve.py / the app all pick it up - no hunting through code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# WHY resolve paths from this file's location instead of using "data/":
# relative paths depend on which folder you ran `python` from. Anchoring to
# the project root means `python src/ingest.py` and `python test_retrieval.py`
# both find the same folders no matter where you launch them.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"            # your PDFs (gitignored)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"     # the persisted vector store (gitignored)

# A "collection" in Chroma is like a table in a database. Naming it lets
# several indexes live side by side in one chroma_db folder if needed later.
COLLECTION_NAME = "documents"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Measured in CHARACTERS (not tokens) because that's what
# RecursiveCharacterTextSplitter counts. ~1000 chars is roughly 150-250 words:
# big enough to hold a complete thought, small enough that one chunk is
# about one topic (so its embedding isn't a blurry average of many topics).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200   # see the long comment in ingest.py for why overlap exists

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2: 384-dimensional vectors, ~90MB download, fast on CPU,
# good enough for English prose. Bigger models retrieve somewhat better but
# are much slower per query on a free (CPU-only) Hugging Face Space.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# How many chunks to fetch per question by default. Too few and the answer
# may be in chunk #5 that you never looked at; too many and you flood the
# LLM (later phases) with irrelevant text.
DEFAULT_K = 4

# Per-generator k (Phase 2 finding): with k=4, answers to two test questions
# sat at retrieval ranks 6 and 7 and were never shown to the LLM. Gemini reads
# thousands of tokens cheaply, so it gets 8 chunks. flan-t5 can only read 512
# tokens (~2 chunks), so giving it 8 would just be truncated away.
GEMINI_K = 8
LOCAL_K = 4

# ---------------------------------------------------------------------------
# Bundled sample document (Phase 4)
# ---------------------------------------------------------------------------
# Committed to the repo so a visitor can try the demo without finding a PDF.
SAMPLE_PDF = PROJECT_ROOT / "sample_docs" / "rag_demo_guide.pdf"

# ---------------------------------------------------------------------------
# Generation (Phase 2)
# ---------------------------------------------------------------------------
# Gemini: verified against the free tier in Nov 2026. "gemini-2.5-flash"
# now returns 404 for new API keys, which is a reminder that hosted model
# names expire - that's why it's a config value, not hard-coded.
#
# WHY flash-LITE: testing showed the free tier allows only 20 requests/day
# for gemini-3.5-flash (from the API's own 429 error). On a public demo that
# daily budget is shared by EVERY visitor. Flash-lite has its own, separate
# quota. Its exact limit wasn't published on Google's public docs page, so
# check https://aistudio.google.com/rate-limit while signed in.
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# Local fallback: runs on CPU with no API key. ~1GB download on first use.
LOCAL_MODEL_NAME = "google/flan-t5-base"
# flan-t5 was trained on inputs of at most 512 tokens. Longer inputs get cut.
LOCAL_MAX_INPUT_TOKENS = 512
MAX_ANSWER_TOKENS = 2048   # includes Gemini's hidden "thinking" tokens - see generate.py

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Retrieval itself is deterministic, but later phases use a local LLM that
# can sample randomly. One shared seed makes runs repeatable so that when a
# result changes, you know it's because YOU changed something.
SEED = 42
