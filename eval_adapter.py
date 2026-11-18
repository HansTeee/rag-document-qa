"""
eval_adapter.py - lets evaluate_rag.py call the RAG pipeline as ask(question).

Setup happens ONCE at import time:
  - the persisted Chroma store built by src/ingest.py from data/ (chroma_db/)
  - the Gemini generator (same class, prompt, model and k the app uses)

Deliberately NO fallback to the local flan-t5 model: if Gemini fails, the call
raises and evaluate_rag.py records it as an ERROR row, so the numbers never
silently mix two different models.

Usage:
    .venv\\Scripts\\python.exe evaluate_rag.py --func eval_adapter:ask --questions eval_questions.json
"""

import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from generate import get_generator  # noqa: E402
from rag import answer_question  # noqa: E402
from retrieve import load_vectorstore  # noqa: E402

_STORE = load_vectorstore()
_GENERATOR = get_generator("gemini")


def ask(question: str) -> dict:
    result = answer_question(question, _GENERATOR, vectorstore=_STORE)
    sources = [
        Document(
            page_content=c.text,
            # evaluate_rag.py follows PyPDFLoader's convention (0-based "page");
            # RetrievedChunk.page_number is 1-based, so convert back.
            metadata={
                "page": c.page_number - 1,
                "page_number_1based": c.page_number,
                "source": c.source_file,
                "similarity": round(c.similarity, 4),
            },
        )
        for c in result.sources
    ]
    return {"answer": result.answer, "sources": sources}
