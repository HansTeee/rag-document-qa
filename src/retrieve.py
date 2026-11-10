"""
Retrieval: question -> the k most similar chunks, with scores and sources.

This is the first half of the QUERYING side of RAG (the second half -
generating an answer - comes in Phase 2).
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ingest import get_embeddings  # noqa: E402  (same model as ingestion!)


@dataclass
class RetrievedChunk:
    """One search result. A small class instead of a raw tuple so calling
    code reads `chunk.similarity` instead of `result[1]`."""
    text: str
    source_file: str
    page_number: int
    distance: float     # cosine distance from Chroma: 0 = same meaning, bigger = less similar
    similarity: float   # 1 - distance: higher = more similar (easier to read)


@lru_cache(maxsize=1)
def load_vectorstore(persist_directory: str = str(config.CHROMA_DIR)) -> Chroma:
    """Open the persisted Chroma store created by ingest.py.

    WHY the same embedding function as ingestion: the query must be turned
    into a vector in the SAME "coordinate system" as the stored chunks.
    Embedding the question with a different model would be like measuring
    distances between a map in miles and a map in kilometres.
    """
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"No vector store at {persist_directory}. Run `python src\\ingest.py` first."
        )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def retrieve(query: str, k: int = config.DEFAULT_K, vectorstore: Chroma | None = None) -> list[RetrievedChunk]:
    """Return the top-k chunks for `query`, most similar first.

    WHY similarity_search_with_score instead of similarity_search: the plain
    version returns chunks with no indication of HOW similar they are. The
    score lets us see whether the best match is a strong hit or merely the
    least-bad option - which matters for questions the documents can't answer.

    `vectorstore` is optional so the Streamlit app (Phase 3) can pass its own
    store (e.g. one built from uploaded files) instead of the on-disk one.
    """
    store = vectorstore if vectorstore is not None else load_vectorstore()
    results = store.similarity_search_with_score(query, k=k)

    return [
        RetrievedChunk(
            text=doc.page_content,
            source_file=doc.metadata.get("source_file", "unknown"),
            page_number=doc.metadata.get("page_number", -1),
            distance=distance,
            similarity=1 - distance,
        )
        for doc, distance in results
    ]
