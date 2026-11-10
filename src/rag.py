"""
The full RAG loop: question -> retrieve top-k -> build prompt -> generate.

Returns the answer AND the chunks it was generated from. Always - even when
the answer looks confident. Showing the evidence is what makes RAG auditable,
and it's the main thing that distinguishes it from just asking a chatbot.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from generate import NO_ANSWER, BaseGenerator  # noqa: E402
from retrieve import RetrievedChunk, retrieve  # noqa: E402


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]   # exactly the chunks the LLM saw

    @property
    def is_refusal(self) -> bool:
        """True if the model said it couldn't answer from the documents."""
        return NO_ANSWER.lower().rstrip(".") in self.answer.lower()


def answer_question(
    question: str,
    generator: BaseGenerator,
    k: int | None = None,
    vectorstore: Chroma | None = None,
) -> RAGResult:
    question = question.strip()
    if not question:
        # WHY check here and not only in the UI: any caller (tests, a future
        # API) gets the same protection against embedding an empty string.
        raise ValueError("Question is empty.")

    # WHY k defaults to the generator's own value: Gemini can read 8 chunks,
    # flan-t5 only ~2 before truncation (see config.py).
    if k is None:
        k = generator.default_k

    # 1. Retrieve: find the k chunks whose meaning is closest to the question.
    chunks = retrieve(question, k=k, vectorstore=vectorstore)

    # 2 + 3. Build the prompt and generate. The generator owns its own prompt
    # format (Gemini and flan-t5 need different ones), so rag.py stays generic.
    answer = generator.generate(question, chunks)

    # 4. Return the answer together with its evidence.
    return RAGResult(question=question, answer=answer, sources=chunks)
