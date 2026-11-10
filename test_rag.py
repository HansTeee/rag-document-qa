"""
End-to-end test: the SAME 10 questions as test_retrieval.py, now with answers.

Run from the project root:
    python test_rag.py              (Gemini, all 10 questions)
    python test_rag.py local        (local flan-t5, all 10 questions)
    python test_rag.py compare      (Gemini vs local side by side on 3 questions)
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from generate import GeneratorError, get_generator  # noqa: E402
from rag import answer_question  # noqa: E402
from test_retrieval import TEST_QUESTIONS  # noqa: E402  - reuse, so the two tests can't drift apart

COMPARE_INDICES = [1, 3, 6]   # Q2 (in docs), Q4 (passing mention), Q7 (unanswerable trap)


def wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(" ".join(text.split()), width=100, initial_indent=indent, subsequent_indent=indent)


def run_all(generator_name: str) -> None:
    try:
        generator = get_generator(generator_name)
    except GeneratorError as e:
        # WHY catch here: a missing key is a setup problem, not a bug, so the
        # user should see one clear sentence, not a 40-line traceback.
        print(f"Setup error: {e}")
        sys.exit(1)

    for i, (category, question, expected) in enumerate(TEST_QUESTIONS, start=1):
        try:
            result = answer_question(question, generator)
        except GeneratorError as e:
            print(f"Q{i}: generation failed: {e}")
            continue
        print("=" * 100)
        print(f"Q{i} [{category}] {question}")
        print(f"   expected: {expected}")
        print(f"   ANSWER ({generator.name}){'  <-- REFUSED' if result.is_refusal else ''}:")
        print(wrap(result.answer))
        print("   SOURCES GIVEN TO THE MODEL:")
        for s in result.sources:
            print(f"      - {s.source_file[:45]}  p{s.page_number}  sim={s.similarity:.3f}")


def run_compare() -> None:
    try:
        gemini = get_generator("gemini")
    except GeneratorError as e:
        print(f"Setup error: {e}")
        sys.exit(1)
    local = get_generator("local")
    for idx in COMPARE_INDICES:
        category, question, expected = TEST_QUESTIONS[idx]
        print("=" * 100)
        print(f"Q{idx + 1} [{category}] {question}")
        print(f"   expected: {expected}")
        for gen in (gemini, local):
            result = answer_question(question, gen)
            print(f"   --- {gen.name.upper()} ---")
            print(wrap(result.answer))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    if mode == "compare":
        run_compare()
    else:
        run_all(mode)
