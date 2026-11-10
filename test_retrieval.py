"""
Retrieval sanity check against the real PDFs in data/.

WHY this file matters more than the pipeline code: a RAG system can only
answer from chunks it retrieves. If retrieval returns the wrong chunks, no
LLM - however good - can fix it. So before adding generation, we look
directly at what comes back for questions where we KNOW the right answer.

Run from the project root (after `python src\\ingest.py`):
    python test_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from retrieve import retrieve  # noqa: E402

# Each test: (category, question, where the answer really is / why it's a trap).
# The "expected" note is for YOU when reading the output - it is not used by code.
TEST_QUESTIONS = [
    # --- Answer is clearly in the documents ---
    ("IN-DOCS", "What dataset was used to train the LUTANG model?",
     "LUTANG paper p2: DAiSEE, 9,068 clips from 112 students"),
    ("IN-DOCS", "What frequency range does the Butterworth bandpass filter use?",
     "LUTANG paper p3: 0.7 Hz to 3.5 Hz (42-210 BPM)"),
    ("IN-DOCS", "How many online video recordings were collected in the rPPG in the wild study?",
     "rPPG paper p1: 231 online video recordings"),

    # --- Mentioned once, in passing ---
    ("PASSING", "What color space was the cheek region converted to for extracting skin pixels?",
     "LUTANG paper p3: YCrCb (one sentence)"),
    ("PASSING", "What software was used to convert the webcam videos to mp4?",
     "rPPG paper p5: HandBrake (one sentence)"),
    ("PASSING", "Does rPPG accuracy drop for people with darker skin tones?",
     "rPPG paper p10: yes, cited as a limitation (one sentence)"),

    # --- Sounds related, but NOT answerable ---
    ("NOT-IN-DOCS", "What exact learning rate value was used to train the LSTM?",
     "TRAP: LUTANG p3 says the LR was tuned with Optuna, but never gives the value"),
    ("NOT-IN-DOCS", "How accurately can rPPG measure blood pressure?",
     "TRAP: neither paper measures blood pressure"),

    # --- Different vocabulary from the documents ---
    ("SYNONYM", "How well does the system catch students who are zoning out and not paying attention?",
     "LUTANG: 'Disengaged' minority class recall 0.34 - paper never says 'zoning out'"),
    ("SYNONYM", "Can a normal laptop camera check your pulse without touching you?",
     "Both: rPPG / 'non-contact' / 'webcam' heart rate - no word overlap with 'pulse without touching'"),
]

TOP_K = 3
PREVIEW_CHARS = 200


def main() -> None:
    # Collect the best score per category so we can compare at the end.
    best_by_category: dict[str, list[float]] = {}

    for i, (category, question, expected) in enumerate(TEST_QUESTIONS, start=1):
        results = retrieve(question, k=TOP_K)

        print("=" * 100)
        print(f"Q{i} [{category}] {question}")
        print(f"   expected: {expected}")
        for rank, r in enumerate(results, start=1):
            # Collapse newlines so each preview stays on a readable block.
            preview = " ".join(r.text.split())[:PREVIEW_CHARS]
            short_name = r.source_file[:30]
            print(f"   #{rank} similarity={r.similarity:.3f} (distance={r.distance:.3f})  "
                  f"[{short_name} p{r.page_number}]")
            print(f"       {preview}")

        best_by_category.setdefault(category, []).append(results[0].similarity)

    # Summary: is the TOP score for unanswerable questions clearly lower?
    # If it is, a threshold could flag "not in documents". If not, it can't.
    print("\n" + "=" * 100)
    print("TOP-1 SIMILARITY BY CATEGORY")
    for category, scores in best_by_category.items():
        formatted = ", ".join(f"{s:.3f}" for s in scores)
        print(f"   {category:12s} {formatted}")


if __name__ == "__main__":
    main()
