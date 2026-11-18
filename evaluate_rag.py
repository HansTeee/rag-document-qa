r"""
evaluate_rag.py — get REAL resume metrics for the RAG Document Q&A Chatbot
===========================================================================
What it measures (on YOUR documents and YOUR questions):
  1. Refusal rate on UNANSWERABLE questions  -> "refused X/Y questions not in the docs"
     (the most important number: does it say "I don't know" instead of inventing?)
  2. False refusals on ANSWERABLE questions   -> it said "I don't know" but the answer WAS there
  3. Answer accuracy on answerable questions  -> the answer contains your expected keywords
  4. Retrieval hit rate                       -> the retrieved chunks contain the right page / keywords
  5. Average + slowest response time

How it connects to your project
  It imports YOUR question-answering function from your project, e.g. src\rag.py.
  Find its name in CMD with:
        findstr /n "def " src\rag.py
  then pass it as   --func src.rag:<function_name>
  The function must take a question string. It may return any of:
        "answer text"
        ("answer text", [sources...])
        {"answer": "...", "sources": [...]}      (also accepts "result", "source_documents", "context")
  Sources may be LangChain Document objects (page_content + metadata) or dicts.
  If your function needs a vector store / chain object first, see the ADAPTER section below.

Usage (Windows CMD, venv activated, from the rag-document-qa folder):
    python evaluate_rag.py --func src.rag:answer_question --questions eval_questions.json
    python evaluate_rag.py --func src.rag:answer_question --questions eval_questions.json --repeats 3

Gemini free tier has a per-minute request limit, so the script waits --sleep seconds
between calls (default 6). Lower it only if you have a higher quota.
"""

import argparse
import csv
import importlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

# Phrases that count as "the bot refused / said it doesn't know".
# Add your own prompt's exact wording here if it is different.
REFUSAL_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bi do not know\b",
    r"don'?t have enough information",
    r"do not have enough information",
    r"not (?:enough|sufficient) information",
    r"(?:is|are) not (?:mentioned|covered|found|provided|present|included|stated|discussed)",
    r"not (?:in|within) the (?:provided |given )?(?:documents?|context|sources?|text)",
    r"(?:documents?|context|sources?) (?:do(?:es)? not|don'?t) (?:contain|mention|say|provide|cover|include)",
    r"(?:cannot|can'?t|unable to) (?:answer|find|determine)",
    r"no (?:relevant )?information (?:about|on|regarding)",
    r"there is no (?:mention|information)",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


# ------------------------------------------------------------------
# ADAPTER — turns whatever your function returns into (answer, sources)
# ------------------------------------------------------------------
def load_ask_function(func_spec):
    """'src.rag:answer_question' -> the function object."""
    if ":" not in func_spec:
        sys.exit("[error] --func must look like module.path:function_name, e.g. src.rag:answer_question")
    module_name, attr = func_spec.split(":", 1)
    sys.path.insert(0, os.getcwd())  # so 'src' is importable when run from the project folder
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        sys.exit(f"[error] Could not import {module_name}: {e}\n"
                 f"        Run this from the project root (the folder that contains src\\).")
    if not hasattr(module, attr):
        funcs = [n for n in dir(module) if callable(getattr(module, n)) and not n.startswith("_")]
        sys.exit(f"[error] {module_name} has no '{attr}'. Callables found: {funcs}")
    obj = getattr(module, attr)
    # If it's a class (e.g. RAGPipeline), build it and look for a typical method
    if isinstance(obj, type):
        inst = obj()
        for m in ("ask", "answer", "query", "invoke", "run", "__call__"):
            if hasattr(inst, m):
                print(f"[info] Using {attr}().{m}(question)")
                return getattr(inst, m)
        sys.exit(f"[error] Class {attr} has no ask/answer/query/invoke/run method.")
    return obj


def _doc_text_and_meta(src):
    if hasattr(src, "page_content"):  # LangChain Document
        return src.page_content or "", dict(getattr(src, "metadata", {}) or {})
    if isinstance(src, dict):
        text = src.get("page_content") or src.get("text") or src.get("content") or src.get("chunk") or ""
        meta = dict(src.get("metadata", {}) or {})
        for k in ("page", "source", "file", "page_number"):
            if k in src and k not in meta:
                meta[k] = src[k]
        return str(text), meta
    if isinstance(src, (tuple, list)) and src:  # (Document, score)
        return _doc_text_and_meta(src[0])
    return str(src), {}


def normalize_result(result):
    """Return (answer_str, [ {text, page, source} ... ])."""
    answer, sources = result, []
    if isinstance(result, dict):
        answer = (result.get("answer") or result.get("result") or result.get("output")
                  or result.get("response") or "")
        sources = (result.get("sources") or result.get("source_documents")
                   or result.get("context") or result.get("docs") or [])
    elif isinstance(result, (tuple, list)):
        answer = result[0] if result else ""
        sources = result[1] if len(result) > 1 else []
    if hasattr(answer, "content"):  # LangChain AIMessage
        answer = answer.content
    norm = []
    for s in sources or []:
        text, meta = _doc_text_and_meta(s)
        page = meta.get("page", meta.get("page_number"))
        norm.append({"text": text, "page": page,
                     "source": re.split(r"[\\/]", str(meta.get("source", meta.get("file", ""))))[-1]})
    return str(answer), norm


# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------
def is_refusal(answer):
    return bool(REFUSAL_RE.search(answer or ""))


def keywords_found(text, keywords):
    text = (text or "").lower()
    return [k for k in keywords if k.lower() in text]


def page_matches(retrieved_page, expected_pages):
    """PyPDFLoader pages are 0-indexed; people usually count from 1. Accept either."""
    if retrieved_page is None:
        return False
    try:
        p = int(retrieved_page)
    except (TypeError, ValueError):
        return False
    return any(p == e or p + 1 == e for e in expected_pages)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the RAG chatbot.")
    ap.add_argument("--func", required=True, help="module:function, e.g. src.rag:answer_question")
    ap.add_argument("--questions", default="eval_questions.json")
    ap.add_argument("--sleep", type=float, default=6.0, help="seconds between calls (API quota)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="ask every UNANSWERABLE question this many times (LLMs are not deterministic)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        sys.exit(f"[error] {qpath} not found. Copy eval_questions_TEMPLATE.json and fill it in.")
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    for q in questions:
        if q.get("question", "").startswith("REPLACE"):
            sys.exit("[error] eval_questions.json still has template questions. Replace them with "
                     "questions about YOUR PDFs first.")

    try:  # load GEMINI/GOOGLE API key from .env, same as your app does
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ask = load_ask_function(args.func)
    print(f"[info] Loaded {args.func}. {len(questions)} questions, sleep {args.sleep}s between calls.\n")

    rows = []
    total_calls = sum(args.repeats if not q.get("answerable", True) else 1 for q in questions)
    call_no = 0
    for qi, q in enumerate(questions, 1):
        answerable = bool(q.get("answerable", True))
        n_runs = 1 if answerable else args.repeats
        for run in range(1, n_runs + 1):
            call_no += 1
            t0 = time.perf_counter()
            try:
                answer, sources = normalize_result(ask(q["question"]))
                error = ""
            except Exception as e:
                answer, sources, error = "", [], f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0

            refused = is_refusal(answer) if not error else False
            kws = q.get("expected_keywords", [])
            found = keywords_found(answer, kws)
            answer_ok = bool(kws) and len(found) == len(kws) and not refused

            retrieved_text = " ".join(s["text"] for s in sources)
            exp_pages = q.get("expected_pages", [])
            exp_file = q.get("expected_file", "")
            hit = None
            if answerable and (exp_pages or kws):
                hit = False
                for s in sources:
                    file_ok = (not exp_file) or exp_file.lower() in s["source"].lower()
                    if file_ok and exp_pages and page_matches(s["page"], exp_pages):
                        hit = True
                if not hit and kws and keywords_found(retrieved_text, kws):
                    hit = True

            status = ("ERROR" if error else
                      ("refused" if refused else "answered"))
            verdict = ""
            if not error:
                if answerable:
                    verdict = "CORRECT" if answer_ok else ("FALSE REFUSAL" if refused else "CHECK")
                else:
                    verdict = "GOOD REFUSAL" if refused else "HALLUCINATED?"
            print(f"[{call_no}/{total_calls}] Q{qi}{'' if n_runs == 1 else f' run {run}'} "
                  f"({'answerable' if answerable else 'NOT in docs'}) -> {status:<8} {verdict:<14} "
                  f"{latency:5.1f}s | {q['question'][:55]}")
            if error:
                print(f"        {error}")

            rows.append({
                "q_no": qi, "run": run, "question": q["question"], "answerable": answerable,
                "category": q.get("category", ""), "answer": answer.replace("\n", " ").strip(),
                "refused": refused, "expected_keywords": "; ".join(kws),
                "keywords_found": "; ".join(found), "answer_keyword_match": answer_ok,
                "retrieval_hit": "" if hit is None else hit,
                # PyPDFLoader pages start at 0, so +1 shows the page number you see in a PDF viewer
                "sources": "; ".join(f"{s['source']} page {int(s['page']) + 1}" if str(s['page']).isdigit()
                                     else s['source'] for s in sources),
                "latency_s": round(latency, 2), "verdict": verdict, "error": error,
                "manual_correct": "",  # fill in yourself after reading the answers
            })
            if call_no < total_calls:
                time.sleep(args.sleep)

    ok_rows = [r for r in rows if not r["error"]]
    ans = [r for r in ok_rows if r["answerable"]]
    unans = [r for r in ok_rows if not r["answerable"]]
    hits = [r for r in ans if r["retrieval_hit"] != ""]
    lat = [r["latency_s"] for r in ok_rows]

    def pct(n, d):
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"

    summary = {
        "n_questions": len(questions),
        "n_calls": len(rows),
        "errors": len(rows) - len(ok_rows),
        "unanswerable_refused": sum(r["refused"] for r in unans),
        "unanswerable_total": len(unans),
        "answerable_false_refusals": sum(r["refused"] for r in ans),
        "answerable_keyword_correct": sum(r["answer_keyword_match"] for r in ans),
        "answerable_total": len(ans),
        "retrieval_hits": sum(bool(r["retrieval_hit"]) for r in hits),
        "retrieval_total": len(hits),
        "avg_latency_s": round(statistics.mean(lat), 2) if lat else None,
        "max_latency_s": round(max(lat), 2) if lat else None,
    }

    print("\n" + "=" * 66)
    print(" RAG EVALUATION RESULTS")
    print("=" * 66)
    print(f" Refused when answer NOT in docs : {pct(summary['unanswerable_refused'], summary['unanswerable_total'])}"
          f"   <- headline number")
    print(f" Correct answers (keyword match) : {pct(summary['answerable_keyword_correct'], summary['answerable_total'])}")
    print(f" False refusals (answer WAS there): {pct(summary['answerable_false_refusals'], summary['answerable_total'])}")
    print(f" Retrieval hit rate (right chunk): {pct(summary['retrieval_hits'], summary['retrieval_total'])}")
    print(f" Avg / slowest response time     : {summary['avg_latency_s']}s / {summary['max_latency_s']}s")
    if summary["errors"]:
        print(f" Calls that crashed              : {summary['errors']} (see 'error' column in the CSV)")
        print("   -> '429' / 'quota' / 'ResourceExhausted' errors mean the free tier limit was hit:"
              " rerun with --sleep 12")
    print("=" * 66)
    print(" Keyword matching is a rough check. Open the CSV, read every answer marked CHECK or\n"
          " HALLUCINATED?, and fill the manual_correct column yourself before quoting numbers.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "rag_eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = out / "rag_eval_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {csv_path}\n[saved] {json_path}")

    if summary["unanswerable_total"] and summary["answerable_total"]:
        print("\nRESUME LINE (copy, then check it reads right):")
        print(f"  ... refused {summary['unanswerable_refused']}/{summary['unanswerable_total']} out-of-scope "
              f"questions and answered {summary['answerable_keyword_correct']}/{summary['answerable_total']} "
              f"in-document questions correctly with page-level citations")


if __name__ == "__main__":
    main()
