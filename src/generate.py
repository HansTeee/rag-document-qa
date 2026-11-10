"""
Generation: (question, retrieved chunks) -> answer text.

This is the second half of the QUERYING side of RAG. The retriever found the
relevant text; the LLM's only job is to read it and answer from it.

Two interchangeable generators share one interface:
  - GeminiGenerator: good answers, needs a free API key, text is sent to Google.
  - LocalGenerator:  no key, nothing leaves your machine, noticeably worse answers.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from retrieve import RetrievedChunk  # noqa: E402


# The exact refusal sentence. Kept as a constant so the app/tests can detect
# it reliably instead of each re-typing the string (and getting it wrong).
NO_ANSWER = "I don't have enough information in these documents to answer that."


class GeneratorError(Exception):
    """A problem the user can fix (missing key, invalid key, quota...).

    WHY a custom exception: the app can catch THIS type and show its friendly
    message, while genuine bugs still surface as normal tracebacks.
    """


# ---------------------------------------------------------------------------
# The prompt - this is where RAG succeeds or fails
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a question-answering assistant for a set of documents the user uploaded.
You will be given numbered context excerpts, each labelled with its source file and page number.

Rules - follow all of them:
1. Answer ONLY from the context excerpts. Do NOT use general knowledge or anything you learned in training, even if you are confident it is true.
2. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else:
   "{NO_ANSWER}"
   Do not guess.
3. Cite the source after every claim in the form (file name, p. N), using the labels on the excerpts.
4. Be concise. Quote exact numbers and terms from the excerpts rather than paraphrasing them."""
# WHY rule 2 is the most important line in the project: without it, whenever
# retrieval fails the model quietly answers from its pretraining, and the user
# cannot tell a document-grounded answer from an invented one.


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Turn retrieved chunks into a labelled context block.

    WHY label each excerpt with file + page: the model can only cite what it
    can see. If the labels aren't in the prompt, any page number it gives
    you is made up.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | file: {chunk.source_file} | page {chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """The user-turn message: the context first, then the question.

    WHY the question comes last: the model reads the evidence first, then the
    question it has to answer - the instruction it acts on is fresh.
    """
    return (
        f"Context excerpts:\n\n{format_context(chunks)}\n\n"
        f"---\nQuestion: {question}\n\n"
        f"Answer (following the rules, with citations):"
    )


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
class BaseGenerator(ABC):
    """Anything with .generate(question, context_chunks) -> str.

    WHY an abstract base class: rag.py and the app only ever call
    .generate(), so they never need to know which LLM is behind it.
    Adding a third backend later means writing one new class, nothing else.
    """

    name: str = "base"
    default_k: int = config.DEFAULT_K   # how many chunks this model can usefully read

    @abstractmethod
    def generate(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        ...


# ---------------------------------------------------------------------------
# Gemini (hosted, free tier)
# ---------------------------------------------------------------------------
class GeminiGenerator(BaseGenerator):
    """Uses Google's `google-genai` SDK.

    NOTE: the brief asked for the older `google-generativeai` package, but
    Google has ended all support for it (it prints a FutureWarning saying so).
    `google-genai` is its official replacement.
    """

    name = "gemini"
    default_k = config.GEMINI_K

    def __init__(self, api_key: str | None = None, model_name: str = config.GEMINI_MODEL_NAME):
        # Imported here, not at the top, so someone using only the local model
        # doesn't need google-genai installed for the file to import.
        from dotenv import load_dotenv
        from google import genai

        # load_dotenv copies KEY=value lines from .env into os.environ. It does
        # NOT override variables that are already set - so on Hugging Face
        # Spaces (where there is no .env) the Space's secret is used instead.
        load_dotenv(config.PROJECT_ROOT / ".env")
        key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise GeneratorError(
                "GEMINI_API_KEY is not set. Add it to a .env file in the project root "
                "(GEMINI_API_KEY=your-key) or set it as an environment variable. "
                "Get a free key at https://aistudio.google.com/apikey - or use the local model instead."
            )

        self._client = genai.Client(api_key=key)
        self._model_name = model_name
        # WHY we never store/print `key` beyond this point: error messages and
        # logs get copied into GitHub issues and screenshots.

    def generate(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        from google.genai import errors, types

        request_config = types.GenerateContentConfig(
            # WHY system_instruction instead of pasting the rules into the
            # prompt: models weight system instructions more heavily than
            # user text, which helps the "only from context" rule hold.
            system_instruction=SYSTEM_PROMPT,
            # WHY temperature 0: we want the most likely, most literal reading
            # of the context - not creative variation.
            temperature=0.0,
            seed=config.SEED,
            # WHY this cap is generous: Gemini 3.x "thinks" before answering,
            # and the thinking tokens count against this limit. At 512, the
            # model spent ~490 tokens thinking and the answer was cut off
            # mid-word (finish_reason=MAX_TOKENS) - found in Phase 2 testing.
            max_output_tokens=config.MAX_ANSWER_TOKENS,
            # "low" keeps a little reasoning (useful for judging whether the
            # excerpts REALLY answer the question) without the latency of more.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )

        # WHY retry: the free tier returns 429 (too many requests per minute)
        # and occasional 503 (overloaded). Both usually clear in seconds, so
        # waiting and retrying beats showing the user an error.
        waits = [5, 15, 30]
        for attempt in range(len(waits) + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=build_prompt(question, context_chunks),
                    config=request_config,
                )
                return (response.text or "").strip() or NO_ANSWER
            except (errors.ClientError, errors.ServerError) as e:
                # A per-DAY quota (the error mentions "PerDay") won't reset for
                # hours, so retrying would just make the user wait for nothing.
                daily_cap = e.code == 429 and "PerDay" in str(e)
                if daily_cap:
                    raise GeneratorError(
                        "Gemini's free-tier DAILY request limit for this model is used up. "
                        "It resets within 24 hours - switch to the local model meanwhile."
                    ) from None
                retryable = e.code in (429, 500, 503)
                if retryable and attempt < len(waits):
                    time.sleep(waits[attempt])
                    continue
                # Not retryable, or out of retries: explain in plain words.
                # We report the status code only - never the request, which
                # carries the key.
                if e.code in (400, 401, 403):
                    raise GeneratorError(
                        f"Gemini rejected the request (HTTP {e.code}) - usually an invalid "
                        "GEMINI_API_KEY. Check the key, or switch to the local model."
                    ) from None
                if e.code == 429:
                    raise GeneratorError(
                        "Gemini free-tier rate limit reached. Wait a minute and try again, "
                        "or switch to the local model."
                    ) from None
                raise GeneratorError(f"Gemini request failed (HTTP {e.code}). Try again shortly.") from None


# ---------------------------------------------------------------------------
# Local (flan-t5-base on CPU)
# ---------------------------------------------------------------------------
class LocalGenerator(BaseGenerator):
    """flan-t5-base via transformers. No key, no network after first download.

    The trade-off, honestly: flan-t5-base is a 250M-parameter model from 2022.
    It extracts short answers from context reasonably, but it does not follow
    multi-rule instructions well and won't reliably produce citations. The
    app shows the source chunks either way, so the user can still verify.
    """

    name = "local"
    default_k = config.LOCAL_K

    def __init__(self, model_name: str = config.LOCAL_MODEL_NAME):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # WHY load the model directly instead of transformers.pipeline():
        # pipeline task names changed between transformers versions; the
        # model classes are the stable API.
        torch.manual_seed(config.SEED)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.eval()   # inference mode: disables dropout
        self._torch = torch

    def _build_local_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """A much shorter prompt than Gemini's.

        WHY different: flan-t5 can only read 512 tokens, and the long rule
        list would use up a third of that. flan-t5 was trained on
        "Read this and answer the question" style tasks, so we use that form.

        WHY the question goes FIRST here: anything past token 512 is cut off.
        If the question came last (as in the Gemini prompt) it would be the
        part that gets cut, and the model would answer nothing. Putting the
        question first means truncation only trims the end of the
        lowest-ranked chunks.

        PROMPT ITERATION (Phase 2 testing): the first version asked flan-t5 to
        reply with our full refusal sentence. It never did - on both
        unanswerable test questions it copied a random span from the context
        instead (e.g. "42 to 210 Beats Per Minute" for "what learning rate?").
        flan-t5 was trained on SQuAD 2.0-style data whose refusal word is
        exactly "unanswerable", so we ask for THAT word and translate it to our
        sentence in generate(). This fixed both trap questions without breaking
        the answerable ones. Page labels were also dropped: citations come from
        the source list the app displays, and this small model can't use them.
        """
        context = "\n\n".join(c.text for c in chunks)
        return (
            "Read the text and answer the question. If the question is unanswerable "
            "from the text, say \"unanswerable\".\n\n"
            f"Question: {question}\n\nText:\n{context}"
        )

    def generate(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        prompt = self._build_local_prompt(question, context_chunks)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.LOCAL_MAX_INPUT_TOKENS,
        )
        with self._torch.no_grad():   # WHY: no training here, so skip gradient bookkeeping (faster, less RAM)
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,          # beam search: considers several candidate answers, picks the best
                do_sample=False,      # deterministic - same input, same output
            )
        answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # Translate flan-t5's trained refusal word into our standard sentence,
        # so the app shows the same refusal whichever generator is selected.
        if not answer or answer.lower().strip(" .\"'") == "unanswerable":
            return NO_ANSWER
        return answer


def get_generator(name: str) -> BaseGenerator:
    """Factory: 'gemini' or 'local' -> a generator instance."""
    if name == "gemini":
        return GeminiGenerator()
    if name == "local":
        return LocalGenerator()
    raise ValueError(f"Unknown generator {name!r}; expected 'gemini' or 'local'.")
