"""
Generates sample_docs/rag_demo_guide.pdf - the small document bundled with
the app so visitors can try it without uploading anything.

WHY generate it instead of bundling a real paper: a document we wrote
ourselves has no copyright questions, stays tiny (a few KB), and contains
facts we know exactly - so we can tell when an answer is right.

Dev-only dependency (not in requirements.txt):  pip install fpdf2
Run:  python scripts\\make_sample_pdf.py
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "rag_demo_guide.pdf"

SECTIONS = [
    ("1. What this demo is",
     "This is a document question-answering assistant built with retrieval-augmented "
     "generation (RAG). It answers questions using only the PDFs it has been given. "
     "This guide is the sample document that is loaded automatically, so visitors can "
     "try the app without uploading anything. Upload your own PDFs in the sidebar to ask "
     "questions about them instead. Uploaded files are processed in memory and are not "
     "stored: they disappear when the session ends or the Space restarts."),
    ("2. Why retrieval-augmented generation exists",
     "A large language model only knows what was in its training data. It has never seen "
     "your private reports, contracts or lecture notes, and its knowledge stops at a "
     "training cut-off date. Fine-tuning a model on your documents is expensive and goes "
     "stale as soon as a document changes. RAG avoids both problems: instead of changing "
     "the model, it looks up the relevant passages at question time and hands them to the "
     "model as context."),
    ("3. Indexing: done once per document",
     "When a PDF is processed, each page is extracted as text with PyPDFLoader, keeping the "
     "page number so answers can cite it. The text is then split into chunks of about 1000 "
     "characters with an overlap of 200 characters between neighbouring chunks. The overlap "
     "exists so that a sentence sitting on a chunk boundary still appears complete in at "
     "least one chunk. Each chunk is converted into a 384-number vector by the "
     "all-MiniLM-L6-v2 embedding model, and the vectors are stored in a Chroma vector "
     "database."),
    ("4. Querying: done for every question",
     "When you ask a question, the question is embedded with the same model, and Chroma "
     "returns the chunks whose vectors are closest by cosine similarity. With the Gemini "
     "generator the app retrieves the top 8 chunks; with the local generator it retrieves "
     "the top 4, because the local model can only read about 512 tokens. The chunks are "
     "placed into a prompt that tells the model to answer only from them, to cite page "
     "numbers, and to reply \"I don't have enough information in these documents to answer "
     "that\" when the chunks do not contain the answer."),
    ("5. The two generators",
     "The default generator is Google Gemini (gemini-3.5-flash-lite) on the free tier. It "
     "gives the best answers, but the question and retrieved text are sent to Google, and "
     "the free tier has a daily request limit shared by everyone using this demo. The "
     "fallback generator is google/flan-t5-base, a 250-million-parameter model that runs "
     "on the CPU with no API key. Nothing leaves the server, but its answers are short and "
     "noticeably less reliable. If no Gemini key is configured, or the daily limit is "
     "reached, the app switches to the local model and says so."),
    ("6. Known limitations",
     "Retrieval quality caps answer quality: the model cannot use a passage that was never "
     "retrieved. Scanned PDFs that contain images of text, rather than real text, will not "
     "work without OCR. The embedding model is trained mainly on English and handles other "
     "languages poorly. Similarity scores cannot reliably tell whether a question is "
     "answerable: in testing, an unanswerable question scored higher than several "
     "answerable ones."),
    ("7. A fact mentioned only once",
     "The project mascot is a capybara named Chunky. This sentence exists to test whether "
     "a single, unrelated fact can still be found by retrieval."),
]


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "RAG Document Q&A - Demo Guide", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Sample document bundled with the app.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for heading, body in SECTIONS:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(3)
    OUT.parent.mkdir(exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
