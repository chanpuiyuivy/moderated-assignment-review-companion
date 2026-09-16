# MARC

**MARC** is the **Moderated Assignment Review Companion**. It is a Streamlit app that reads scanned student PDFs, compares them with a marking scheme, lets you moderate alternative answers, and writes three reports (individual, by question, whole class).

## Run locally

```powershell
cd $env:USERPROFILE\moderated-assignment-review-companion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. **Student scripts** — upload one PDF per student. Name files with class and class number (`4A12.pdf`, `4A-12.pdf`, `5B_03.pdf`). The app OCRs handwriting/print. You can edit a transcript before continuing.
2. **Marking scheme** — upload PDF or Word (`.docx`), or paste text. Optionally use OpenAI to structure messy schemes.
3. **Review scheme** — edit question wording, suggested answers, marks (default **1**), and type (`mcq` / `short` / `long` / `calculation`).
4. **Moderate & mark** — tentative class stats; **by question** (answer frequencies, accept alternatives and their marks); **by student** (fix OCR, override marks, self-rated score).
5. **Final reports** — individual (totals, self-score discrepancy, mistakes, improvement); by question (answer counts, weak types); overall (average, median, max, min, standard deviation). Download Markdown and CSV.

## Recognition

- **RapidOCR** (default): local, no API key. Works best on reasonably clear scans.
- **OpenAI vision**: better on messy handwriting. Paste a key in the sidebar (or set `OPENAI_API_KEY`).

Self-rated scores are picked up from phrases such as `self: 8/10` or `self-rated 7`.

Handwriting OCR is never perfect — use the review screens to correct text and accepted wordings before you trust the final reports.
