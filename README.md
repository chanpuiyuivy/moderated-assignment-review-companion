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

Use **Next** and **Back** at the bottom of each screen (there is no step list in the sidebar).

1. **Upload scripts** — one PDF per student. Name files with class and class number (`4A12.pdf`, `4A-12.pdf`, `5B_03.pdf`). Read scripts, then **Next**.
2. **Upload scheme** — PDF, Word (`.docx`), or pasted text. Parse, then **Next**.
3. **Review scheme** — edit wording, answers, and marks (default **1**). Fill-in-the-blank items may only show the answer. Name / class / class number are not treated as questions. Then **Next**.
4. **Moderate answers** — **by question** (each ID, then all alternative answers for that ID); **by student** (fix OCR, override marks). Then **View final report**.
5. **Final reports** — individual, by question, and overall (average, median, max, min, standard deviation). Download Markdown and CSV.

## Recognition

- **RapidOCR** (default): local, no API key. Works best on reasonably clear scans.
- **OpenAI vision**: better on messy handwriting. Paste a key in the sidebar (or set `OPENAI_API_KEY`).

Self-rated scores are picked up from phrases such as `self: 8/10` or `self-rated 7`.

Handwriting OCR is never perfect — use the review screens to correct text and accepted wordings before you trust the final reports.
