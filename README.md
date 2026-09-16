# M.A.R.C

**M.A.R.C** is the **Moderated Assignment Review Companion**. It is a Streamlit app that reads scanned student PDFs, compares them with a marking scheme, lets you moderate alternative answers, and writes three reports (individual, by question, whole class).

## Run locally

```powershell
cd $env:USERPROFILE\moderated-assignment-review-companion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Workflow

Use **Next** and **Back** at the bottom of each screen.

1. **Upload scripts** — one PDF per student. Name files with class and class number (`4A12.pdf`, `4A-12.pdf`, `5B_03.pdf`). Read scripts, then **Next**.
2. **Marking scheme** — upload or re-upload PDF/Word, or paste text. Edit, add, or delete questions on the same page, then **Next**.
3. **Moderate answers** — **by question** (each ID, alternative answers, links to student PDFs); **by student** (fix OCR, override marks). Statistics update as you edit. Then **View final report**.
4. **Final reports** — one student at a time; by-question charts/lists; overall table, histogram, and sortable score list (below 50% in pink). Download Markdown and CSV.

## Recognition

- **RapidOCR** (default): local, no API key. Works best on reasonably clear scans.
- **OpenAI vision**: better on messy handwriting. Paste a key in the sidebar (or set `OPENAI_API_KEY`).

Self-rated scores are picked up from phrases such as `self: 8/10` or `self-rated 7`.

Handwriting OCR is never perfect — use the review screens to correct text and accepted wordings before you trust the final reports.
