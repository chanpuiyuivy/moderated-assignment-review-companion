from __future__ import annotations

import re

from .models import Question

Q_START = re.compile(
    r"(?m)^\s*(?:question\s*)?(?P<id>\d+[a-z]?|\([a-z]\)|[a-z]\))\s*[.)]?\s+",
    re.I,
)
MARKS = re.compile(
    r"\[(?P<m>\d+(?:\.\d+)?)(?:\s*marks?)?\]|\((?P<m2>\d+(?:\.\d+)?)\s*marks?\)|(?P<m3>\d+(?:\.\d+)?)\s*marks?",
    re.I,
)
ANSWER_SPLIT = re.compile(
    r"(?im)^\s*(?:suggested\s*)?(?:answer|ans|marking\s*points?|solution)\s*[:\-]\s*"
)


def infer_type(prompt: str, answer: str) -> str:
    blob = f"{prompt}\n{answer}".lower()
    if re.search(r"\b[abcd]\s*[).]", blob) or "multiple choice" in blob:
        return "mcq"
    if re.search(r"[\d]+\s*[\+\-\*/=]", answer) or "calculate" in blob:
        return "calculation"
    if len(answer) > 80 or "explain" in blob or "discuss" in blob:
        return "long"
    return "short"


def _chunk_questions(text: str) -> list[tuple[str, str]]:
    matches = list(Q_START.finditer(text))
    if not matches:
        return [("1", text.strip())]
    chunks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        qid = match.group("id").strip("(). ").lower()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append((qid, body))
    return chunks


def parse_scheme_text(text: str) -> list[Question]:
    cleaned = text.replace("\r\n", "\n")
    questions: list[Question] = []
    for qid, body in _chunk_questions(cleaned):
        marks = 1.0
        mark_match = MARKS.search(body)
        if mark_match:
            raw = mark_match.group("m") or mark_match.group("m2") or mark_match.group("m3")
            try:
                marks = float(raw)
            except ValueError:
                marks = 1.0
        parts = ANSWER_SPLIT.split(body, maxsplit=1)
        if len(parts) == 2:
            prompt, answer = parts[0].strip(), parts[1].strip()
        else:
            lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            if len(lines) >= 2:
                prompt, answer = lines[0], "\n".join(lines[1:])
            else:
                prompt, answer = body.strip(), ""
        prompt = MARKS.sub("", prompt).strip()
        questions.append(
            Question(
                id=qid,
                prompt=prompt,
                suggested_answer=answer,
                marks=marks,
                question_type=infer_type(prompt, answer),
            )
        )
    return questions


def questions_from_openai_dicts(rows: list[dict]) -> list[Question]:
    out: list[Question] = []
    for i, row in enumerate(rows, start=1):
        qid = str(row.get("id") or i)
        prompt = str(row.get("prompt") or "").strip()
        answer = str(row.get("suggested_answer") or row.get("answer") or "").strip()
        try:
            marks = float(row.get("marks") or 1)
        except (TypeError, ValueError):
            marks = 1.0
        qtype = str(row.get("question_type") or infer_type(prompt, answer))
        out.append(
            Question(
                id=qid,
                prompt=prompt,
                suggested_answer=answer,
                marks=marks,
                question_type=qtype,
            )
        )
    return out
