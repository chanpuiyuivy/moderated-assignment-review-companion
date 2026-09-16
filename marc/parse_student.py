from __future__ import annotations

import re
from typing import Optional

from .models import Question, StudentAnswer, StudentScript

SELF_SCORE = re.compile(
    r"(?i)self[\s\-_]?(?:rated|rating|score|mark)?\s*[:\-]?\s*(?P<s>\d+(?:\.\d+)?)(?:\s*/\s*(?P<t>\d+(?:\.\d+)?))?"
)


def extract_self_rated(text: str) -> Optional[float]:
    match = SELF_SCORE.search(text)
    if not match:
        return None
    try:
        return float(match.group("s"))
    except ValueError:
        return None


def split_student_answers(text: str, questions: list[Question]) -> dict[str, str]:
    """Slice OCR text into per-question answers using scheme question ids."""
    if not questions:
        return {}
    pattern_parts = []
    for q in questions:
        qid = re.escape(q.id)
        pattern_parts.append(rf"(?:question\s*)?{qid}\s*[.)]|{qid}\s*[.)]")
    splitter = re.compile(r"(?im)^\s*(" + "|".join(pattern_parts) + r")")
    matches = list(splitter.finditer(text))
    assigned: dict[str, str] = {q.id: "" for q in questions}
    if not matches:
        assigned[questions[0].id] = text.strip()
        return assigned

    id_from_header = re.compile(r"(?i)(?:question\s*)?(\d+[a-z]?|[a-z])")
    for i, match in enumerate(matches):
        header = match.group(1)
        found = id_from_header.search(header)
        qid = found.group(1).lower() if found else questions[min(i, len(questions) - 1)].id
        if qid not in assigned:
            for q in questions:
                if q.id.lower() == qid or q.id.lower().startswith(qid):
                    qid = q.id
                    break
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        assigned[qid] = (assigned.get(qid, "") + "\n" + body).strip()
    return assigned


def attach_answers(script: StudentScript, questions: list[Question]) -> None:
    mapped = split_student_answers(script.ocr_text, questions)
    script.self_rated = extract_self_rated(script.ocr_text)
    script.answers = {}
    for q in questions:
        script.answers[q.id] = StudentAnswer(
            question_id=q.id,
            raw_text=mapped.get(q.id, ""),
            awarded=0.0,
            max_marks=q.marks,
        )
