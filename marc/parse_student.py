from __future__ import annotations

import re
from typing import Optional

from .models import Question, StudentAnswer, StudentScript
from .parse_scheme import looks_like_identity, split_numbered_items

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
    assigned: dict[str, str] = {q.id: "" for q in questions}
    if not questions:
        return assigned
    items = split_numbered_items(text)
    id_map = {q.id.lower(): q.id for q in questions}
    for qid, body in items:
        body = body.strip()
        if not body or looks_like_identity(body):
            continue
        target = id_map.get(qid.lower())
        if target is None:
            for q in questions:
                if q.id.lower().startswith(qid.lower()) or qid.lower().startswith(q.id.lower()):
                    target = q.id
                    break
        if target:
            assigned[target] = (assigned[target] + "\n" + body).strip()
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
