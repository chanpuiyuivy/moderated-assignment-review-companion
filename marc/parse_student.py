from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz

from .models import Question, StudentAnswer, StudentScript
from .parse_scheme import FILL_IN, looks_like_identity, split_numbered_items

SELF_SCORE = re.compile(
    r"(?i)self[\s\-_]?(?:rated|rating|score|mark)?\s*[:\-]?\s*(?P<s>\d+(?:\.\d+)?)(?:\s*/\s*(?P<t>\d+(?:\.\d+)?))?"
)
CROSSED = re.compile(r"[×x]\s*$")


def extract_self_rated(text: str) -> Optional[float]:
    match = SELF_SCORE.search(text)
    if not match:
        return None
    try:
        return float(match.group("s"))
    except ValueError:
        return None


def _clean_line(text: str) -> str:
    text = text.strip()
    if CROSSED.search(text) or "×" in text:
        text = re.sub(r"×.*$", "", text).strip()
        if CROSSED.search(text) or not text:
            return ""
    return text


def _looks_printed_stem(line: str, questions: list[Question]) -> bool:
    if len(line) < 8:
        return False
    for q in questions:
        prompt = (q.prompt or "").strip()
        if not prompt or prompt == FILL_IN:
            continue
        if fuzz.partial_ratio(line.lower(), prompt.lower()) >= 86:
            return True
    return False


def _put(assigned: dict[str, str], id_map: dict[str, str], qid: str, body: str, questions: list[Question]) -> None:
    body = _clean_line(body)
    if not body or looks_like_identity(body) or _looks_printed_stem(body, questions):
        return
    if re.fullmatch(r"(?i)(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?[.)]?", body):
        return
    target = id_map.get(qid.lower())
    if target is None:
        for key, real in id_map.items():
            if key.startswith(qid.lower()) or qid.lower().startswith(key):
                target = real
                break
    if target and not assigned[target]:
        assigned[target] = body
    elif target:
        # keep the first snippet for that id; do not glue later questions together
        if len(body) < len(assigned[target]) and assigned[target].count("\n") > 0:
            assigned[target] = body


def split_student_answers(
    text: str,
    questions: list[Question],
    handwritten: Optional[list[tuple[str, str]]] = None,
) -> dict[str, str]:
    assigned: dict[str, str] = {q.id: "" for q in questions}
    if not questions:
        return assigned
    id_map = {q.id.lower(): q.id for q in questions}

    unlabeled: list[str] = []
    if handwritten:
        for qid, body in handwritten:
            body = _clean_line(body)
            if not body or looks_like_identity(body):
                continue
            if re.fullmatch(r"(?i)(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?[.)]?", body):
                continue
            if qid and qid.lower() in id_map:
                _put(assigned, id_map, qid, body, questions)
            else:
                unlabeled.append(body)
        empties = [q.id for q in questions if not assigned[q.id]]
        for qid, body in zip(empties, unlabeled):
            assigned[qid] = body

    if not any(assigned.values()):
        for qid, body in split_numbered_items(text):
            _put(assigned, id_map, qid, body, questions)

    if not any(assigned.values()):
        lines = []
        for ln in text.splitlines():
            ln = _clean_line(ln)
            ln = re.sub(r"^\d+[a-z]?\s*[.)]\s*", "", ln).strip()
            if not ln or looks_like_identity(ln) or _looks_printed_stem(ln, questions):
                continue
            if len(ln) > 60:
                continue
            lines.append(ln)
        for q, ln in zip(questions, lines):
            assigned[q.id] = ln

    return assigned


def attach_answers(script: StudentScript, questions: list[Question]) -> None:
    mapped = split_student_answers(script.ocr_text, questions, script.handwritten_answers)
    script.self_rated = extract_self_rated(script.ocr_text)
    script.answers = {}
    for q in questions:
        script.answers[q.id] = StudentAnswer(
            question_id=q.id,
            raw_text=mapped.get(q.id, ""),
            awarded=0.0,
            max_marks=q.marks,
        )
