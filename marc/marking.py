from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz

from .models import AcceptedVariant, Question, StudentAnswer, StudentScript


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(normalize(a).split()), set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def similarity(student: str, key: str) -> float:
    if not student.strip() or not key.strip():
        return 0.0
    ns, nk = normalize(student), normalize(key)
    if ns == nk:
        return 1.0
    return max(
        fuzz.token_set_ratio(ns, nk) / 100.0,
        fuzz.ratio(ns, nk) / 100.0,
        _token_overlap(student, key),
    )


def score_answer(
    student_text: str,
    question: Question,
    variants: list[AcceptedVariant],
) -> tuple[float, str, str | None]:
    keys = [question.suggested_answer] + [v.text for v in variants if v.enabled and v.text.strip()]
    best_score = 0.0
    best_reason = "no match"
    best_variant = None
    best_marks = 0.0

    for idx, key in enumerate(keys):
        sim = similarity(student_text, key)
        marks_cap = question.marks if idx == 0 else next(
            (v.marks for v in variants if v.enabled and v.text == key), question.marks
        )
        threshold = 0.72 if question.question_type in {"long", "calculation"} else 0.82
        partial_cut = 0.55 if question.question_type == "long" else 0.7
        if sim >= threshold:
            award = marks_cap
            reason = f"accepted match ({sim:.0%})"
        elif sim >= partial_cut:
            award = round(marks_cap * 0.5, 2)
            reason = f"partial match ({sim:.0%})"
        else:
            award = 0.0
            reason = f"low similarity ({sim:.0%})"
        if award > best_marks or (award == best_marks and sim > best_score):
            best_marks = award
            best_score = sim
            best_reason = reason
            best_variant = None if idx == 0 else key

    if not student_text.strip():
        return 0.0, "blank", None
    return best_marks, best_reason, best_variant


def apply_marking(
    students: list[StudentScript],
    questions: list[Question],
    variants: list[AcceptedVariant],
) -> None:
    by_q: dict[str, list[AcceptedVariant]] = defaultdict(list)
    for v in variants:
        by_q[v.question_id].append(v)
    for student in students:
        for q in questions:
            ans = student.answers.get(q.id)
            if ans is None:
                ans = StudentAnswer(question_id=q.id, raw_text="", max_marks=q.marks)
                student.answers[q.id] = ans
            ans.max_marks = q.marks
            awarded, reason, variant = score_answer(ans.raw_text, q, by_q.get(q.id, []))
            ans.awarded = awarded
            ans.match_reason = reason
            ans.accepted_variant = variant


def cluster_answers(students: list[StudentScript], question_id: str, cutoff: int = 85) -> list[dict]:
    groups: list[dict] = []
    for student in students:
        ans = student.answers.get(question_id)
        text = (ans.raw_text if ans else "").strip()
        if not text:
            text = "(blank)"
        placed = False
        for group in groups:
            if fuzz.token_set_ratio(normalize(text), normalize(group["canonical"])) >= cutoff:
                group["count"] += 1
                group["students"].append(student.student_id)
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "canonical": text if text != "(blank)" else "",
                    "label": text,
                    "count": 1,
                    "students": [student.student_id],
                }
            )
    groups.sort(key=lambda g: (-g["count"], g["label"]))
    return groups
