from __future__ import annotations

import re

from .models import Question

Q_START = re.compile(
    r"(?m)^\s*(?:question|q)?\s*(?P<id>\d+[a-z]?)\s*[.):]\s*",
    re.I,
)
MARKS = re.compile(
    r"\[(?P<m>\d+(?:\.\d+)?)(?:\s*marks?)?\]|\((?P<m2>\d+(?:\.\d+)?)\s*marks?\)|(?P<m3>\d+(?:\.\d+)?)\s*marks?",
    re.I,
)
ANSWER_SPLIT = re.compile(
    r"(?im)^\s*(?:suggested\s*)?(?:answer|ans|marking\s*points?|solution)\s*[:\-]\s*"
)
QUESTION_HINTS = re.compile(
    r"\?|\b(what|which|who|when|where|why|how|explain|define|calculate|describe|discuss|state|give|find|identify)\b",
    re.I,
)
METADATA_LABEL = re.compile(
    r"^(?:student\s*)?(?:name|class(?:\s*(?:name|no\.?|number|code|id))?|classno|"
    r"class\s*#|index(?:\s*no\.?)?|register(?:\s*no\.?)?|student\s*(?:no\.?|id|number)|"
    r"date|title|subject)\s*[:.\-]*$",
    re.I,
)
FILL_IN = "(Fill in the blank)"


def infer_type(prompt: str, answer: str) -> str:
    blob = f"{prompt}\n{answer}".lower()
    if prompt.strip() == FILL_IN:
        return "short"
    if re.search(r"\b[abcd]\s*[).]", blob) or "multiple choice" in blob:
        return "mcq"
    if re.search(r"[\d]+\s*[\+\-\*/=]", answer) or "calculate" in blob:
        return "calculation"
    if len(answer) > 80 or "explain" in blob or "discuss" in blob:
        return "long"
    return "short"


def question_sort_key(qid: str) -> tuple:
    parts: list[tuple[int, int | str]] = []
    for token in re.findall(r"\d+|[a-zA-Z]+", str(qid)):
        if token.isdigit():
            parts.append((0, int(token)))
        else:
            parts.append((1, token.lower()))
    return tuple(parts) or ((1, str(qid).lower()),)


def is_metadata_question(qid: str, prompt: str, answer: str) -> bool:
    for text in (prompt, answer, str(qid)):
        cleaned = re.sub(r"^\d+[a-z]?\s*[.):]\s*", "", text.strip(), flags=re.I)
        cleaned = cleaned.strip(" :.-")
        if cleaned and METADATA_LABEL.match(cleaned):
            return True
    return False


def _strip_qid_prefix(body: str, qid: str) -> str:
    return re.sub(rf"^(?:question|q)?\s*{re.escape(qid)}\s*[.):]\s*", "", body, flags=re.I).strip()


def _split_prompt_answer(body: str, qid: str) -> tuple[str, str]:
    body = _strip_qid_prefix(body, qid)
    parts = ANSWER_SPLIT.split(body, maxsplit=1)
    if len(parts) == 2:
        prompt, answer = parts[0].strip(), parts[1].strip()
        prompt = MARKS.sub("", prompt).strip()
        if not prompt or METADATA_LABEL.match(prompt):
            prompt = FILL_IN
        return prompt, answer

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return FILL_IN, ""
    joined = "\n".join(lines)
    prompt_line = MARKS.sub("", lines[0]).strip()
    rest = "\n".join(lines[1:]).strip()

    if QUESTION_HINTS.search(prompt_line) or "____" in prompt_line:
        return prompt_line, rest
    if rest and QUESTION_HINTS.search(rest) is None and len(prompt_line) > 80:
        return prompt_line, rest
    if rest and QUESTION_HINTS.search(prompt_line):
        return prompt_line, rest
    if not rest:
        return FILL_IN, prompt_line
    if not QUESTION_HINTS.search(joined):
        return FILL_IN, joined
    return prompt_line, rest


def _chunk_questions(text: str) -> list[tuple[str, str]]:
    matches = list(Q_START.finditer(text))
    if not matches:
        return []
    chunks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        qid = match.group("id").strip().lower()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append((qid, text[start:end].strip()))
    return chunks


def _prefer_text(current: str, incoming: str) -> str:
    if not current:
        return incoming
    if not incoming:
        return current
    if current == FILL_IN and incoming != FILL_IN:
        return incoming
    if len(incoming) > len(current):
        return incoming
    return current


def finalize_questions(questions: list[Question]) -> list[Question]:
    by_id: dict[str, Question] = {}
    for q in questions:
        qid = str(q.id).strip()
        if not qid:
            continue
        prompt = (q.prompt or "").strip() or FILL_IN
        answer = (q.suggested_answer or "").strip()
        if is_metadata_question(qid, prompt, answer):
            continue
        try:
            marks = float(q.marks) if q.marks else 1.0
        except (TypeError, ValueError):
            marks = 1.0
        if qid in by_id:
            existing = by_id[qid]
            existing.prompt = _prefer_text(existing.prompt, prompt)
            existing.suggested_answer = _prefer_text(existing.suggested_answer, answer)
            existing.marks = max(existing.marks, marks)
            existing.question_type = infer_type(existing.prompt, existing.suggested_answer)
            continue
        by_id[qid] = Question(
            id=qid,
            prompt=prompt,
            suggested_answer=answer,
            marks=marks,
            question_type=q.question_type or infer_type(prompt, answer),
        )
    return sorted(by_id.values(), key=lambda q: question_sort_key(q.id))


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
        prompt, answer = _split_prompt_answer(body, qid)
        questions.append(
            Question(
                id=qid,
                prompt=prompt,
                suggested_answer=MARKS.sub("", answer).strip(),
                marks=marks,
                question_type=infer_type(prompt, answer),
            )
        )
    return finalize_questions(questions)


def questions_from_openai_dicts(rows: list[dict]) -> list[Question]:
    out: list[Question] = []
    for i, row in enumerate(rows, start=1):
        qid = str(row.get("id") or i).strip()
        prompt = str(row.get("prompt") or "").strip() or FILL_IN
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
    return finalize_questions(out)
