from __future__ import annotations

import re

from .models import Question

FILL_IN = "(Fill in the blank)"

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
ITEM_MARK = re.compile(
    r"(?:(?<=^)|(?<=[\s;|/]))(?P<id>\d+[a-z]?)[.)]\s+",
    re.I | re.M,
)
IDENTITY_LINE = re.compile(
    r"(?ix)^\s*"
    r"(?:(?:student\s*)?name|class(?:\s*(?:name|no\.?|number|code|id))?|classno|"
    r"index(?:\s*no\.?)?|register(?:\s*no\.?)?|student\s*(?:no\.?|id|number)|date|subject|title)"
    r"\b[\s:.\-_/]*[\w\s.]*$"
)
HEADER_LINE = re.compile(
    r"(?ix).*(?:_{3,}|\bF\.\s*\d\b|\bGroup\s*\d+\b|\bIC\b|"
    r"\b(?:English|Mathematics|Maths|Chinese|Science|History|Geography)\b).*"
)


def _plain(text: str) -> str:
    text = re.sub(r"^\d+[a-z]?\s*[.)]\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"[_—–\-.:/#]+", " ", text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_worksheet_header(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False
    if raw.count("_") >= 4:
        return True
    if re.search(r"_+\s*\([^)]*\)", raw):
        return True
    if HEADER_LINE.search(raw) and (
        raw.count("_") >= 2 or re.search(r"\bF\.\s*\d\b", raw) or re.search(r"\bGroup\s*\d+", raw, re.I)
    ):
        return True
    return False


def looks_like_identity(text: str) -> bool:
    if looks_like_worksheet_header(text):
        return True
    cleaned = _plain(text)
    if not cleaned:
        return False
    if IDENTITY_LINE.match(text.strip()):
        return True
    words = [w for w in cleaned.split() if w not in {"the", "a", "an", "of", "my", "your"}]
    meta = {
        "name",
        "class",
        "no",
        "number",
        "classname",
        "classno",
        "index",
        "register",
        "student",
        "id",
        "date",
        "subject",
        "title",
    }
    if words and set(words) <= meta:
        return True
    if re.match(r"^(student )?name\b", cleaned) and len(words) <= 5:
        return True
    if re.match(r"^class( no| number| name)?\b", cleaned) and len(words) <= 6:
        return True
    return False


def strip_identity_fields(text: str) -> str:
    kept: list[str] = []
    for line in text.replace("\r\n", "\n").splitlines():
        if looks_like_identity(line):
            continue
        stripped = re.sub(
            r"(?i)\b(?:student\s*)?name\s*[:\-].*?(?=\bclass\b|\d+\s*[.)]|$)",
            " ",
            line,
        )
        stripped = re.sub(
            r"(?i)\bclass(?:\s*(?:no\.?|number))?\s*[:\-]\s*[A-Za-z0-9\-]*",
            " ",
            stripped,
        )
        if looks_like_identity(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def infer_type(prompt: str, answer: str) -> str:
    if re.fullmatch(r"[A-Da-d]", answer.strip()):
        return "mcq"
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
    if looks_like_identity(qid) or looks_like_identity(prompt) or looks_like_identity(answer):
        return True
    if looks_like_identity(f"{prompt} {answer}"):
        return True
    return False


def split_numbered_items(text: str) -> list[tuple[str, str]]:
    """Split 'visited 2. were walking 3. went' into (1, visited), (2, were walking), (3, went)."""
    text = strip_identity_fields(text).strip()
    if not text:
        return []
    matches = list(ITEM_MARK.finditer(text))
    if not matches:
        return []
    items: list[tuple[str, str]] = []
    first = matches[0]
    prefix = text[: first.start()].strip(" \n\t;|-")
    first_id = first.group("id").lower()
    if prefix and not looks_like_identity(prefix):
        if first_id.isdigit() and int(first_id) >= 2:
            items.append((str(int(first_id) - 1), prefix))
        elif not first_id.isdigit():
            items.append(("1", prefix))
    for i, match in enumerate(matches):
        qid = match.group("id").lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip(" \n\t;|")
        if looks_like_identity(body):
            continue
        items.append((qid, body))
    return items


def _split_prompt_answer(body: str, qid: str) -> tuple[str, str]:
    body = re.sub(rf"^(?:question|q)?\s*{re.escape(qid)}\s*[.):]\s*", "", body, flags=re.I).strip()
    parts = ANSWER_SPLIT.split(body, maxsplit=1)
    if len(parts) == 2:
        prompt, answer = parts[0].strip(), parts[1].strip()
        prompt = MARKS.sub("", prompt).strip()
        if not prompt or looks_like_identity(prompt):
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
    if not rest:
        return FILL_IN, prompt_line
    if not QUESTION_HINTS.search(joined):
        return FILL_IN, joined
    return prompt_line, rest


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


def strip_answer_qnums(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\(\s*\d{1,2}[a-z]?\s*\)", " ", t, flags=re.I)
    t = re.sub(r"^(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?\s*[.)]\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def finalize_questions(questions: list[Question]) -> list[Question]:
    by_id: dict[str, Question] = {}
    for q in questions:
        qid = str(q.id).strip()
        if not qid:
            continue
        prompt = (q.prompt or "").strip() or FILL_IN
        answer = strip_answer_qnums((q.suggested_answer or "").strip())
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
    questions: list[Question] = []
    for qid, body in split_numbered_items(text):
        marks = 1.0
        mark_match = MARKS.search(body)
        if mark_match:
            raw = mark_match.group("m") or mark_match.group("m2") or mark_match.group("m3")
            try:
                marks = float(raw)
            except ValueError:
                marks = 1.0
        prompt, answer = _split_prompt_answer(body, qid)
        answer = strip_answer_qnums(answer)
        if is_metadata_question(qid, prompt, answer):
            continue
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
