from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
from PIL import Image

from .pdf_io import extract_pdf_text, pdf_pages_as_images


_OCR_ENGINE = None
_OCR_IMPORT_ERROR: Optional[Exception] = None

Q_NEAR = re.compile(r"(?i)\b(?:q(?:uestion)?\s*)?(\d{1,2}[a-z]?)\b")
CROSSED = re.compile(r"[×x]\s*$|~~")
PRINTED_LABEL = re.compile(r"(?i)^(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?[.)]?$")
MCQ_LETTER = re.compile(r"(?i)^[a-d][.)]?$")


def _get_rapidocr():
    global _OCR_ENGINE, _OCR_IMPORT_ERROR
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_IMPORT_ERROR is not None:
        raise _OCR_IMPORT_ERROR

    last_error: Optional[Exception] = None
    for module_name in ("rapidocr", "rapidocr_onnxruntime"):
        try:
            module = __import__(module_name, fromlist=["RapidOCR"])
            _OCR_ENGINE = module.RapidOCR()
            return _OCR_ENGINE
        except Exception as exc:
            last_error = exc
            continue
    _OCR_IMPORT_ERROR = last_error or ModuleNotFoundError("rapidocr")
    raise _OCR_IMPORT_ERROR


def _parse_ocr_result(result) -> list[tuple[str, tuple[float, float, float, float], float]]:
    items: list[tuple[str, tuple[float, float, float, float], float]] = []
    if result is None:
        return items
    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    if txts is not None and boxes is not None:
        if scores is None:
            scores = [0.8] * len(txts)
        for txt, box, score in zip(txts, boxes, scores):
            if not txt:
                continue
            arr = np.array(box, dtype=float)
            xs, ys = arr[:, 0], arr[:, 1]
            items.append(
                (
                    str(txt).strip(),
                    (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                    float(score),
                )
            )
        return items
    if isinstance(result, tuple):
        rows = result[0] or []
        for row in rows:
            if len(row) < 2 or not row[1]:
                continue
            box = np.array(row[0], dtype=float)
            xs, ys = box[:, 0], box[:, 1]
            score = float(row[2]) if len(row) > 2 else 0.8
            items.append(
                (
                    str(row[1]).strip(),
                    (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                    score,
                )
            )
    return items


def ocr_items(image: Image.Image) -> list[tuple[str, tuple[float, float, float, float], float]]:
    engine = _get_rapidocr()
    result = engine(np.array(image.convert("RGB")))
    return _parse_ocr_result(result)


def ocr_image(image: Image.Image) -> str:
    return "\n".join(t for t, _, _ in ocr_items(image) if t)


def _red_layer(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    try:
        import cv2
    except ImportError:
        # emphasise red channel without OpenCV
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = (r > 90) & (r > g + 15) & (r > b + 15)
        out = np.full_like(rgb, 255)
        out[mask] = rgb[mask]
        return Image.fromarray(out)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 30, 40), (18, 255, 255)) | cv2.inRange(hsv, (155, 30, 40), (180, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    out = np.full_like(rgb, 255)
    out[mask > 0] = rgb[mask > 0]
    return Image.fromarray(out)


def _find_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    try:
        import cv2
    except ImportError:
        return []
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 14 or bh < 12:
            continue
        area = bw * bh
        if area < 280 or area > 0.12 * w * h:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.45 or aspect > 14:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) < 4:
            continue
        roi = gray[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue
        white_ratio = float(np.mean(roi > 175))
        if white_ratio < 0.35:
            continue
        found.append((x, y, bw, bh))
    found.sort(key=lambda b: (round(b[1] / 18), b[0]))
    # drop nested boxes
    kept: list[tuple[int, int, int, int]] = []
    for box in found:
        x, y, bw, bh = box
        inner = False
        for ox, oy, ow, oh in kept:
            if x >= ox and y >= oy and x + bw <= ox + ow and y + bh <= oy + oh:
                inner = True
                break
        if not inner:
            kept.append(box)
    return kept[:50]


def _clean_snippet(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or CROSSED.search(ln):
            continue
        ln = re.sub(r"[×x]\s*$", "", ln).strip()
        if ln:
            lines.append(ln)
    return " ".join(lines).strip()


def _nearest_question_id(
    box: tuple[float, float, float, float],
    printed: list[tuple[str, tuple[float, float, float, float]]],
) -> Optional[str]:
    x0, y0, x1, y1 = box
    cy = (y0 + y1) / 2
    best = None
    best_dist = 1e9
    for text, (px0, py0, px1, py1) in printed:
        match = Q_NEAR.search(text.strip())
        if not match:
            if re.fullmatch(r"\d{1,2}[a-z]?", text.strip(), re.I):
                qid = text.strip().lower()
            else:
                continue
        else:
            qid = match.group(1).lower()
        pcy = (py0 + py1) / 2
        if abs(pcy - cy) > 48 and py1 > y0 + 10:
            continue
        # prefer number to the left or slightly above the blank/box
        if px0 > x1 + 30:
            continue
        dist = abs(pcy - cy) * 2 + max(0, x0 - px1)
        if dist < best_dist:
            best_dist = dist
            best = qid
    return best


def _red_ink_share(image: Image.Image) -> float:
    rgb = np.array(image.convert("RGB"))
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (r > 90) & (r > g + 15) & (r > b + 15)
    return float(np.mean(mask))


def _shrink(image: Image.Image, max_side: int = 1600) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def _center_in_rect(box: tuple[float, float, float, float], rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    x, y, w, h = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return x <= cx <= x + w and y <= cy <= y + h


def _find_char_enclosures(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Circles and small character-sized boxes used for MCQ keys."""
    try:
        import cv2
    except ImportError:
        return []
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    found: list[tuple[int, int, int, int]] = []
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=80,
        param2=18,
        minRadius=7,
        maxRadius=max(18, int(min(h, w) * 0.06)),
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            found.append((int(cx - r), int(cy - r), int(2 * r), int(2 * r)))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 10:
            continue
        area = bw * bh
        if area < 120 or area > 3500:
            continue
        aspect = bw / max(bh, 1)
        if not 0.65 <= aspect <= 1.5:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        circular = 4 * np.pi * max(cv2.contourArea(cnt), 1) / max(peri * peri, 1)
        if len(approx) >= 4 or circular > 0.65:
            found.append((x, y, bw, bh))
    found.sort(key=lambda b: (round(b[1] / 16), b[0]))
    kept: list[tuple[int, int, int, int]] = []
    for box in found:
        x, y, bw, bh = box
        if any(x >= ox and y >= oy and x + bw <= ox + ow and y + bh <= oy + oh for ox, oy, ow, oh in kept):
            continue
        kept.append(box)
    return kept[:40]


def _looks_printed_stem(text: str, score: float) -> bool:
    if PRINTED_LABEL.match(text):
        return True
    if len(text) > 28 and score >= 0.9:
        return True
    if score >= 0.96 and MCQ_LETTER.match(text):
        return True
    return False


def _looks_handwritten(text: str, score: float, in_blank: bool) -> bool:
    snippet = _clean_snippet(text)
    if not snippet or PRINTED_LABEL.match(snippet):
        return False
    if in_blank and re.search(r"[A-Za-z]{2,}", snippet):
        return True
    if in_blank and MCQ_LETTER.match(snippet):
        return True
    if score < 0.91 and re.search(r"[A-Za-z]{2,}", snippet):
        return True
    if score < 0.88 and re.search(r"[a-z]{2,}", snippet):
        return True
    return False


def _text_inside(items, rect: tuple[int, int, int, int]) -> str:
    hits = [text for text, box, *_ in items if _center_in_rect(box, rect)]
    return _clean_snippet(" ".join(hits))


def is_answer_like(text: str, *, allow_mcq: bool) -> bool:
    t = _clean_snippet(text)
    if not t:
        return False
    if PRINTED_LABEL.match(t):
        return False
    if MCQ_LETTER.match(t):
        return allow_mcq
    if re.search(r"[A-Za-z]{2,}", t):
        return True
    return False


def _reading_order(items):
    return sorted(items, key=lambda it: (round(it[1][1] / 22), it[1][0]))


def extract_handwritten_answers(image: Image.Image) -> tuple[str, list[tuple[str, str]]]:
    """Student scripts: keep handwriting in blanks; ignore printed numbers/letters."""
    image = _shrink(image)
    full_items = ocr_items(image)
    blanks = _find_boxes(image)
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text, box, score in _reading_order(full_items):
        snippet = _clean_snippet(text)
        in_blank = any(_center_in_rect(box, rect) for rect in blanks)
        if _looks_printed_stem(snippet, score) and not in_blank:
            continue
        if not _looks_handwritten(snippet, score, in_blank):
            continue
        if not is_answer_like(snippet, allow_mcq=in_blank):
            continue
        key = f"{snippet.lower()}|{int(box[1] / 12)}"
        if key in seen:
            continue
        seen.add(key)
        qid = _nearest_question_id(box, [(t, b) for t, b, _ in full_items]) or ""
        ordered.append((qid, snippet))
    lines = [f"{(qid or i)}. {text}" for i, (qid, text) in enumerate(ordered, start=1)]
    return "\n".join(lines), ordered


def extract_scheme_answers(image: Image.Image) -> list[tuple[str, str]]:
    """Marking scheme: red ink first, then circled / character-bordered letters."""
    image = _shrink(image)
    full_items = ocr_items(image)
    red_items = ocr_items(_red_layer(image)) if _red_ink_share(image) >= 0.0003 else []
    enclosures = _find_char_enclosures(image)
    printed = [(t, b) for t, b, _ in full_items]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(qid: str, snippet: str) -> None:
        snippet = _clean_snippet(snippet)
        if not snippet or PRINTED_LABEL.match(snippet):
            return
        key = f"{qid}|{snippet.lower()}"
        if key in seen:
            return
        seen.add(key)
        out.append((qid, snippet))

    for text, box, _score in _reading_order(red_items):
        snippet = _clean_snippet(text)
        if is_answer_like(snippet, allow_mcq=True) or MCQ_LETTER.match(snippet):
            add(_nearest_question_id(box, printed) or "", snippet)

    pool = red_items or full_items
    for rect in enclosures:
        snippet = _text_inside(red_items, rect) or _text_inside(pool, rect)
        if not snippet:
            continue
        if not (MCQ_LETTER.match(snippet) or is_answer_like(snippet, allow_mcq=True)):
            continue
        qid = _nearest_question_id(
            (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), printed
        ) or ""
        add(qid, snippet)

    if not out:
        for text, box, _score in _reading_order(full_items):
            snippet = _clean_snippet(text)
            if is_answer_like(snippet, allow_mcq=False):
                add(_nearest_question_id(box, printed) or "", snippet)
    return out


def ocr_pdf_bytes(data: bytes) -> tuple[str, Optional[str], list[tuple[str, str]]]:
    structured: list[tuple[str, str]] = []
    try:
        images = pdf_pages_as_images(data, scale=1.5)
        parts = []
        for img in images:
            text, items = extract_handwritten_answers(img)
            if text:
                parts.append(text)
            structured.extend(items)
        ocr_text = "\n".join(parts).strip()
    except Exception as exc:
        return "", f"Image OCR unavailable ({exc}).", []

    if structured:
        numbered = "\n".join(
            f"{qid or i}. {ans}" for i, (qid, ans) in enumerate(structured, start=1) if ans
        )
        return numbered or ocr_text, None, structured
    return ocr_text, None, []


def transcribe_scheme_pdf(data: bytes) -> tuple[str, list[tuple[str, str]]]:
    """OCR a marking scheme, treating red ink as the official answers."""
    native = extract_pdf_text(data)
    structured: list[tuple[str, str]] = []
    try:
        images = pdf_pages_as_images(data, scale=1.6)
        for img in images:
            structured.extend(extract_scheme_answers(img))
    except Exception:
        structured = []
    filled: list[tuple[str, str]] = []
    used: set[str] = set()
    n = 1
    for qid, ans in structured:
        ans = ans.strip()
        if not ans:
            continue
        if qid and qid not in used:
            filled.append((qid, ans))
            used.add(qid)
            continue
        if qid and qid in used:
            continue
        while str(n) in used:
            n += 1
        filled.append((str(n), ans))
        used.add(str(n))
        n += 1
    if filled:
        return "\n".join(f"{qid}. {ans}" for qid, ans in filled), filled
    return native, []


def openai_transcribe_script(images: list[Image.Image], api_key: str, model: str) -> str:
    import base64
    from io import BytesIO

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Transcribe a scanned student worksheet. Focus on HANDWRITING in blanks and boxes, "
                "not printed questions, question numbers, or printed option letters. "
                "Ignore crossed-out text (×). Ignore name/class headers. "
                "Return one line per question as '1. answer' using the printed question numbers."
            ),
        }
    ]
    for img in images:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def transcribe_student_pdf(
    data: bytes, engine: str, openai_key: Optional[str], openai_model: str
) -> tuple[str, Optional[str], list[tuple[str, str]]]:
    if engine == "openai" and openai_key:
        images = pdf_pages_as_images(data, scale=1.8)
        text = openai_transcribe_script(images, openai_key, openai_model)
        from .parse_scheme import split_numbered_items

        structured = split_numbered_items(text)
        return text, None, structured
    return ocr_pdf_bytes(data)


def openai_extract_scheme(text: str, api_key: str, model: str) -> Optional[list[dict]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "Extract marking-scheme items as JSON with key 'questions'. "
        "Each item: id (string like '1' or '1a'), prompt, suggested_answer, "
        "marks (number), question_type (mcq|short|long|calculation). "
        "If the source is a list of answers such as 'visited 2. were walking 3. went', "
        "create separate items 1=visited, 2=were walking, 3=went. "
        "Red ink is the official answer. Circled or character-bordered letters (A–D) are MCQ keys. "
        "Ignore name, class, class number, underscore blanks, 'F.1', Group, and subject headers "
        "(e.g. 'IC F.1 Group 8 English') — those are not questions. "
        "Fill-in-the-blank items may have no question stem; set prompt to "
        "'(Fill in the blank)' and put the key in suggested_answer. "
        "Default marks to 1 if unspecified. Return JSON only.\n\n"
        f"{text[:12000]}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    questions = data.get("questions")
    if isinstance(questions, list):
        return questions
    return None


def resolve_openai_key(manual: str) -> str:
    return (manual or os.getenv("OPENAI_API_KEY") or "").strip()
