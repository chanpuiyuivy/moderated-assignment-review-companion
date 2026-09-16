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
PRINTED_LABEL = re.compile(
    r"(?i)^(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?[.)]?$|^\(\s*\d{1,2}[a-z]?\s*\)$"
)
MCQ_LETTER = re.compile(r"(?i)^[a-d][.)]?$")
ANSWER_QNUM = re.compile(
    r"(?i)(?:^\(\s*\d{1,2}[a-z]?\s*\)\s*)|(?:\(\s*\d{1,2}[a-z]?\s*\))|(?:^(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?\s*[.)]\s*)"
)


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
            try:
                _OCR_ENGINE = module.RapidOCR(params={"Global.use_cls": False})
            except TypeError:
                try:
                    _OCR_ENGINE = module.RapidOCR(use_angle_cls=False)
                except TypeError:
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


def strip_answer_qnums(text: str) -> str:
    """Remove markers like (1), 1., Q1 from an answer string."""
    t = _clean_snippet(text)
    t = re.sub(r"\(\s*\d{1,2}[a-z]?\s*\)", " ", t, flags=re.I)
    t = re.sub(r"^(?:q(?:uestion)?\s*)?\d{1,2}[a-z]?\s*[.)]\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if PRINTED_LABEL.match(t):
        return ""
    return t


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
        # Running headers / page numbers at the very top are not question ids.
        if py1 < 42 and re.fullmatch(r"\d{1,3}", qid):
            continue
        pcy = (py0 + py1) / 2
        if abs(pcy - cy) > 72 and py1 > y0 + 10:
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


def _shrink(image: Image.Image, max_side: int = 1200) -> Image.Image:
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
        minDist=14,
        param1=70,
        param2=14,
        minRadius=6,
        maxRadius=max(28, int(min(h, w) * 0.08)),
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


def _crop_rect(image: Image.Image, rect: tuple[int, int, int, int], pad: int = 4) -> Image.Image:
    x, y, w, h = rect
    return image.crop(
        (max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad)
    )


def _ink_ratio(image: Image.Image, rect: tuple[int, int, int, int]) -> float:
    crop = np.array(_crop_rect(image, rect, pad=0).convert("L"))
    if crop.size == 0:
        return 0.0
    h, w = crop.shape
    y0, y1 = max(1, h // 6), h - max(1, h // 6)
    x0, x1 = max(1, w // 6), w - max(1, w // 6)
    inner = crop[y0:y1, x0:x1]
    if inner.size == 0:
        inner = crop
    return float(np.mean(inner < 150))


def _as_mcq_letter(text: str) -> str:
    snippet = strip_answer_qnums(text)
    if not snippet or not MCQ_LETTER.match(snippet):
        return ""
    return re.sub(r"[^A-Da-d]", "", snippet).upper()


def _mcq_groups(cands: list[dict]) -> list[list[dict]]:
    """Nearby A–D marks belong to one question; a repeated letter starts a new group."""
    remaining = list(cands)
    groups: list[list[dict]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            leftover: list[dict] = []
            used = {c["letter"] for c in group}
            for cand in remaining:
                if cand["letter"] in used:
                    leftover.append(cand)
                    continue
                dist = min(
                    abs(cand["cx"] - g["cx"]) + abs(cand["cy"] - g["cy"]) * 1.4 for g in group
                )
                if dist < 150:
                    group.append(cand)
                    used.add(cand["letter"])
                    changed = True
                else:
                    leftover.append(cand)
            remaining = leftover
        groups.append(group)
    return groups


def _choose_mcq_answer(group: list[dict]) -> tuple[str, str]:
    best = max(group, key=lambda c: (c.get("red", 0.0), c.get("ink", 0.0)))
    qid = best.get("qid") or ""
    if not qid:
        counts: dict[str, int] = {}
        for cand in group:
            q = cand.get("qid") or ""
            if q:
                counts[q] = counts.get(q, 0) + 1
        if counts:
            qid = max(counts, key=counts.get)
    return qid, best["letter"]


def _find_table_cells(image: Image.Image) -> list[tuple[int, int, int, int]]:
    try:
        import cv2
    except ImportError:
        return []
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 25), 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 25, 25))))
    grid = cv2.add(hor, ver)
    grid = cv2.dilate(grid, np.ones((2, 2), np.uint8), iterations=1)
    holes = cv2.bitwise_not(grid)
    holes = cv2.erode(holes, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells: list[tuple[int, int, int, int]] = []
    page = w * h
    for cnt in contours:
        x, y, bw_, bh = cv2.boundingRect(cnt)
        area = bw_ * bh
        if area < 400 or area > 0.2 * page:
            continue
        if bw_ < 18 or bh < 14:
            continue
        cells.append((x, y, bw_, bh))
    cells.sort(key=lambda b: (round(b[1] / 16), b[0]))
    return cells[:40]


def extract_handwritten_answers(image: Image.Image) -> tuple[str, list[tuple[str, str]], str]:
    """Answers in blanks, table cells, or circled MCQ options. One OCR pass per page."""
    image = _shrink(image)
    full_items = ocr_items(image)
    printed = [(t, b) for t, b, _ in full_items]
    header_cut = image.size[1] * 0.38
    header = "\n".join(t for t, b, _ in full_items if b[1] < header_cut)
    cells = _find_table_cells(image)
    blanks = _find_boxes(image)
    circles = _find_char_enclosures(image)

    def qid_for(rect: tuple[int, int, int, int], local_text: list[str]) -> str:
        for t in local_text:
            m = re.search(r"(?i)(?:q(?:uestion)?\s*)?(\d{1,2}[a-z]?)|\((\d{1,2}[a-z]?)\)", t)
            if m:
                return (m.group(1) or m.group(2)).lower()
        return _nearest_question_id(
            (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), printed
        ) or ""

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(qid: str, snippet: str) -> None:
        snippet = strip_answer_qnums(snippet)
        if not snippet or PRINTED_LABEL.match(snippet):
            return
        key = f"{qid}|{snippet.lower()}"
        if key in seen:
            return
        seen.add(key)
        ordered.append((qid, snippet))

    mcq_cands: list[dict] = []
    for rect in circles:
        inside = [strip_answer_qnums(t) for t, b, _ in full_items if _center_in_rect(b, rect)]
        letters = [_as_mcq_letter(t) for t in inside]
        letters = [t for t in letters if t]
        if not letters:
            continue
        letter = letters[0]
        qid = qid_for(rect, inside)
        x, y, w, h = rect
        mcq_cands.append(
            {
                "letter": letter,
                "qid": qid,
                "cx": x + w / 2,
                "cy": y + h / 2,
                "red": _red_ink_share(_crop_rect(image, rect, pad=2)),
                "ink": _ink_ratio(image, rect),
                "box": (x, y, x + w, y + h),
            }
        )
    for group in _mcq_groups(mcq_cands):
        qid, letter = _choose_mcq_answer(group)
        add(qid, letter)

    regions = cells if cells else blanks
    for rect in regions:
        inside_items = [(t, b) for t, b, _ in full_items if _center_in_rect(b, rect)]
        local = [t for t, _ in inside_items]
        leftover = [
            strip_answer_qnums(t)
            for t in local
            if strip_answer_qnums(t) and not PRINTED_LABEL.match(t) and not MCQ_LETTER.match(t)
        ]
        words = [t for t in leftover if re.search(r"[A-Za-z]{2,}", t)]
        if words:
            add(qid_for(rect, local), " ".join(words))

    if not ordered:
        for text, box, score in _reading_order(full_items):
            snippet = strip_answer_qnums(text)
            if _looks_printed_stem(text, score):
                continue
            if not re.search(r"[A-Za-z]{2,}", snippet) and not MCQ_LETTER.match(snippet):
                continue
            add(_nearest_question_id(box, printed) or "", snippet)

    lines = [f"{(qid or i)}. {text}" for i, (qid, text) in enumerate(ordered, start=1)]
    return "\n".join(lines), ordered, header


def _header_text(image: Image.Image) -> str:
    w, h = image.size
    top = image.crop((0, 0, w, max(40, int(h * 0.38))))
    return "\n".join(t for t, _, _ in ocr_items(_shrink(top)) if t)


def extract_scheme_answers(image: Image.Image) -> list[tuple[str, str]]:
    """Marking scheme: red ink first; MCQ options collapse to one letter per question."""
    image = _shrink(image)
    full_items = ocr_items(image)
    red_items = ocr_items(_red_layer(image))
    printed = [(t, b) for t, b, _ in full_items]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    paired = re.compile(r"(?i)^(?:q(?:uestion)?\s*)?(\d{1,2}[a-z]?)\s*[.)]?\s*([A-D])(?:\s*[.)])?$")

    def add(qid: str, snippet: str) -> None:
        snippet = strip_answer_qnums(snippet)
        if MCQ_LETTER.match(snippet):
            snippet = re.sub(r"[^A-Da-d]", "", snippet).upper()
        if not snippet or PRINTED_LABEL.match(snippet):
            return
        key = f"{qid}|{snippet.lower()}"
        if key in seen:
            return
        seen.add(key)
        out.append((qid, snippet))

    for text, box, _score in _reading_order(red_items):
        m = paired.match(text.strip())
        if m:
            add(m.group(1).lower(), m.group(2).upper())
            continue
        snippet = strip_answer_qnums(text)
        if MCQ_LETTER.match(snippet):
            continue
        if is_answer_like(snippet, allow_mcq=False):
            add(_nearest_question_id(box, printed) or "", snippet)

    mcq_cands: list[dict] = []
    for text, box, _score in red_items:
        letter = _as_mcq_letter(text)
        if not letter:
            continue
        x0, y0, x1, y1 = box
        mcq_cands.append(
            {
                "letter": letter,
                "qid": _nearest_question_id(box, printed) or "",
                "cx": (x0 + x1) / 2,
                "cy": (y0 + y1) / 2,
                "red": 1.0,
                "ink": 1.0,
                "box": box,
            }
        )
    for rect in _find_char_enclosures(image):
        snippet = _text_inside(red_items, rect) or _text_inside(full_items, rect)
        letter = _as_mcq_letter(snippet)
        if not letter:
            continue
        red = _red_ink_share(_crop_rect(image, rect, pad=2))
        ink = _ink_ratio(image, rect)
        if red < 0.012 and ink < 0.20:
            continue
        x, y, w, h = rect
        box = (x, y, x + w, y + h)
        mcq_cands.append(
            {
                "letter": letter,
                "qid": _nearest_question_id(box, printed) or "",
                "cx": x + w / 2,
                "cy": y + h / 2,
                "red": red,
                "ink": ink,
                "box": box,
            }
        )
    for group in _mcq_groups(mcq_cands):
        qid, letter = _choose_mcq_answer(group)
        add(qid, letter)

    if not any(MCQ_LETTER.match(a) for _, a in out):
        for text, box, _score in _reading_order(full_items):
            m = paired.match(text.strip())
            if m:
                add(m.group(1).lower(), m.group(2).upper())
    if not out:
        for text, box, _score in _reading_order(full_items):
            snippet = strip_answer_qnums(text)
            if MCQ_LETTER.match(snippet):
                continue
            if is_answer_like(snippet, allow_mcq=False):
                add(_nearest_question_id(box, printed) or "", snippet)
    return out


def ocr_pdf_bytes(data: bytes) -> tuple[str, Optional[str], list[tuple[str, str]]]:
    structured: list[tuple[str, str]] = []
    try:
        images = pdf_pages_as_images(data, scale=1.2)
        parts = []
        for img in images:
            text, items, _header = extract_handwritten_answers(img)
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


def _next_scheme_qid(used: set[str]) -> str:
    n = 1
    while str(n) in used:
        n += 1
    return str(n)


def transcribe_scheme_pdf(data: bytes, on_progress=None) -> tuple[str, list[tuple[str, str]]]:
    """OCR a marking scheme, treating red ink as the official answers."""
    native = extract_pdf_text(data)
    structured: list[tuple[str, str]] = []
    try:
        images = pdf_pages_as_images(data, scale=1.35)
        total = max(len(images), 1)
        if on_progress:
            on_progress(0, total, f"Starting marking-scheme OCR ({total} page(s))…")
        for i, img in enumerate(images):
            if on_progress:
                on_progress(i, total, f"Recognising page {i + 1} of {total}…")
            structured.extend(extract_scheme_answers(img))
            if on_progress:
                on_progress(i + 1, total, f"Finished page {i + 1} of {total}")
    except Exception:
        structured = []
        if on_progress:
            on_progress(1, 1, "OCR failed")
    filled: list[tuple[str, str]] = []
    used: set[str] = set()
    for qid, ans in structured:
        ans = strip_answer_qnums(ans.strip())
        if not ans:
            continue
        if qid and qid not in used:
            filled.append((qid, ans))
            used.add(qid)
            continue
        if qid and qid in used:
            prev = next(a for k, a in filled if k == qid)
            if MCQ_LETTER.match(ans) and not MCQ_LETTER.match(prev):
                filled = [(k, ans if k == qid else a) for k, a in filled]
            continue
        nid = _next_scheme_qid(used)
        filled.append((nid, ans))
        used.add(nid)
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
) -> tuple[str, Optional[str], list[tuple[str, str]], tuple[str, str, str]]:
    from .filename import extract_identity_from_text
    from .parse_scheme import split_numbered_items

    images = pdf_pages_as_images(data, scale=1.4 if engine == "openai" else 1.2)
    ident = ("", "", "")

    def _merge_ident(extra: str) -> tuple[str, str, str]:
        other = extract_identity_from_text(extra)
        return (ident[0] or other[0], ident[1] or other[1], ident[2] or other[2])

    if engine == "openai" and openai_key:
        if images:
            ident = extract_identity_from_text(_header_text(images[0]))
        text = openai_transcribe_script(images, openai_key, openai_model)
        return text, None, split_numbered_items(text), _merge_ident(text)

    structured: list[tuple[str, str]] = []
    parts: list[str] = []
    header = ""
    try:
        for i, img in enumerate(images):
            text, items, page_header = extract_handwritten_answers(img)
            if i == 0:
                header = page_header
                ident = extract_identity_from_text(page_header)
            if text:
                parts.append(text)
            structured.extend(items)
    except Exception as exc:
        return "", f"Image OCR unavailable ({exc}).", [], ident
    ocr_text = "\n".join(parts).strip()
    if structured:
        numbered = "\n".join(
            f"{qid or i}. {ans}" for i, (qid, ans) in enumerate(structured, start=1) if ans
        )
        ocr_text = numbered or ocr_text
    return ocr_text, None, structured, _merge_ident(header + "\n" + ocr_text)


def openai_extract_scheme(text: str, api_key: str, model: str) -> Optional[list[dict]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "Extract marking-scheme items as JSON with key 'questions'. "
        "Each item: id (string like '1' or '1a'), prompt, suggested_answer, marks (number). "
        "If the source is a list of answers such as 'visited 2. were walking 3. went', "
        "create separate items 1=visited, 2=were walking, 3=went. "
        "Red ink is the official answer. For multiple choice, create ONE question and put "
        "only the keyed option letter (A–D) in suggested_answer — do not create a row for "
        "every printed option. Circled or character-bordered letters are the MCQ key. "
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
