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


def _parse_ocr_result(result) -> list[tuple[str, tuple[float, float, float, float]]]:
    items: list[tuple[str, tuple[float, float, float, float]]] = []
    if result is None:
        return items
    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    if txts is not None and boxes is not None:
        for txt, box in zip(txts, boxes):
            if not txt:
                continue
            arr = np.array(box, dtype=float)
            xs, ys = arr[:, 0], arr[:, 1]
            items.append((str(txt).strip(), (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))))
        return items
    if isinstance(result, tuple):
        rows = result[0] or []
        for row in rows:
            if len(row) < 2 or not row[1]:
                continue
            box = np.array(row[0], dtype=float)
            xs, ys = box[:, 0], box[:, 1]
            items.append((str(row[1]).strip(), (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))))
    return items


def ocr_items(image: Image.Image) -> list[tuple[str, tuple[float, float, float, float]]]:
    engine = _get_rapidocr()
    result = engine(np.array(image.convert("RGB")))
    return _parse_ocr_result(result)


def ocr_image(image: Image.Image) -> str:
    return "\n".join(t for t, _ in ocr_items(image) if t)


def _red_layer(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    try:
        import cv2
    except ImportError:
        # emphasise red channel without OpenCV
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = (r > 120) & (r > g + 25) & (r > b + 25)
        out = np.full_like(rgb, 255)
        out[mask] = rgb[mask]
        return Image.fromarray(out)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (12, 255, 255)) | cv2.inRange(hsv, (165, 50, 50), (180, 255, 255))
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
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


def _crop(image: Image.Image, box: tuple[int, int, int, int], pad: int = 4) -> Image.Image:
    x, y, w, h = box
    return image.crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad))


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


def extract_handwritten_answers(image: Image.Image) -> tuple[str, list[tuple[str, str]]]:
    """Return (full OCR text, list of (question_id or '', handwritten answer))."""
    full_items = ocr_items(image)
    red_items = ocr_items(_red_layer(image))
    red_texts = {t.lower() for t, _ in red_items if t}
    printed = [(t, b) for t, b in full_items if t.lower() not in red_texts]

    boxed: list[tuple[str, str]] = []
    for rect in _find_boxes(image):
        crop = _crop(image, rect)
        red_crop = _red_layer(crop)
        snippet = _clean_snippet(ocr_image(red_crop) or ocr_image(crop))
        if not snippet:
            continue
        qid = _nearest_question_id(
            (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), printed + red_items
        )
        boxed.append((qid or "", snippet))

    if boxed:
        ordered = boxed
    else:
        ordered = [( _nearest_question_id(b, printed) or "", t) for t, b in red_items if _clean_snippet(t)]
        ordered = [(q, _clean_snippet(t)) for q, t in ordered if t]

    lines = []
    for i, (qid, text) in enumerate(ordered, start=1):
        label = qid or str(i)
        lines.append(f"{label}. {text}")
    full_text = "\n".join(t for t, _ in full_items)
    red_text = "\n".join(t for t, _ in red_items)
    handwritten_block = "\n".join(lines)
    combined = "\n".join(p for p in (handwritten_block, red_text, full_text) if p)
    return combined, ordered


def ocr_pdf_bytes(data: bytes) -> tuple[str, Optional[str], list[tuple[str, str]]]:
    native = extract_pdf_text(data)
    structured: list[tuple[str, str]] = []
    try:
        images = pdf_pages_as_images(data, scale=2.2)
        parts = []
        for img in images:
            text, items = extract_handwritten_answers(img)
            if text:
                parts.append(text)
            structured.extend(items)
        ocr_text = "\n".join(parts).strip()
    except Exception as exc:
        return native, f"Image OCR unavailable ({exc}). Used extracted PDF text only.", []

    if structured:
        numbered = "\n".join(
            f"{qid or i}. {ans}" for i, (qid, ans) in enumerate(structured, start=1) if ans
        )
        return numbered or ocr_text or native, None, structured
    if ocr_text:
        return ocr_text, None, []
    return native, None, []


def openai_transcribe_script(images: list[Image.Image], api_key: str, model: str) -> str:
    import base64
    from io import BytesIO

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Transcribe a scanned student worksheet. Focus on HANDWRITING and RED ink, "
                "not printed questions. MCQ answers are often a single letter (A–D) inside a box, "
                "sometimes written in red. Fill-in answers sit on blanks or in boxes. "
                "Ignore crossed-out text (×). Ignore name/class headers and underscores. "
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
        "Ignore name, class, class number, underscore blanks, 'F.1', Group, subject headers "
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
