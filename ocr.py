from __future__ import annotations

import json
import os
from typing import Optional

# openai is imported only when vision OCR / scheme structuring is used.

from PIL import Image

from .pdf_io import extract_pdf_text, pdf_pages_as_images


_OCR_ENGINE = None


def _get_rapidocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def ocr_image(image: Image.Image) -> str:
    import numpy as np

    engine = _get_rapidocr()
    result = engine(np.array(image.convert("RGB")))
    if result is None:
        return ""
    txts = getattr(result, "txts", None)
    if txts:
        return "\n".join(str(t) for t in txts if t)
    if isinstance(result, tuple):
        rows = result[0] or []
        return "\n".join(row[1] for row in rows if len(row) > 1 and row[1])
    return str(result).strip()


def ocr_pdf_bytes(data: bytes) -> str:
    native = extract_pdf_text(data)
    images = pdf_pages_as_images(data)
    ocr_parts = [ocr_image(img) for img in images]
    ocr_text = "\n".join(p for p in ocr_parts if p).strip()
    if native and len(native) >= max(40, int(0.4 * len(ocr_text or native))):
        return native if len(native) >= len(ocr_text) else f"{native}\n\n{ocr_text}"
    if ocr_text:
        return f"{native}\n\n{ocr_text}".strip() if native else ocr_text
    return native


def openai_transcribe_script(images: list[Image.Image], api_key: str, model: str) -> str:
    from io import BytesIO

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are marking a scanned student assignment. Transcribe ALL handwriting "
                "and printed text faithfully. Keep original question numbers. "
                "If the student wrote a self-score (e.g. 'self: 8/10'), keep it. "
                "Return plain text only."
            ),
        }
    ]
    for img in images:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        import base64

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


def transcribe_student_pdf(data: bytes, engine: str, openai_key: Optional[str], openai_model: str) -> str:
    if engine == "openai" and openai_key:
        images = pdf_pages_as_images(data, scale=1.6)
        return openai_transcribe_script(images, openai_key, openai_model)
    return ocr_pdf_bytes(data)


def openai_extract_scheme(text: str, api_key: str, model: str) -> Optional[list[dict]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "Extract marking-scheme questions as JSON with key 'questions'. "
        "Each item: id (string like '1' or '1a'), prompt, suggested_answer, "
        "marks (number), question_type (mcq|short|long|calculation). "
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
