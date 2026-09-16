from __future__ import annotations

import re
from pathlib import Path


FILENAME_PATTERNS = [
    re.compile(r"(?P<cls>[0-9]{1,2}[A-Za-z])[-_ ]?(?P<num>\d{1,3})", re.I),
    re.compile(
        r"(?P<cls>[A-Za-z]{1,6}[0-9]{1,2}[A-Za-z]?)[-_ ]?(?:no|num)?[-_ ]?(?P<num>\d{1,3})",
        re.I,
    ),
    re.compile(r"(?P<cls>[A-Za-z0-9]+)[-_ ](?P<num>\d{1,3})", re.I),
]


def parse_student_filename(name: str) -> tuple[str, str]:
    """Return (class_name, class_number) from a PDF filename."""
    stem = Path(name).stem.strip()
    for pattern in FILENAME_PATTERNS:
        match = pattern.search(stem)
        if match:
            cls = match.group("cls").upper().replace(" ", "")
            num = str(int(match.group("num")))
            return cls, num
    return "UNKNOWN", ""


def filename_has_class_info(name: str) -> bool:
    cls, num = parse_student_filename(name)
    return cls != "UNKNOWN" and bool(num)


def extract_identity_from_text(text: str) -> tuple[str, str, str]:
    """Return (name, class, class_number) found on a script header."""
    blob = (text or "").replace("\r", "\n")
    name = ""
    cls = ""
    num = ""
    m = re.search(
        r"(?i)\b(?:student\s*)?name\s*[:\-.]?\s*([A-Za-z][A-Za-z .'\-]{1,48})",
        blob,
    )
    if m:
        name = re.split(r"(?i)\s+(?:class|no|number)\b", m.group(1), maxsplit=1)[0]
        name = re.sub(r"\s+", " ", name).strip(" .-")
        if name.lower() in {"class", "no", "number"}:
            name = ""
    m = re.search(
        r"(?i)\bclass\s*[:\-.]?\s*([0-9]{1,2}\s*[A-Za-z]|[A-Za-z]\.?\s*[0-9]{1,2})",
        blob,
    )
    if m:
        cls = re.sub(r"\s+", "", m.group(1)).upper()
        if cls.lower().startswith("no"):
            cls = ""
    m = re.search(r"(?i)\b(?:class\s*)?(?:no\.?|number)\s*[:\-.]?\s*([0-9]{1,3})\b", blob)
    if m:
        num = str(int(m.group(1)))
    return name, cls, num
