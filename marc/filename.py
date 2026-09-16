from __future__ import annotations

import re
from pathlib import Path


FILENAME_PATTERNS = [
    re.compile(r"(?P<cls>[0-9]{1,2}[A-Za-z])[-_ ]?(?P<num>\d{1,3})", re.I),
    re.compile(r"(?P<cls>[A-Za-z]{1,6}[0-9]{1,2}[A-Za-z]?)[-_ ]?(?:no|num|#)?[-_ ]?(?P<num>\d{1,3})", re.I),
    re.compile(r"(?P<cls>[A-Za-z0-9]+)[-_ ](?P<num>\d{1,3})", re.I),
]


def parse_student_filename(name: str) -> tuple[str, str]:
    """Return (class_name, class_number) from a PDF filename."""
    stem = Path(name).stem.strip()
    for pattern in FILENAME_PATTERNS:
        match = pattern.search(stem)
        if match:
            cls = match.group("cls").upper()
            num = str(int(match.group("num")))
            return cls, num
    return "UNKNOWN", stem
