from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    id: str
    prompt: str
    suggested_answer: str
    marks: float = 1.0
    question_type: str = "short"  # mcq | short | long | calculation


@dataclass
class StudentAnswer:
    question_id: str
    raw_text: str
    awarded: float = 0.0
    max_marks: float = 1.0
    match_reason: str = ""
    accepted_variant: Optional[str] = None


@dataclass
class StudentScript:
    filename: str
    class_name: str
    class_number: str
    ocr_text: str = ""
    pdf_bytes: bytes = b""
    handwritten_answers: list[tuple[str, str]] = field(default_factory=list)
    self_rated: Optional[float] = None
    answers: dict[str, StudentAnswer] = field(default_factory=dict)
    notes: str = ""

    @property
    def student_id(self) -> str:
        return f"{self.class_name}-{self.class_number}"

    @property
    def total(self) -> float:
        return round(sum(a.awarded for a in self.answers.values()), 2)


@dataclass
class AcceptedVariant:
    question_id: str
    text: str
    marks: float
    enabled: bool = True
