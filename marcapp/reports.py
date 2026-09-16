from __future__ import annotations

from collections import Counter
from statistics import mean, median, pstdev

from .models import Question, StudentScript


def class_stats(students: list[StudentScript]) -> dict:
    scores = [s.total for s in students]
    if not scores:
        return {
            "n": 0,
            "average": 0.0,
            "median": 0.0,
            "max": 0.0,
            "min": 0.0,
            "stdev": 0.0,
        }
    return {
        "n": len(scores),
        "average": round(mean(scores), 2),
        "median": round(median(scores), 2),
        "max": round(max(scores), 2),
        "min": round(min(scores), 2),
        "stdev": round(pstdev(scores), 2) if len(scores) > 1 else 0.0,
    }


def max_total(questions: list[Question]) -> float:
    return round(sum(q.marks for q in questions), 2)


def student_sort_key(student: StudentScript) -> tuple:
    num = int(student.class_number) if str(student.class_number).isdigit() else 0
    return (student.class_name, num, student.filename)


def mistakes_for(student: StudentScript, questions: list[Question]) -> list[str]:
    return _mistakes(student, questions)


def improvement_for(student: StudentScript, questions: list[Question]) -> str:
    return _improvement(student, questions)


def _mistakes(student: StudentScript, questions: list[Question]) -> list[str]:
    lines = []
    by_id = {q.id: q for q in questions}
    for qid, ans in student.answers.items():
        q = by_id.get(qid)
        if not q:
            continue
        if ans.awarded < q.marks:
            lost = q.marks - ans.awarded
            expected = q.suggested_answer.strip() or "(no key)"
            got = ans.raw_text.strip() or "(blank)"
            lines.append(
                f"- Q{qid} (lost {lost:g}/{q.marks:g}): "
                f"wrote “{got[:180]}”; expected “{expected[:180]}”."
            )
    return lines


def _improvement(student: StudentScript, questions: list[Question]) -> str:
    missed: list[str] = []
    by_id = {q.id: q for q in questions}
    for qid, ans in student.answers.items():
        q = by_id.get(qid)
        if q and ans.awarded < q.marks:
            missed.append(qid)
    if not missed:
        return "This student met the marking scheme on every question."
    shown = ", ".join(f"Q{qid}" for qid in missed[:8])
    extra = f" (+{len(missed) - 8} more)" if len(missed) > 8 else ""
    return f"Marks were lost on {shown}{extra}. Review those items against the scheme."


def individual_report(students: list[StudentScript], questions: list[Question]) -> str:
    ceiling = max_total(questions)
    chunks = ["# Individual report\n"]
    for s in sorted(students, key=lambda x: (x.class_name, int(x.class_number) if x.class_number.isdigit() else 0)):
        chunks.append(f"## {s.student_id} ({s.filename})")
        chunks.append(f"- App score: **{s.total:g} / {ceiling:g}**")
        if s.self_rated is not None:
            delta = round(s.self_rated - s.total, 2)
            direction = "overestimated" if delta > 0 else "underestimated" if delta < 0 else "matched"
            chunks.append(
                f"- Self-rated score: **{s.self_rated:g}** (discrepancy {delta:+g}; student {direction})"
            )
        else:
            chunks.append("- Self-rated score: not found on the script")
        mistakes = _mistakes(s, questions)
        chunks.append("### Mistakes")
        chunks.extend(mistakes or ["- None recorded."])
        chunks.append("### Room for improvement")
        chunks.append(_improvement(s, questions))
        chunks.append("")
    return "\n".join(chunks)


def question_report(students: list[StudentScript], questions: list[Question]) -> str:
    n = max(len(students), 1)
    chunks = ["# Report by question\n"]
    for q in questions:
        chunks.append(f"## Question {q.id} ({q.marks:g} mark(s))")
        chunks.append(q.prompt.strip() or "_No prompt text._")
        chunks.append(f"**Suggested answer:** {q.suggested_answer.strip() or '(none)'}")
        freq: Counter[str] = Counter()
        awards = []
        for s in students:
            ans = s.answers.get(q.id)
            text = (ans.raw_text.strip() if ans and ans.raw_text.strip() else "(blank)")
            freq[text] += 1
            awards.append(ans.awarded if ans else 0.0)
        avg = round(sum(awards) / n, 2)
        chunks.append(f"**Mean mark:** {avg:g} / {q.marks:g}")
        chunks.append("**Answer frequencies:**")
        for text, count in freq.most_common():
            chunks.append(f"- {count} student(s): {text[:300]}")
        chunks.append("")
    return "\n".join(chunks)


def overall_report(students: list[StudentScript], questions: list[Question]) -> str:
    stats = class_stats(students)
    ceiling = max_total(questions)
    lines = [
        "# Overall class report\n",
        f"- Students: **{stats['n']}**",
        f"- Maximum available: **{ceiling:g}**",
        f"- Average: **{stats['average']}**",
        f"- Median: **{stats['median']}**",
        f"- Maximum: **{stats['max']}**",
        f"- Minimum: **{stats['min']}**",
        f"- Standard deviation: **{stats['stdev']}**",
        "",
        "## Score list",
    ]
    for s in sorted(students, key=lambda x: -x.total):
        self_bit = f"; self-rated {s.self_rated:g}" if s.self_rated is not None else ""
        lines.append(f"- {s.student_id}: {s.total:g}{self_bit}")
    return "\n".join(lines)
