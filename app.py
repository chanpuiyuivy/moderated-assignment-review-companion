from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from marc.filename import parse_student_filename
from marc.marking import apply_marking, cluster_answers
from marc.models import AcceptedVariant, Question, StudentScript
from marc.ocr import resolve_openai_key, transcribe_student_pdf, openai_extract_scheme
from marc.parse_scheme import parse_scheme_text, questions_from_openai_dicts
from marc.parse_student import attach_answers
from marc.pdf_io import extract_scheme_text
from marc.reports import class_stats, individual_report, max_total, overall_report, question_report

STEPS = [
    "Student scripts",
    "Marking scheme",
    "Review scheme",
    "Moderate & mark",
    "Final reports",
]


def init_state() -> None:
    defaults = {
        "step": 0,
        "students": [],
        "questions": [],
        "variants": [],
        "scheme_text": "",
        "processed": False,
        "marked": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def questions_to_frame(questions: list[Question]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": q.id,
                "prompt": q.prompt,
                "suggested_answer": q.suggested_answer,
                "marks": q.marks,
                "question_type": q.question_type,
            }
            for q in questions
        ]
    )


def frame_to_questions(df: pd.DataFrame) -> list[Question]:
    rows: list[Question] = []
    for _, row in df.iterrows():
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        try:
            marks = float(row.get("marks") or 1)
        except (TypeError, ValueError):
            marks = 1.0
        qtype = str(row.get("question_type") or "short").strip() or "short"
        rows.append(
            Question(
                id=qid,
                prompt=str(row.get("prompt") or ""),
                suggested_answer=str(row.get("suggested_answer") or ""),
                marks=marks,
                question_type=qtype,
            )
        )
    return rows


def sidebar_settings() -> tuple[str, str, str]:
    st.sidebar.markdown("**MARC**")
    st.sidebar.caption("Moderated Assignment Review Companion")
    st.sidebar.header("Recognition")
    engine = st.sidebar.selectbox(
        "Handwriting / scan engine",
        options=["rapidocr", "openai"],
        format_func=lambda x: "Local RapidOCR (no API)" if x == "rapidocr" else "OpenAI vision (better handwriting)",
    )
    openai_key = st.sidebar.text_input("OpenAI API key", type="password", value="")
    model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")
    st.sidebar.caption(
        "Filenames should encode class and number, e.g. `4A12.pdf`, `4A-12.pdf`, `5B_03.pdf`."
    )
    return engine, openai_key, model


def page_students(engine: str, openai_key: str, model: str) -> None:
    st.header("1. Upload student assignments")
    st.write(
        "Upload one PDF per student. The app reads printed text and handwriting, "
        "then later splits answers by question number."
    )
    files = st.file_uploader("Student PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        preview = []
        for f in files:
            cls, num = parse_student_filename(f.name)
            preview.append({"file": f.name, "class": cls, "number": num})
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    if st.button("Read scripts (OCR)", type="primary", disabled=not files):
        key = resolve_openai_key(openai_key)
        if engine == "openai" and not key:
            st.error("Paste an OpenAI API key in the sidebar, or switch to local RapidOCR.")
            return
        students: list[StudentScript] = []
        progress = st.progress(0.0, text="Starting…")
        for i, f in enumerate(files):
            cls, num = parse_student_filename(f.name)
            data = f.getvalue()
            progress.progress((i + 0.3) / len(files), text=f"Reading {f.name}")
            try:
                text = transcribe_student_pdf(data, engine, key, model)
            except Exception as exc:
                st.warning(f"{f.name}: OCR failed ({exc}). Stored empty transcript.")
                text = ""
            students.append(
                StudentScript(filename=f.name, class_name=cls, class_number=num, ocr_text=text)
            )
            progress.progress((i + 1) / len(files), text=f"Finished {f.name}")
        st.session_state.students = students
        st.session_state.processed = True
        st.success(f"Loaded {len(students)} script(s).")

    if st.session_state.students:
        with st.expander("Preview transcripts", expanded=not st.session_state.processed):
            pick = st.selectbox(
                "Student",
                options=st.session_state.students,
                format_func=lambda s: s.student_id,
            )
            pick.ocr_text = st.text_area("Recognised text (editable)", value=pick.ocr_text, height=240)
        if st.button("Continue to marking scheme"):
            go(1)


def page_scheme(engine: str, openai_key: str, model: str) -> None:
    st.header("2. Upload marking scheme")
    uploaded = st.file_uploader("Marking scheme", type=["pdf", "docx"])
    pasted = st.text_area("Or paste the scheme text", height=180)
    use_llm = st.checkbox("Use OpenAI to structure the scheme (recommended for messy PDFs)", value=False)

    if st.button("Parse scheme", type="primary"):
        if uploaded is None and not pasted.strip():
            st.error("Upload a file or paste text.")
            return
        text = pasted.strip()
        if uploaded is not None:
            try:
                extracted = extract_scheme_text(uploaded.name, uploaded.getvalue())
            except Exception as exc:
                st.error(str(exc))
                return
            text = extracted if extracted.strip() else pasted
        st.session_state.scheme_text = text
        questions = parse_scheme_text(text)
        key = resolve_openai_key(openai_key)
        if use_llm and key:
            try:
                rows = openai_extract_scheme(text, key, model)
                if rows:
                    questions = questions_from_openai_dicts(rows)
            except Exception as exc:
                st.warning(f"OpenAI parse failed; using local parser. ({exc})")
        elif use_llm and not key:
            st.warning("No OpenAI key — used the local parser.")
        st.session_state.questions = questions
        st.session_state.variants = []
        st.success(f"Recognised {len(questions)} question(s). Edit them on the next step.")
        go(2)

    if st.session_state.scheme_text:
        with st.expander("Raw scheme text"):
            st.text(st.session_state.scheme_text[:8000])


def page_review_scheme() -> None:
    st.header("3. Review questions and suggested answers")
    st.write("Correct any OCR or parsing mistakes. Default mark is 1 unless you change it.")
    if not st.session_state.questions:
        st.info("Parse a marking scheme first.")
        return
    edited = st.data_editor(
        questions_to_frame(st.session_state.questions),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("Question id", required=True),
            "prompt": st.column_config.TextColumn("Question", width="large"),
            "suggested_answer": st.column_config.TextColumn("Suggested answer", width="large"),
            "marks": st.column_config.NumberColumn("Marks", min_value=0.0, step=0.5),
            "question_type": st.column_config.SelectboxColumn(
                "Type", options=["mcq", "short", "long", "calculation"]
            ),
        },
        hide_index=True,
        key="scheme_editor",
    )
    if st.button("Save scheme and split student answers", type="primary"):
        questions = frame_to_questions(edited)
        st.session_state.questions = questions
        for student in st.session_state.students:
            attach_answers(student, questions)
        apply_marking(st.session_state.students, questions, st.session_state.variants)
        st.session_state.marked = True
        st.success("Student answers were aligned to the scheme and given tentative marks.")
        go(3)


def page_moderate() -> None:
    st.header("4. Moderate answers (Google Form-style review)")
    students: list[StudentScript] = st.session_state.students
    questions: list[Question] = st.session_state.questions
    if not students or not questions:
        st.info("Load scripts and a scheme first.")
        return

    stats = class_stats(students)
    ceiling = max_total(questions)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", stats["n"])
    c2.metric("Average", f"{stats['average']} / {ceiling:g}")
    c3.metric("Median", stats["median"])
    c4.metric("Min / Max", f"{stats['min']} / {stats['max']}")
    c5.metric("Stdev", stats["stdev"])

    tab_q, tab_s = st.tabs(["By question", "By student"])

    with tab_q:
        qids = [q.id for q in questions]
        pick = st.selectbox("Question", qids, format_func=lambda i: f"Q{i}")
        q = next(x for x in questions if x.id == pick)
        st.markdown(f"**Prompt:** {q.prompt or '_missing_'}")
        st.markdown(f"**Official answer:** {q.suggested_answer or '_missing_'} — **{q.marks:g}** mark(s)")
        clusters = cluster_answers(students, pick)
        chart = pd.DataFrame(
            {"answer": [c["label"][:80] for c in clusters], "students": [c["count"] for c in clusters]}
        )
        st.bar_chart(chart, x="answer", y="students")

        st.subheader("Accept alternative answers")
        st.caption("For longer items, tick a cluster and set how many marks that wording should earn.")
        existing = [v for v in st.session_state.variants if v.question_id == pick]
        new_variants: list[AcceptedVariant] = [v for v in st.session_state.variants if v.question_id != pick]
        for i, cluster in enumerate(clusters):
            cols = st.columns([4, 1, 1])
            with cols[0]:
                st.write(f"{cluster['count']}×  {cluster['label'][:400]}")
            matched = next((v for v in existing if v.text == cluster["canonical"]), None)
            accept = cols[1].checkbox(
                "Accept",
                value=bool(matched and matched.enabled),
                key=f"acc-{pick}-{i}",
            )
            marks = cols[2].number_input(
                "Marks",
                min_value=0.0,
                max_value=float(q.marks),
                value=float(matched.marks if matched else q.marks),
                step=0.5,
                key=f"mk-{pick}-{i}",
            )
            if accept and cluster["canonical"]:
                new_variants.append(
                    AcceptedVariant(question_id=pick, text=cluster["canonical"], marks=marks, enabled=True)
                )
        extra = st.text_area("Add another accepted wording", key=f"extra-{pick}")
        extra_marks = st.number_input(
            "Marks for extra wording", min_value=0.0, max_value=float(q.marks), value=float(q.marks), key=f"exm-{pick}"
        )
        if extra.strip():
            new_variants.append(
                AcceptedVariant(question_id=pick, text=extra.strip(), marks=extra_marks, enabled=True)
            )
        if st.button("Apply accepted answers and re-mark", type="primary"):
            st.session_state.variants = new_variants
            apply_marking(students, questions, st.session_state.variants)
            st.success("Marks updated.")
            st.rerun()

    with tab_s:
        student = st.selectbox(
            "Student",
            students,
            format_func=lambda s: f"{s.student_id}  ({s.total:g}/{ceiling:g})",
        )
        st.write(f"File: `{student.filename}`")
        self_col, _ = st.columns(2)
        self_val = student.self_rated if student.self_rated is not None else 0.0
        new_self = self_col.number_input(
            "Self-rated score (0 if none)",
            min_value=0.0,
            value=float(self_val),
            step=0.5,
        )
        student.self_rated = None if new_self == 0 and student.self_rated is None else new_self
        if student.self_rated is not None:
            st.info(f"Discrepancy (self − app): {student.self_rated - student.total:+.2f}")

        for q in questions:
            ans = student.answers[q.id]
            with st.expander(f"Q{q.id} — {ans.awarded:g}/{q.marks:g}  ({ans.match_reason})"):
                ans.raw_text = st.text_area(
                    "Recognised answer",
                    value=ans.raw_text,
                    key=f"ans-{student.student_id}-{q.id}",
                    height=100,
                )
                ans.awarded = st.number_input(
                    "Awarded marks",
                    min_value=0.0,
                    max_value=float(q.marks),
                    value=float(ans.awarded),
                    step=0.5,
                    key=f"aw-{student.student_id}-{q.id}",
                )

        if st.button("Re-run automatic marking (keeps only un-edited structure)"):
            apply_marking(students, questions, st.session_state.variants)
            st.rerun()

    if st.button("Finalise and open reports"):
        go(4)


def page_reports() -> None:
    st.header("5. Final statistical reports")
    students: list[StudentScript] = st.session_state.students
    questions: list[Question] = st.session_state.questions
    if not students or not questions:
        st.info("Nothing to report yet.")
        return

    ind = individual_report(students, questions)
    byq = question_report(students, questions)
    overall = overall_report(students, questions)

    t1, t2, t3 = st.tabs(["Individual", "By question", "Overall"])
    with t1:
        st.markdown(ind)
        st.download_button("Download individual report", ind, file_name="individual_report.md")
    with t2:
        st.markdown(byq)
        st.download_button("Download question report", byq, file_name="question_report.md")
    with t3:
        st.markdown(overall)
        st.download_button("Download overall report", overall, file_name="overall_report.md")
        rows = []
        for s in students:
            row = {
                "class": s.class_name,
                "number": s.class_number,
                "total": s.total,
                "self_rated": s.self_rated,
                "discrepancy": (None if s.self_rated is None else round(s.self_rated - s.total, 2)),
            }
            for q in questions:
                ans = s.answers.get(q.id)
                row[f"Q{q.id}"] = ans.awarded if ans else 0
                row[f"Q{q.id}_text"] = ans.raw_text if ans else ""
            rows.append(row)
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("Download marks CSV", csv, file_name="marks.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="MARC — Moderated Assignment Review Companion", layout="wide")
    init_state()
    st.title("MARC")
    st.caption(
        "Moderated Assignment Review Companion — upload scripts and a scheme, "
        "correct recognition, accept alternative answers, then export reports."
    )

    engine, openai_key, model = sidebar_settings()
    step = st.sidebar.radio("Workflow", STEPS, index=st.session_state.step)
    st.session_state.step = STEPS.index(step)

    pages = [page_students, page_scheme, page_review_scheme, page_moderate, page_reports]
    if st.session_state.step == 0:
        pages[0](engine, openai_key, model)
    elif st.session_state.step == 1:
        pages[1](engine, openai_key, model)
    else:
        pages[st.session_state.step]()


if __name__ == "__main__":
    main()
