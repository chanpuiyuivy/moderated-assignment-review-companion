from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from marc.filename import extract_identity_from_text, filename_has_class_info, parse_student_filename
from marc.marking import apply_marking, cluster_answers, is_accepted_answer, score_answer, similarity
from marc.models import AcceptedVariant, Question, StudentScript
from marc.ocr import openai_extract_scheme, resolve_openai_key, transcribe_scheme_pdf, transcribe_student_pdf
from marc.parse_scheme import FILL_IN, finalize_questions, parse_scheme_text, questions_from_openai_dicts
from marc.parse_student import attach_answers
from marc.pdf_io import extract_scheme_text
from marc.reports import (
    class_stats,
    individual_report,
    max_total,
    overall_report,
    question_report,
    student_sort_key,
)

STEPS = [
    "Upload scripts",
    "Marking scheme",
    "Moderate answers",
    "Final reports",
]


def init_state() -> None:
    defaults = {
        "step": 0,
        "students": [],
        "questions": [],
        "variants": [],
        "scheme_text": "",
        "scheme_filename": "",
        "scheme_file_id": "",
        "scheme_editor_epoch": 0,
        "processed": False,
        "ocr_warning": "",
        "report_student_idx": 0,
        "student_info_epoch": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go(step: int) -> None:
    st.session_state.step = max(0, min(step, len(STEPS) - 1))
    st.rerun()


def show_step_banner() -> None:
    step = st.session_state.step
    st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)} · {STEPS[step]}")


def nav_row(*, back: int | None, next_step: int | None, next_label: str, next_disabled: bool = False) -> None:
    left, right, _ = st.columns([1, 2, 4])
    if back is not None and left.button("Back", key="nav_back"):
        go(back)
    if next_step is not None and right.button(
        next_label, type="primary", disabled=next_disabled, key="nav_next"
    ):
        go(next_step)


def questions_to_frame(questions: list[Question]) -> pd.DataFrame:
    if not questions:
        return pd.DataFrame(columns=["id", "prompt", "suggested_answer", "marks", "question_type"])
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
    return finalize_questions(rows)


def sidebar_settings() -> tuple[str, str, str]:
    st.sidebar.markdown("**M.A.R.C.**")
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
    st.header("Upload student assignments")
    st.write("Upload one PDF per student. Then read the scripts before going to the marking scheme.")
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
        warnings: list[str] = []
        progress = st.progress(0.0, text="Starting…")
        for i, f in enumerate(files):
            cls, num = parse_student_filename(f.name)
            data = f.getvalue()
            progress.progress((i + 0.3) / len(files), text=f"Reading {f.name}")
            try:
                text, warn, handwritten, ident = transcribe_student_pdf(data, engine, key, model)
            except Exception as exc:
                st.warning(f"{f.name}: could not read this PDF ({exc}).")
                text, warn, handwritten, ident = "", str(exc), [], ("", "", "")
            ocr_name, ocr_cls, ocr_num = ident
            extra_name, extra_cls, extra_num = extract_identity_from_text(text)
            ocr_name = ocr_name or extra_name
            ocr_cls = ocr_cls or extra_cls
            ocr_num = ocr_num or extra_num
            from_file = filename_has_class_info(f.name)
            if not from_file:
                if ocr_cls:
                    cls = ocr_cls
                if ocr_num:
                    num = ocr_num
            elif cls == "UNKNOWN" and ocr_cls:
                cls = ocr_cls
            if not num and ocr_num:
                num = ocr_num
            if warn:
                warnings.append(f"{f.name}: {warn}")
            students.append(
                StudentScript(
                    filename=f.name,
                    class_name=cls,
                    class_number=num,
                    student_name=ocr_name,
                    ocr_text=text,
                    pdf_bytes=data,
                    handwritten_answers=handwritten,
                )
            )
            progress.progress((i + 1) / len(files), text=f"Finished {f.name}")
        st.session_state.students = students
        st.session_state.processed = True
        st.session_state.student_info_epoch += 1
        st.session_state.ocr_warning = "\n".join(warnings)
        st.success(f"Loaded {len(students)} script(s).")

    if st.session_state.ocr_warning:
        st.info(st.session_state.ocr_warning)

    if st.session_state.students:
        st.subheader("Student info")
        st.caption("Name, class, and class number are taken from the file name when present, otherwise from the script after OCR. You can edit the table.")
        info_df = pd.DataFrame(
            [
                {
                    "file": s.filename,
                    "name": s.student_name,
                    "class": "" if s.class_name == "UNKNOWN" else s.class_name,
                    "class no.": s.class_number,
                }
                for s in st.session_state.students
            ]
        )
        edited = st.data_editor(
            info_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["file"],
            key=f"student_info_{st.session_state.student_info_epoch}",
        )
        for student, row in zip(st.session_state.students, edited.to_dict("records")):
            student.student_name = str(row.get("name") or "").strip()
            cls_val = str(row.get("class") or "").strip()
            student.class_name = cls_val or "UNKNOWN"
            student.class_number = str(row.get("class no.") or "").strip()

    nav_row(
        back=None,
        next_step=1,
        next_label="Next: marking scheme",
        next_disabled=not st.session_state.students,
    )


def _parse_scheme_bytes(filename: str, data: bytes, pasted: str, use_llm: bool, openai_key: str, model: str) -> bool:
    try:
        if filename.lower().endswith(".docx"):
            extracted = extract_scheme_text(filename, data)
        elif filename.lower().endswith(".doc"):
            st.error("Please save .doc files as .docx or PDF before uploading.")
            return False
        else:
            bar = st.progress(0.0, text="Recognising marking scheme…")
            extracted, _ = transcribe_scheme_pdf(
                data,
                on_progress=lambda done, total, msg: bar.progress(
                    min(done / max(total, 1), 1.0), text=msg
                ),
            )
            if not extracted.strip():
                extracted = extract_scheme_text(filename, data)
            bar.empty()
    except Exception as exc:
        st.error(str(exc))
        return False
    text = extracted if extracted.strip() else pasted
    _finish_parse(text, use_llm, openai_key, model, filename)
    return True


def _finish_parse(text: str, use_llm: bool, openai_key: str, model: str, filename: str) -> None:
    st.session_state.scheme_text = text
    st.session_state.scheme_filename = filename
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
    st.session_state.questions = finalize_questions(questions)
    st.session_state.variants = []
    st.session_state.scheme_editor_epoch += 1


def page_scheme(engine: str, openai_key: str, model: str) -> None:
    st.header("Marking scheme")
    st.write("Upload or re-upload a scheme, then add, delete, or edit questions in the table.")
    uploaded = st.file_uploader("Marking scheme (PDF or Word)", type=["pdf", "docx"], key="scheme_file")
    pasted = st.text_area("Or paste the scheme text", height=120, key="scheme_paste")
    use_llm = st.checkbox("Use OpenAI to structure the scheme", value=False)

    file_token = ""
    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", uploaded.name)
        file_token = f"{uploaded.name}:{len(uploaded.getvalue())}:{file_id}"
        if file_token != st.session_state.scheme_file_id:
            if _parse_scheme_bytes(uploaded.name, uploaded.getvalue(), pasted, use_llm, openai_key, model):
                st.session_state.scheme_file_id = file_token
                st.success(f"Parsed **{uploaded.name}**.")

    parse_cols = st.columns([2, 4])
    if parse_cols[0].button("Parse pasted text", disabled=not pasted.strip()):
        _finish_parse(pasted, use_llm, openai_key, model, "pasted text")
        st.session_state.scheme_file_id = ""
        st.success("Parsed pasted text.")

    if st.session_state.scheme_filename:
        st.info(f"Current file: **{st.session_state.scheme_filename}**. Upload another file to replace it.")

    if st.session_state.scheme_text:
        with st.expander("Recognised scheme text"):
            st.text(st.session_state.scheme_text[:8000])

    scheme_loaded = bool(st.session_state.scheme_filename or st.session_state.scheme_text)
    if not scheme_loaded:
        nav_row(back=0, next_step=2, next_label="Next: moderate answers", next_disabled=True)
        return

    st.subheader("Questions")
    add_cols = st.columns([1, 4])
    if add_cols[0].button("Add question"):
        used = {q.id for q in st.session_state.questions}
        nxt = 1
        while str(nxt) in used:
            nxt += 1
        st.session_state.questions.append(
            Question(id=str(nxt), prompt=FILL_IN, suggested_answer="", marks=1.0, question_type="short")
        )
        st.session_state.scheme_editor_epoch += 1
        st.rerun()

    epoch = st.session_state.scheme_editor_epoch
    hdr = st.columns([0.9, 2.6, 2.6, 0.8, 1.2, 0.45])
    hdr[0].markdown("**ID**")
    hdr[1].markdown("**Question**")
    hdr[2].markdown("**Suggested answer**")
    hdr[3].markdown("**Marks**")
    hdr[4].markdown("**Type**")
    hdr[5].markdown("")

    built: list[Question] = []
    delete_at: int | None = None
    for i, q in enumerate(st.session_state.questions):
        cols = st.columns([0.9, 2.6, 2.6, 0.8, 1.2, 0.45])
        qid = cols[0].text_input("ID", value=q.id, key=f"sch_id_{epoch}_{i}", label_visibility="collapsed")
        prompt = cols[1].text_input("Question", value=q.prompt, key=f"sch_pr_{epoch}_{i}", label_visibility="collapsed")
        answer = cols[2].text_input(
            "Suggested answer", value=q.suggested_answer, key=f"sch_ans_{epoch}_{i}", label_visibility="collapsed"
        )
        marks = cols[3].number_input(
            "Marks",
            min_value=0.0,
            value=float(q.marks),
            step=0.5,
            key=f"sch_mk_{epoch}_{i}",
            label_visibility="collapsed",
        )
        qtype = cols[4].selectbox(
            "Type",
            options=["mcq", "short", "long", "calculation"],
            index=["mcq", "short", "long", "calculation"].index(q.question_type)
            if q.question_type in {"mcq", "short", "long", "calculation"}
            else 1,
            key=f"sch_ty_{epoch}_{i}",
            label_visibility="collapsed",
        )
        if cols[5].button("❌", key=f"sch_del_{epoch}_{i}", help="Delete this question"):
            delete_at = i
        built.append(
            Question(
                id=str(qid).strip() or q.id,
                prompt=prompt,
                suggested_answer=answer,
                marks=float(marks),
                question_type=str(qtype),
            )
        )
    if delete_at is not None:
        built.pop(delete_at)
        st.session_state.questions = built
        st.session_state.scheme_editor_epoch += 1
        st.rerun()

    left, right, _ = st.columns([1, 2, 4])
    if left.button("Back", key="nav_back"):
        st.session_state.questions = built
        go(0)
    if right.button("Next: moderate answers", type="primary", disabled=not built, key="nav_next"):
        questions = finalize_questions(built)
        st.session_state.questions = questions
        for student in st.session_state.students:
            attach_answers(student, questions)
        apply_marking(st.session_state.students, questions, st.session_state.variants)
        go(2)


def _variant_sig(variants: list[AcceptedVariant]) -> str:
    payload = "|".join(f"{v.question_id}::{v.text}::{v.marks:.2f}" for v in variants)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _pdf_links(scripts: list[StudentScript], key_prefix: str) -> None:
    with_pdf = [s for s in scripts if s.pdf_bytes]
    if not with_pdf:
        return
    cols = st.columns(len(with_pdf))
    for i, student in enumerate(with_pdf):
        with cols[i]:
            _open_pdf_in_tab(student.student_id, student.pdf_bytes, f"{key_prefix}_{i}_{student.filename}")


def _open_pdf_in_tab(label: str, data: bytes, element_id: str) -> None:
    import base64

    safe_id = "pdf" + hashlib.md5(element_id.encode()).hexdigest()[:12]
    b64 = base64.b64encode(data).decode("ascii")
    components.html(
        f"""
        <a id="{safe_id}" target="_blank" rel="noopener noreferrer"
           style="display:inline-block;padding:4px 10px;border:1px solid #d0d5dd;border-radius:8px;
                  background:#f0f2f6;text-decoration:none;color:#111;font-family:sans-serif;
                  font-size:0.85rem;white-space:nowrap;">{label}</a>
        <script>
        (function() {{
          const raw = atob("{b64}");
          const arr = new Uint8Array(raw.length);
          for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
          const url = URL.createObjectURL(new Blob([arr], {{type: "application/pdf"}}));
          const a = document.getElementById("{safe_id}");
          if (a) {{ a.href = url; }}
        }})();
        </script>
        """,
        height=40,
    )


def page_moderate() -> None:
    st.header("Moderate answers")
    students: list[StudentScript] = st.session_state.students
    questions: list[Question] = st.session_state.questions
    if not students or not questions:
        st.info("Load scripts and a scheme first.")
        nav_row(back=1, next_step=None, next_label="View final report", next_disabled=True)
        return

    metrics_box = st.container()
    ceiling = max_total(questions)
    apply_marking(students, questions, st.session_state.variants)

    tab_q, tab_s = st.tabs(["By question", "By student"])
    collected: list[AcceptedVariant] = []
    wrong_ids: set[str] = set()

    with tab_q:
        st.caption(
            "Only questions with incorrect answers are shown. Tick **Accept** to count that wording. "
            "Open a student PDF in a new tab to check the script."
        )
        any_wrong = False
        for q in questions:
            clusters = cluster_answers(students, q.id)
            existing = [v for v in st.session_state.variants if v.question_id == q.id]
            alternatives = [
                c
                for c in clusters
                if c["canonical"]
                and similarity(c["canonical"], q.suggested_answer) < 0.82
            ]
            wrong_students = [
                s
                for s in students
                if s.answers.get(q.id) is not None and s.answers[q.id].awarded < q.marks
            ]
            if not alternatives or not wrong_students:
                continue
            any_wrong = True
            wrong_ids.add(q.id)
            with st.container(border=True):
                st.markdown(f"**Question {q.id}** · {q.marks:g} mark(s) · {q.question_type}")
                st.markdown(q.prompt.strip() or FILL_IN)
                st.success(f"Official answer: {q.suggested_answer or '(none)'}")
                for i, cluster in enumerate(alternatives):
                    pdf_scripts = [s for s in cluster["students"] if s.pdf_bytes][:6]
                    n_pdf = max(len(pdf_scripts), 1)
                    weights = [4.2, 0.9, 0.9] + [0.9] * n_pdf
                    cols = st.columns(weights)
                    cols[0].markdown(f"{cluster['count']}× **{cluster['label'][:280]}**")
                    matched = next((v for v in existing if v.text == cluster["canonical"]), None)
                    accept = cols[1].checkbox(
                        "Accept",
                        value=bool(matched and matched.enabled),
                        key=f"acc_{q.id}_{i}",
                    )
                    marks = cols[2].number_input(
                        "Marks",
                        min_value=0.0,
                        max_value=float(q.marks),
                        value=float(matched.marks if matched else q.marks),
                        step=0.5,
                        key=f"mk_{q.id}_{i}",
                        label_visibility="collapsed",
                    )
                    for pi, student in enumerate(pdf_scripts):
                        with cols[3 + pi]:
                            _open_pdf_in_tab(
                                student.student_id,
                                student.pdf_bytes,
                                f"pdf_{q.id}_{i}_{pi}_{student.filename}",
                            )
                    if accept and cluster["canonical"]:
                        collected.append(
                            AcceptedVariant(
                                question_id=q.id, text=cluster["canonical"], marks=marks, enabled=True
                            )
                        )
        if not any_wrong:
            st.success("No incorrect answers to moderate.")

    kept = [v for v in st.session_state.variants if v.question_id not in wrong_ids]
    collected = kept + collected

    st.session_state.variants = collected
    apply_marking(students, questions, collected)
    vkey = _variant_sig(collected)

    with tab_s:
        student = st.selectbox(
            "Student",
            students,
            format_func=lambda s: f"{s.display_label}  ({s.total:g}/{ceiling:g})",
            key="moderate_student",
        )
        meta1, meta2 = st.columns(2)
        meta1.write(f"File: `{student.filename}`")
        if student.pdf_bytes:
            with meta2:
                _open_pdf_in_tab("Open this student’s PDF", student.pdf_bytes, f"stu_pdf_{student.filename}")
        self_val = student.self_rated if student.self_rated is not None else 0.0
        new_self = st.number_input(
            "Self-rated score (0 if none)",
            min_value=0.0,
            value=float(self_val),
            step=0.5,
            key=f"self_{student.filename}",
        )
        student.self_rated = None if new_self == 0 and student.self_rated is None else new_self
        if student.self_rated is not None:
            st.info(f"Discrepancy (self − app): {student.self_rated - student.total:+.2f}")

        for idx, q in enumerate(questions):
            ans = student.answers.get(q.id)
            if ans is None:
                continue
            with st.container(border=True):
                st.markdown(f"**Q{q.id}** · {ans.match_reason}")
                new_text = st.text_area(
                    "Recognised answer",
                    value=ans.raw_text,
                    key=f"ans_{student.filename}_{idx}_{q.id}",
                    height=80,
                )
                ans.raw_text = new_text
                awarded, reason, variant = score_answer(ans.raw_text, q, [v for v in collected if v.question_id == q.id])
                ans.match_reason = reason
                ans.accepted_variant = variant
                ans.awarded = st.number_input(
                    "Awarded marks",
                    min_value=0.0,
                    max_value=float(q.marks),
                    value=float(awarded),
                    step=0.5,
                    key=f"aw_{student.filename}_{idx}_{q.id}_{vkey}_{hashlib.md5(new_text.encode('utf-8', errors='ignore')).hexdigest()[:6]}",
                )

    stats = class_stats(students)
    with metrics_box:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Students", stats["n"])
        c2.metric("Average", f"{stats['average']} / {ceiling:g}")
        c3.metric("Median", stats["median"])
        c4.metric("Min / Max", f"{stats['min']} / {stats['max']}")
        c5.metric("Stdev", stats["stdev"])

    st.divider()
    left, right, _ = st.columns([1, 2, 4])
    if left.button("Back", key="nav_back"):
        go(1)
    if right.button("View final report", type="primary", key="nav_next"):
        go(3)


def _render_individual(students: list[StudentScript], questions: list[Question]) -> None:
    ceiling = max_total(questions)
    ordered = sorted(students, key=student_sort_key)
    labels = [s.display_label for s in ordered]
    current_label = st.session_state.get("report_student_select", labels[0])
    if current_label not in labels:
        current_label = labels[0]
    pick = st.selectbox("Student", labels, index=labels.index(current_label), key="report_student_select")
    current = next(s for s in ordered if s.display_label == pick)
    pos = labels.index(pick)

    prev_c, next_c, _ = st.columns([1, 1, 4])
    if prev_c.button("Previous student", disabled=pos == 0):
        st.session_state.report_student_select = labels[pos - 1]
        st.rerun()
    if next_c.button("Next student", disabled=pos == len(labels) - 1):
        st.session_state.report_student_select = labels[pos + 1]
        st.rerun()

    st.subheader(current.display_label)
    st.caption(current.filename)
    st.write(f"**Score:** {current.total:g} / {ceiling:g}")
    if current.self_rated is not None:
        delta = round(current.self_rated - current.total, 2)
        st.write(f"**Self-rated:** {current.self_rated:g} (discrepancy {delta:+g})")
    else:
        st.write("**Self-rated:** not found")

    rows = []
    correct_flags: list[bool] = []
    variants = st.session_state.variants
    for q in questions:
        ans = current.answers.get(q.id)
        got = ans.raw_text if ans else ""
        awarded = ans.awarded if ans else 0.0
        correct = awarded >= q.marks or is_accepted_answer(got, q, variants)
        correct_flags.append(bool(got.strip()) and correct)
        if not got.strip():
            correct_flags[-1] = False
        rows.append(
            {
                "ID": q.id,
                "Question": q.prompt,
                "Student answer": got,
                "Expected": q.suggested_answer,
                "Mark": awarded,
                "Out of": q.marks,
            }
        )
    df = pd.DataFrame(rows)

    def _colour_answers(_df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=_df.index, columns=_df.columns)
        for i, ok in enumerate(correct_flags):
            colour = "background-color: #c8e6c9" if ok else "background-color: #ef9a9a"
            styles.at[i, "Student answer"] = colour
            styles.at[i, "Mark"] = colour
        return styles

    st.caption("Green = correct. Red = incorrect.")
    st.dataframe(df.style.apply(_colour_answers, axis=None), use_container_width=True, hide_index=True)


def _short_bar_chart(clusters: list[dict], question: Question, variants: list[AcceptedVariant]):
    rows = []
    for c in clusters:
        label = c["label"][:60] or "(blank)"
        correct = is_accepted_answer(c["canonical"] or c["label"], question, variants) and c["label"] != "(blank)"
        if c["label"] != "(blank)" and is_accepted_answer(c["label"], question, variants):
            correct = True
        rows.append({"answer": label, "students": c["count"], "kind": "Correct" if correct else "Other"})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("answer:N", sort="-y", title="Answer"),
            y=alt.Y("students:Q", title="Number of students"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["Correct", "Other"], range=["#2e7d32", "#90a4ae"]),
                legend=alt.Legend(title=""),
            ),
            tooltip=["answer", "students", "kind"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_by_question(students: list[StudentScript], questions: list[Question]) -> None:
    variants = st.session_state.variants
    for q in questions:
        with st.container(border=True):
            st.markdown(f"### Question {q.id} ({q.question_type}, {q.marks:g} mark(s))")
            st.write(q.prompt or FILL_IN)
            st.success(f"Suggested answer: {q.suggested_answer or '(none)'}")
            clusters = cluster_answers(students, q.id)
            if q.question_type == "long":
                for c in clusters:
                    correct = is_accepted_answer(c["canonical"] or c["label"], q, variants) and c["label"] != "(blank)"
                    left, right = st.columns([6, 1])
                    text = c["label"][:500]
                    if correct:
                        left.markdown(
                            f"<div style='background:#e8f5e9;padding:8px;border-radius:6px'>{text}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        left.write(text)
                    right.markdown(f"**{c['count']}**")
            else:
                _short_bar_chart(clusters, q, variants)


def _render_overall(students: list[StudentScript], questions: list[Question]) -> None:
    stats = class_stats(students)
    ceiling = max_total(questions) or 1.0
    summary = pd.DataFrame(
        [
            {"Statistic": "Students", "Value": stats["n"]},
            {"Statistic": "Maximum available", "Value": ceiling},
            {"Statistic": "Average", "Value": stats["average"]},
            {"Statistic": "Median", "Value": stats["median"]},
            {"Statistic": "Maximum", "Value": stats["max"]},
            {"Statistic": "Minimum", "Value": stats["min"]},
            {"Statistic": "Standard deviation", "Value": stats["stdev"]},
        ]
    )
    st.subheader("Class statistics")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    hist_df = pd.DataFrame({"score": [s.total for s in students]})
    hist = (
        alt.Chart(hist_df)
        .mark_bar(color="#1565c0")
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=12), title="Score"),
            y=alt.Y("count()", title="Number of students"),
            tooltip=[alt.Tooltip("count()", title="Students")],
        )
        .properties(height=280, title="Score distribution")
    )
    st.altair_chart(hist, use_container_width=True)

    rows = []
    for s in students:
        pct = round(100 * s.total / ceiling, 1) if ceiling else 0
        rows.append(
            {
                "Class": s.class_name,
                "Class number": int(s.class_number) if str(s.class_number).isdigit() else s.class_number,
                "Student": s.display_label,
                "Score": s.total,
                "Percent": pct,
                "Self-rated": s.self_rated,
            }
        )
    df = pd.DataFrame(rows)

    styler = df.style.apply(
        lambda col: [
            "background-color: #ffc0cb" if df.iloc[i]["Percent"] < 50 else ""
            for i in range(len(df))
        ]
        if col.name == "Score"
        else [""] * len(df),
        axis=0,
    )
    st.subheader("Score list")
    st.caption("Scores below 50% are highlighted in pink. Use the column menu (⋮) to sort.")
    st.dataframe(styler, use_container_width=True, hide_index=True)


def page_reports() -> None:
    st.header("Final statistical reports")
    students: list[StudentScript] = st.session_state.students
    questions: list[Question] = st.session_state.questions
    if not students or not questions:
        st.info("Nothing to report yet.")
        nav_row(back=2, next_step=None, next_label="Next", next_disabled=True)
        return

    t1, t2, t3 = st.tabs(["Individual", "By question", "Overall"])
    with t1:
        _render_individual(students, questions)
        st.download_button(
            "Download all individual reports",
            individual_report(students, questions),
            file_name="individual_report.md",
        )
    with t2:
        _render_by_question(students, questions)
        st.download_button(
            "Download question report",
            question_report(students, questions),
            file_name="question_report.md",
        )
    with t3:
        _render_overall(students, questions)
        st.download_button(
            "Download overall report",
            overall_report(students, questions),
            file_name="overall_report.md",
        )
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

    nav_row(back=2, next_step=None, next_label="Next", next_disabled=True)


def main() -> None:
    st.set_page_config(page_title="M.A.R.C. — Moderated Assignment Review Companion", layout="wide")
    init_state()
    st.title("M.A.R.C.")
    st.caption(
        "Moderated Assignment Review Companion — upload scripts and a scheme, "
        "correct recognition, accept alternative answers, then export reports."
    )
    show_step_banner()

    engine, openai_key, model = sidebar_settings()
    step = st.session_state.step
    if step == 0:
        page_students(engine, openai_key, model)
    elif step == 1:
        page_scheme(engine, openai_key, model)
    elif step == 2:
        page_moderate()
    else:
        page_reports()


if __name__ == "__main__":
    main()
