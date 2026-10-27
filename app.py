
import os
import io
import re
import json
import time

import streamlit as st
import fitz
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Interview Studio",
    page_icon="🎯",
    layout="wide"
)

# Read the API key locally or from Streamlit Cloud Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY", "")

MODEL_FALLBACKS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
]

# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #0b1020 0%, #17152d 100%
        );
        color: #f4f4f8;
    }

    [data-testid="stSidebar"] {
        background-color: #0d1426;
        border-right: 1px solid #29324a;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    .stButton > button {
        background: linear-gradient(
            90deg, #7c3aed, #a855f7
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg, #6d28d9, #9333ea
        );
        color: white;
        border: 1px solid #c4b5fd;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        background: #171e35;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #303a56;
    }

    .subtitle {
        color: #c4c7d4;
        font-size: 16px;
    }

    .small-note {
        color: #a6acc0;
        font-size: 13px;
    }

    hr {
        border-color: #30364b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. GEMINI CLIENT
# --------------------------------------------------

if not API_KEY:
    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY "
        "to your local .env file or Streamlit Cloud Secrets."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


def generate_with_fallback(prompt):
    """
    Try available models in order.
    Retry temporary 503 errors with increasing delays.
    Fall back to another model when necessary.
    """

    errors = []

    for model in MODEL_FALLBACKS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response.text and response.text.strip():
                    return response.text.strip(), model

                errors.append(
                    f"{model}: The model returned an empty response."
                )
                break

            except Exception as exc:
                error_text = str(exc)
                errors.append(f"{model}: {error_text[:250]}")

                is_temporary = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text.upper()
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text.upper()
                )

                if is_temporary and attempt == 0:
                    time.sleep(2)
                    continue

                # A missing/unsupported model should be skipped.
                # Other errors are also allowed to fall back.
                break

    raise RuntimeError(
        "All configured Gemini models failed.\n"
        + "\n".join(errors[-6:])
    )


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

defaults = {
    "history": [],
    "question": "",
    "evaluation": None,
    "answer": "",
    "current_role": "Java Developer",
    "current_level": "Beginner",
    "resume_text": "",
    "resume_name": "",
    "question_model": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 5. HEADER
# --------------------------------------------------

st.title("🎯 AI Interview Studio")

st.markdown(
    '<p class="subtitle">Practice interviews, receive '
    'AI-powered feedback, and prepare confidently for your '
    'next opportunity.</p>',
    unsafe_allow_html=True
)

st.caption(
    "Personalized questions • Intelligent evaluation • "
    "Actionable feedback"
)

st.divider()


# --------------------------------------------------
# 6. SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Interview Settings")

    roles = [
        "Java Developer",
        "Python Developer",
        "AI/ML Engineer",
        "Data Analyst",
        "Software Engineer",
    ]

    levels = [
        "Beginner",
        "Intermediate",
        "Advanced",
    ]

    selected_role = st.selectbox(
        "Select your role",
        roles,
        index=roles.index(st.session_state.current_role)
        if st.session_state.current_role in roles else 0
    )

    selected_level = st.selectbox(
        "Select difficulty",
        levels,
        index=levels.index(st.session_state.current_level)
        if st.session_state.current_level in levels else 0
    )

    st.session_state.current_role = selected_role
    st.session_state.current_level = selected_level

    st.divider()

    st.subheader("📄 Resume-Based Interview")

    uploaded_resume = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Text PDFs and scanned PDFs are supported."
    )

    if uploaded_resume is not None:
        if st.session_state.resume_name != uploaded_resume.name:
            st.session_state.resume_text = ""
            st.session_state.resume_name = ""

            try:
                with fitz.open(
                    stream=uploaded_resume.getvalue(),
                    filetype="pdf"
                ) as pdf:
                    extracted_text = "\n".join(
                        page.get_text()
                        for page in pdf
                    )

                # OCR is used only when normal text extraction fails.
                if not extracted_text.strip():
                    with st.spinner(
                        "Scanned PDF detected. Trying OCR..."
                    ):
                        try:
                            import easyocr
                            import numpy as np
                            from PIL import Image

                            reader = easyocr.Reader(
                                ["en"],
                                gpu=False
                            )

                            ocr_pages = []

                            with fitz.open(
                                stream=uploaded_resume.getvalue(),
                                filetype="pdf"
                            ) as pdf:
                                for page in pdf:
                                    pix = page.get_pixmap(
                                        matrix=fitz.Matrix(2, 2),
                                        alpha=False
                                    )

                                    image = Image.open(
                                        io.BytesIO(
                                            pix.tobytes("png")
                                        )
                                    )

                                    words = reader.readtext(
                                        np.array(image),
                                        detail=0,
                                        paragraph=True
                                    )

                                    ocr_pages.append(
                                        "\n".join(words)
                                    )

                            extracted_text = "\n".join(ocr_pages)

                        except Exception as ocr_error:
                            st.warning(
                                "OCR could not read this PDF. "
                                "Try a clearer scan or a text-based PDF."
                            )

                if extracted_text.strip():
                    st.session_state.resume_text = (
                        extracted_text[:12000]
                    )
                    st.session_state.resume_name = (
                        uploaded_resume.name
                    )

                    st.success(
                        "Resume text extracted successfully!"
                    )
                else:
                    st.warning(
                        "No readable text found in this PDF."
                    )

            except Exception as exc:
                st.error(
                    f"Could not read the PDF: {exc}"
                )

        if st.session_state.resume_text:
            st.caption(
                f"Resume loaded: {st.session_state.resume_name}"
            )
            st.caption(
                f"Extracted {len(st.session_state.resume_text)} "
                "characters."
            )
        else:
            st.info(
                "Resume uploaded, but readable text has not "
                "been extracted."
            )

        if st.button(
            "Remove Resume",
            use_container_width=True
        ):
            st.session_state.resume_text = ""
            st.session_state.resume_name = ""
            st.rerun()

    st.divider()

    if st.button(
        "🔄 Start New Interview",
        use_container_width=True
    ):
        st.session_state.question = ""
        st.session_state.evaluation = None
        st.session_state.answer = ""
        st.session_state.question_model = ""
        st.rerun()

    st.divider()

    st.caption(
        "AI Interview Studio | Powered by Streamlit and Gemini"
    )


# --------------------------------------------------
# 7. RESUME STATUS
# --------------------------------------------------

if st.session_state.resume_text:
    st.info(
        "Resume-based mode is enabled. Questions can be "
        "tailored to the skills and projects identified in your PDF."
    )
else:
    st.info(
        "General interview mode is enabled. Upload a readable "
        "resume to personalize your questions."
    )


# --------------------------------------------------
# 8. GENERATE INTERVIEW QUESTION
# --------------------------------------------------

st.subheader("💡 Interview Question")

if st.button(
    "✨ Generate Interview Question",
    use_container_width=False
):
    role = st.session_state.current_role
    level = st.session_state.current_level
    resume = st.session_state.resume_text

    if resume:
        resume_context = (
            "Use the candidate's resume to personalize the "
            "question. Focus on their skills, projects, tools, "
            "or experience. Do not invent details not present "
            "in the resume.\n\n"
            f"RESUME:\n{resume}"
        )
    else:
        resume_context = (
            "Generate a suitable general interview question "
            "for the selected role."
        )

    prompt = f"""
You are an experienced technical interviewer.

Candidate role: {role}
Difficulty: {level}

{resume_context}

Generate exactly ONE useful interview question.
The question should match the selected role and difficulty.
Avoid asking a question that is too broad.
Do not include the answer or evaluation rubric.
Return only the interview question.
"""

    with st.spinner(
        "Generating your personalized interview question..."
    ):
        try:
            question_text, used_model = generate_with_fallback(
                prompt
            )

            st.session_state.question = question_text
            st.session_state.question_model = used_model
            st.session_state.evaluation = None
            st.session_state.answer = ""

        except Exception as exc:
            st.error(
                "Gemini could not generate a question after "
                "trying the configured models."
            )
            st.code(str(exc))


if st.session_state.question:
    st.markdown("### Your Question")
    st.markdown(st.session_state.question)

    if st.session_state.question_model:
        st.caption(
            f"Generated using: {st.session_state.question_model}"
        )

    st.divider()

    # --------------------------------------------------
    # 9. ANSWER INPUT AND EVALUATION
    # --------------------------------------------------

    st.subheader("✍️ Submit Your Answer")

    answer = st.text_area(
        "Write your answer below",
        value=st.session_state.answer,
        height=180,
        placeholder=(
            "Explain your answer clearly. Include examples "
            "where appropriate..."
        ),
        key="answer_input"
    )

    if st.button("🧠 Evaluate My Answer"):
        st.session_state.answer = answer

        if not answer.strip():
            st.warning(
                "Please enter your answer before evaluating it."
            )

        else:
            evaluation_prompt = f"""
You are a fair and constructive technical interviewer.

Role: {st.session_state.current_role}
Difficulty: {st.session_state.current_level}

Interview question:
{st.session_state.question}

Candidate answer:
{answer}

Evaluate the candidate's answer based on technical accuracy,
relevance, clarity, and completeness.

Return ONLY valid JSON with these keys:
{{
  "score": 0,
  "verdict": "Needs Improvement",
  "feedback": "Constructive feedback",
  "strengths": ["strength 1", "strength 2"],
  "missing_concepts": ["concept 1", "concept 2"],
  "sample_answer": "A clear example of a good answer"
}}

Rules:
- score must be an integer from 0 to 100.
- verdict should reflect the quality of the answer.
- feedback should explain what was good and what to improve.
- strengths and missing_concepts must be JSON arrays of strings.
- sample_answer must answer the question accurately.
- Do not award a high score to an incorrect answer.
- Return JSON only, without Markdown fences.
"""

            with st.spinner(
                "Evaluating your answer..."
            ):
                try:
                    result_text, used_model = (
                        generate_with_fallback(
                            evaluation_prompt
                        )
                    )

                    # Extract JSON even if the model adds fences.
                    match = re.search(
                        r"\{.*\}",
                        result_text,
                        flags=re.DOTALL
                    )

                    if not match:
                        raise ValueError(
                            "The model did not return valid JSON."
                        )

                    evaluation = json.loads(match.group(0))

                    score = int(evaluation.get("score", 0))
                    score = max(0, min(100, score))

                    evaluation["score"] = score
                    evaluation["verdict"] = str(
                        evaluation.get(
                            "verdict",
                            "Evaluation completed"
                        )
                    )
                    evaluation["feedback"] = str(
                        evaluation.get("feedback", "")
                    )
                    evaluation["strengths"] = (
                        evaluation.get("strengths", [])
                    )
                    evaluation["missing_concepts"] = (
                        evaluation.get("missing_concepts", [])
                    )
                    evaluation["sample_answer"] = str(
                        evaluation.get("sample_answer", "")
                    )

                    st.session_state.evaluation = evaluation

                    st.session_state.history.append({
                        "role": st.session_state.current_role,
                        "level": st.session_state.current_level,
                        "question": st.session_state.question,
                        "answer": answer,
                        "score": score,
                        "verdict": evaluation["verdict"],
                    })

                    st.success(
                        f"Answer evaluated using {used_model}."
                    )

                except Exception as exc:
                    st.error(
                        "Could not evaluate your answer. "
                        "Please try again."
                    )
                    st.code(str(exc))


# --------------------------------------------------
# 10. DISPLAY EVALUATION
# --------------------------------------------------

evaluation = st.session_state.evaluation

if evaluation:
    st.divider()
    st.header("📊 Your Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Score",
            f"{evaluation['score']}/100"
        )

    with col2:
        st.metric(
            "Verdict",
            evaluation["verdict"]
        )

    st.subheader("📝 Feedback")
    st.write(evaluation["feedback"])

    st.subheader("✅ Strengths")

    strengths = evaluation.get("strengths", [])

    if strengths:
        for item in strengths:
            st.markdown(f"- {item}")
    else:
        st.write("No specific strengths were listed.")

    st.subheader("📌 Concepts to Improve")

    missing = evaluation.get("missing_concepts", [])

    if missing:
        for item in missing:
            st.markdown(f"- {item}")
    else:
        st.write(
            "No specific missing concepts were listed."
        )

    st.subheader("💡 Example of a Strong Answer")
    st.write(evaluation["sample_answer"])


# --------------------------------------------------
# 11. PERFORMANCE DASHBOARD
# --------------------------------------------------

st.divider()
st.header("📈 Performance Dashboard")

history = st.session_state.history

if not history:
    st.info(
        "Your performance statistics will appear here after "
        "you complete and evaluate your first answer."
    )

else:
    total = len(history)
    average = sum(
        item["score"] for item in history
    ) / total

    best = max(
        item["score"] for item in history
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Questions Evaluated", total)
    col2.metric("Average Score", f"{average:.1f}/100")
    col3.metric("Best Score", f"{best}/100")

    st.subheader("Interview History")

    for index, item in enumerate(
        reversed(history),
        start=1
    ):
        with st.expander(
            f"Question {total - index + 1} — "
            f"{item['score']}/100 — {item['role']}"
        ):
            st.write("**Difficulty:**", item["level"])
            st.write("**Question:**", item["question"])
            st.write("**Your answer:**", item["answer"])
            st.write("**Verdict:**", item["verdict"])

    report = json.dumps(
        history,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        "⬇️ Download Interview History (JSON)",
        data=report,
        file_name="interview_history.json",
        mime="application/json"
    )

    if st.button("🗑️ Clear Performance History"):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p class="small-note">
    AI Interview Studio | Practice, improve, and prepare
    confidently. Scores and feedback are AI-generated and
    should be treated as guidance rather than a definitive
    hiring decision.
    </p>
    """,
    unsafe_allow_html=True
)
