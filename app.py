
import os
import io
import re
import json
import time

import streamlit as st
import fitz
from dotenv import load_dotenv
from google import genai


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Student Dashboard | AI Interview Studio",
    page_icon="🎯",
    layout="wide"
)

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY", "")

MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
]


# ==================================================
# 2. DESIGN
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #0b1020, #17152d
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
            90deg, #7c3aed, #c026d3
        );
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
    }

    .stButton > button:hover {
        border: 1px solid #d8b4fe;
        color: white;
    }

    [data-testid="stMetric"] {
        background: #171e35;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #303a56;
    }

    .profile-card {
        background: #141d35;
        border: 1px solid #303a56;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 12px;
    }

    .muted {
        color: #b7bfd5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 3. GEMINI CLIENT
# ==================================================

if not API_KEY:
    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY "
        "to your .env file or Streamlit Cloud Secrets."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


def generate_with_fallback(prompt):
    """Try configured models and retry temporary failures."""

    errors = []

    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response.text and response.text.strip():
                    return response.text.strip(), model

                errors.append(f"{model}: Empty response.")
                break

            except Exception as exc:
                message = str(exc)
                errors.append(f"{model}: {message[:300]}")

                temporary = any(
                    term in message.upper()
                    for term in [
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED"
                    ]
                )

                if temporary and attempt == 0:
                    time.sleep(2)
                    continue

                break

    raise RuntimeError(
        "All configured models failed:\n"
        + "\n".join(errors[-6:])
    )


# ==================================================
# 4. SESSION STATE
# ==================================================

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
    "navigation": "🏠 Dashboard",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def open_interview():
    """Navigate to interview practice."""
    st.session_state.navigation = "💬 Interview Practice"


def new_interview():
    """Reset the current question and answer."""
    st.session_state.question = ""
    st.session_state.evaluation = None
    st.session_state.answer = ""


# ==================================================
# 5. SIDEBAR AND NAVIGATION
# ==================================================

with st.sidebar:
    st.title("🎯 AI Interview Studio")
    st.caption("Your personal interview preparation space")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💬 Interview Practice",
            "📊 Performance"
        ],
        key="navigation"
    )

    st.divider()

    st.subheader("⚙️ Interview Settings")

    roles = [
        "Java Developer",
        "Python Developer",
        "AI/ML Engineer",
        "Data Analyst",
        "Software Engineer"
    ]

    levels = [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]

    st.session_state.current_role = st.selectbox(
        "Select your role",
        roles,
        index=roles.index(st.session_state.current_role)
        if st.session_state.current_role in roles else 0
    )

    st.session_state.current_level = st.selectbox(
        "Select difficulty",
        levels,
        index=levels.index(st.session_state.current_level)
        if st.session_state.current_level in levels else 0
    )

    st.divider()

    st.subheader("📄 Your Resume")

    uploaded_resume = st.file_uploader(
        "Upload resume (PDF)",
        type=["pdf"]
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
                        p.get_text() for p in pdf
                    )

                # Use OCR for scanned PDFs.
                if not extracted_text.strip():
                    with st.spinner("Reading scanned PDF..."):
                        try:
                            import easyocr
                            import numpy as np
                            from PIL import Image

                            reader = easyocr.Reader(
                                ["en"],
                                gpu=False
                            )

                            pages_text = []

                            with fitz.open(
                                stream=uploaded_resume.getvalue(),
                                filetype="pdf"
                            ) as pdf:
                                for pdf_page in pdf:
                                    pix = pdf_page.get_pixmap(
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

                                    pages_text.append(
                                        "\n".join(words)
                                    )

                            extracted_text = "\n".join(
                                pages_text
                            )

                        except Exception as ocr_error:
                            st.warning(
                                "OCR failed. Try a clearer PDF "
                                "or a PDF containing selectable text."
                            )

                if extracted_text.strip():
                    st.session_state.resume_text = (
                        extracted_text[:12000]
                    )
                    st.session_state.resume_name = (
                        uploaded_resume.name
                    )
                    st.success("Resume text extracted!")
                else:
                    st.warning("No readable text found.")

            except Exception as exc:
                st.error(f"PDF error: {exc}")

    if st.session_state.resume_text:
        st.caption(
            f"Loaded: {st.session_state.resume_name}"
        )
        st.caption(
            f"{len(st.session_state.resume_text)} characters extracted"
        )

        if st.button("Remove Resume"):
            st.session_state.resume_text = ""
            st.session_state.resume_name = ""
            st.rerun()

    st.divider()

    st.button(
        "🔄 Start New Interview",
        on_click=new_interview,
        use_container_width=True
    )

    st.caption("Powered by Streamlit and Gemini")


# ==================================================
# 6. STUDENT DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.title("👋 Welcome, Haswika!")

    st.markdown(
        "Your personal space to practise interviews, "
        "improve your skills, and track your progress."
    )

    st.divider()

    st.subheader("👩‍🎓 Student Profile")

    profile_left, profile_right = st.columns([1.2, 1])

    with profile_left:
        st.markdown(
            """
            <div class="profile-card">
                <h2>Haswika Akula</h2>
                <p class="muted">B.E. CSE (AI-ML)</p>
                <p>Keshav Memorial Engineering College</p>
                <p>Academic Year: 2023–2027</p>
                <p>Career Goal: AI/ML Engineer</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with profile_right:
        st.markdown(
            """
            <div class="profile-card">
                <h3>Technical Skills</h3>
                <p>Python • Java • C • SQL</p>
                <p>Machine Learning • Streamlit</p>
                <p>Gemini API • GitHub</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Update the profile details and skills to match "
        "your current information."
    )

    st.divider()

    st.subheader("📈 Your Preparation Overview")

    history = st.session_state.history

    scores = [
        item["score"] for item in history
    ]

    average_score = (
        sum(scores) / len(scores)
        if scores else 0
    )

    best_score = max(scores, default=0)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Questions Practiced",
        len(history)
    )

    metric2.metric(
        "Average Score",
        f"{average_score:.1f}%"
    )

    metric3.metric(
        "Best Score",
        f"{best_score}%"
    )

    st.divider()

    st.subheader("✨ Your Interview Tools")

    card1, card2 = st.columns(2)

    with card1:
        st.markdown("### 📄 Resume-Based Questions")
        st.write(
            "Generate questions tailored to the skills "
            "and projects in your uploaded resume."
        )

    with card2:
        st.markdown("### 🧠 AI Answer Evaluation")
        st.write(
            "Receive scores, feedback, strengths, and "
            "suggestions for improvement."
        )

    card3, card4 = st.columns(2)

    with card3:
        st.markdown("### 📊 Performance Tracking")
        st.write(
            "Review your scores and track your practice history."
        )

    with card4:
        st.markdown("### 🎯 Role-Specific Practice")
        st.write(
            "Select your desired job role and difficulty level."
        )

    st.divider()

    st.subheader("🚀 Ready to Practise?")

    st.write(
        "Click below to enter interview practice and "
        "generate your first question."
    )

    st.button(
        "▶ Start Interview Practice",
        on_click=open_interview,
        use_container_width=True
    )

    st.divider()

    st.subheader("💡 Quick Tips")

    st.markdown(
        """
        - Explain concepts in your own words.
        - Use examples from your projects.
        - Keep answers clear and structured.
        - Review feedback after every question.
        - Practise regularly to build confidence.
        """
    )


# ==================================================
# 7. INTERVIEW PRACTICE
# ==================================================

elif page == "💬 Interview Practice":

    st.title("💬 Interview Practice")

    st.write(
        f"Role: **{st.session_state.current_role}** "
        f"| Difficulty: **{st.session_state.current_level}**"
    )

    if st.session_state.resume_text:
        st.success(
            "Resume-based mode is enabled. Questions "
            "can be personalized using your resume."
        )
    else:
        st.info(
            "General interview mode is active. Upload a "
            "resume in the sidebar for personalized questions."
        )

    st.divider()

    st.subheader("💡 Generate an Interview Question")

    if st.button("✨ Generate Interview Question"):

        role = st.session_state.current_role
        level = st.session_state.current_level
        resume = st.session_state.resume_text

        if resume:
            resume_context = (
                "Personalize the question using the candidate's "
                "resume. Do not invent skills or projects.\n\n"
                f"Resume:\n{resume}"
            )
        else:
            resume_context = (
                "Generate a general technical question "
                "for the selected role."
            )

        prompt = f"""
You are an experienced technical interviewer.

Role: {role}
Difficulty: {level}

{resume_context}

Generate exactly one relevant interview question.
Do not provide the answer.
Return only the question.
"""

        with st.spinner("Generating your question..."):
            try:
                question_text, used_model = (
                    generate_with_fallback(prompt)
                )

                st.session_state.question = question_text
                st.session_state.question_model = used_model
                st.session_state.evaluation = None
                st.session_state.answer = ""

            except Exception as exc:
                st.error(
                    "Unable to generate a question. "
                    "The configured Gemini models may be unavailable."
                )
                st.code(str(exc))

    if st.session_state.question:

        st.markdown("### Your Question")
        st.write(st.session_state.question)

        if st.session_state.question_model:
            st.caption(
                f"Model: {st.session_state.question_model}"
            )

        st.divider()

        st.subheader("✍️ Your Answer")

        answer = st.text_area(
            "Write your answer below",
            height=180,
            placeholder="Type your answer here...",
            key="answer_input"
        )

        if st.button("🧠 Evaluate My Answer"):

            if not answer.strip():
                st.warning(
                    "Please enter an answer first."
                )

            else:
                evaluation_prompt = f"""
You are an experienced technical interviewer.

Role: {st.session_state.current_role}
Difficulty: {st.session_state.current_level}

Question:
{st.session_state.question}

Candidate answer:
{answer}

Evaluate accuracy, relevance, clarity, and completeness.

Return ONLY valid JSON with this structure:
{{
  "score": 0,
  "verdict": "Needs Improvement",
  "feedback": "Constructive feedback",
  "strengths": ["strength"],
  "missing_concepts": ["concept"],
  "sample_answer": "A good example answer"
}}

The score must be an integer from 0 to 100.
Be fair and technically accurate.
Return JSON only, without Markdown fences.
"""

                with st.spinner("Evaluating your answer..."):
                    try:
                        result_text, used_model = (
                            generate_with_fallback(
                                evaluation_prompt
                            )
                        )

                        match = re.search(
                            r"\{.*\}",
                            result_text,
                            flags=re.DOTALL
                        )

                        if not match:
                            raise ValueError(
                                "No JSON object found in the response."
                            )

                        evaluation = json.loads(match.group(0))

                        score = int(
                            evaluation.get("score", 0)
                        )
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
                        st.session_state.answer = answer

                        st.session_state.history.append({
                            "role": st.session_state.current_role,
                            "level": st.session_state.current_level,
                            "question": st.session_state.question,
                            "answer": answer,
                            "score": score,
                            "verdict": evaluation["verdict"],
                        })

                        st.success(
                            f"Evaluation completed using {used_model}."
                        )

                    except Exception as exc:
                        st.error(
                            "Could not evaluate the answer."
                        )
                        st.code(str(exc))

    # Display evaluation outside the button block so that
    # it remains visible after Streamlit reruns.

    evaluation = st.session_state.evaluation

    if evaluation:
        st.divider()
        st.header("📊 Answer Evaluation")

        score_col, verdict_col = st.columns(2)

        score_col.metric(
            "Your Score",
            f"{evaluation['score']}/100"
        )

        verdict_col.metric(
            "Verdict",
            evaluation["verdict"]
        )

        st.subheader("📝 Feedback")
        st.write(evaluation["feedback"])

        st.subheader("✅ Strengths")

        for item in evaluation.get("strengths", []):
            st.markdown(f"- {item}")

        st.subheader("📌 Areas to Improve")

        for item in evaluation.get("missing_concepts", []):
            st.markdown(f"- {item}")

        st.subheader("💡 Example Answer")
        st.write(evaluation["sample_answer"])


# ==================================================
# 8. PERFORMANCE DASHBOARD
# ==================================================

elif page == "📊 Performance":

    st.title("📊 Performance Dashboard")

    history = st.session_state.history

    if not history:
        st.info(
            "No interview results yet. Complete an interview "
            "to see your performance here."
        )

    else:
        scores = [
            item["score"] for item in history
        ]

        average_score = sum(scores) / len(scores)
        best_score = max(scores)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Questions Evaluated",
            len(history)
        )

        col2.metric(
            "Average Score",
            f"{average_score:.1f}%"
        )

        col3.metric(
            "Best Score",
            f"{best_score}%"
        )

        st.divider()

        st.subheader("📋 Interview History")

        for number, item in enumerate(
            reversed(history),
            start=1
        ):
            with st.expander(
                f"Interview {number} — "
                f"{item['score']}/100 — {item['role']}"
            ):
                st.write("**Difficulty:**", item["level"])
                st.write("**Question:**", item["question"])
                st.write("**Your answer:**", item["answer"])
                st.write("**Verdict:**", item["verdict"])

        st.download_button(
            "⬇️ Download Interview History",
            data=json.dumps(
                history,
                indent=2,
                ensure_ascii=False
            ),
            file_name="interview_history.json",
            mime="application/json"
        )

        if st.button("🗑️ Clear Performance History"):
            st.session_state.history = []
            st.rerun()


# ==================================================
# 9. FOOTER
# ==================================================

st.divider()

st.caption(
    "AI Interview Studio | Built with Streamlit and Gemini. "
    "AI-generated scores are guidance, not a hiring decision."
)
