import io
import json
import os
import re
import time

import fitz
import streamlit as st
from dotenv import load_dotenv
from google import genai
from supabase import create_client
from supabase.client import Client


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="AI Interview Studio",
    page_icon="🎯",
    layout="wide",
)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")

try:
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

MODEL_FALLBACKS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
]

if not GEMINI_API_KEY:
    st.error(
        "Gemini API key is missing. Add GEMINI_API_KEY to your "
        "local .env file or Streamlit Cloud Secrets."
    )
    st.stop()

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "Supabase settings are missing. Add SUPABASE_URL and "
        "SUPABASE_KEY to your local .env file or Streamlit Cloud Secrets."
    )
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# 2. PAGE DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #17152d);
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
        background: linear-gradient(90deg, #7c3aed, #c026d3);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
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
        padding: 20px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. SESSION STATE
# =========================================================

DEFAULTS = {
    "authenticated": False,
    "student": None,
    "supabase_client": None,
    "navigation": "🏠 Dashboard",
    "profile": None,
    "resume_text": "",
    "resume_name": "",
    "question": "",
    "question_model": "",
    "evaluation": None,
    "answer": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. HELPERS
# =========================================================

def new_supabase_client() -> Client:
    """Create an isolated Supabase client for one Streamlit session."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user_id() -> str:
    student = st.session_state.student
    if not student:
        raise RuntimeError("No authenticated student is available.")
    return str(student.id)


def generate_with_fallback(prompt: str) -> tuple[str, str]:
    """Try configured Gemini models, retrying temporary failures."""
    errors = []

    for model in MODEL_FALLBACKS:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                if response.text and response.text.strip():
                    return response.text.strip(), model

                errors.append(f"{model}: Empty response.")
                break

            except Exception as exc:
                message = str(exc)
                errors.append(f"{model}: {message[:250]}")

                temporary = any(
                    marker in message.upper()
                    for marker in (
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                    )
                )

                if temporary and attempt == 0:
                    time.sleep(2)
                    continue

                break

    raise RuntimeError(
        "All configured Gemini models failed:\n"
        + "\n".join(errors[-6:])
    )


def load_or_create_profile():
    """Load the signed-in student's profile or create it once."""
    client = st.session_state.supabase_client
    student = st.session_state.student
    user_id = str(student.id)

    result = (
        client.table("student_profiles")
        .select("*")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )

    if result.data:
        profile = result.data[0]
    else:
        metadata = getattr(student, "user_metadata", None) or {}
        email = getattr(student, "email", "") or ""
        default_name = metadata.get("full_name") or email.split("@")[0]

        profile_data = {
            "id": user_id,
            "full_name": default_name,
            "course": "B.E. CSE (AI-ML)",
            "college": "Keshav Memorial Engineering College",
            "graduation_year": 2027,
            "career_goal": "AI/ML Engineer",
        }

        inserted = (
            client.table("student_profiles")
            .insert(profile_data)
            .execute()
        )

        profile = inserted.data[0] if inserted.data else profile_data

    st.session_state.profile = profile


def load_history() -> list[dict]:
    """Load only the current student's history; RLS also enforces this."""
    client = st.session_state.supabase_client
    user_id = get_user_id()

    result = (
        client.table("interview_history")
        .select(
            "id, role, difficulty, question, answer, score, verdict, "
            "feedback, strengths, missing_concepts, sample_answer, created_at"
        )
        .eq("student_id", user_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )

    return result.data or []


def save_interview(evaluation: dict, answer: str):
    """Save one evaluated answer to the authenticated student's history."""
    client = st.session_state.supabase_client

    row = {
        "student_id": get_user_id(),
        "role": st.session_state.current_role,
        "difficulty": st.session_state.current_level,
        "question": st.session_state.question,
        "answer": answer,
        "score": int(evaluation["score"]),
        "verdict": str(evaluation.get("verdict", "")),
        "feedback": str(evaluation.get("feedback", "")),
        "strengths": evaluation.get("strengths", []),
        "missing_concepts": evaluation.get("missing_concepts", []),
        "sample_answer": str(evaluation.get("sample_answer", "")),
    }

    client.table("interview_history").insert(row).execute()


def clear_current_interview():
    st.session_state.question = ""
    st.session_state.question_model = ""
    st.session_state.evaluation = None
    st.session_state.answer = ""


# =========================================================
# 5. LOGIN AND REGISTRATION
# =========================================================

if not st.session_state.authenticated:
    st.title("🎓 AI Interview Studio")
    st.write("Sign in or create a student account to continue.")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email address")
            login_password = st.text_input("Password", type="password")

            login_submit = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if login_submit:
            if not login_email.strip() or not login_password:
                st.warning("Enter your email and password.")
            else:
                try:
                    auth_client = new_supabase_client()

                    auth_result = auth_client.auth.sign_in_with_password(
                        {
                            "email": login_email.strip(),
                            "password": login_password,
                        }
                    )

                    if not auth_result.user or not auth_result.session:
                        st.error("Login did not return a valid session.")
                    else:
                        st.session_state.student = auth_result.user
                        st.session_state.supabase_client = auth_client
                        st.session_state.authenticated = True
                        st.session_state.navigation = "🏠 Dashboard"
                        st.session_state.profile = None

                        load_or_create_profile()
                        st.rerun()

                except Exception as exc:
                    st.error(
                        "Login failed. Check your credentials and whether "
                        "email confirmation is required."
                    )
                    st.caption(str(exc)[:500])

    with register_tab:
        with st.form("register_form"):
            register_name = st.text_input("Full name")
            register_email = st.text_input("Email address")

            register_password = st.text_input(
                "Create password (at least 8 characters)",
                type="password",
            )

            register_password_confirm = st.text_input(
                "Confirm password",
                type="password",
            )

            register_submit = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

        if register_submit:
            if not register_name.strip():
                st.warning("Enter your full name.")

            elif not register_email.strip():
                st.warning("Enter your email address.")

            elif len(register_password) < 8:
                st.warning("Use a password with at least 8 characters.")

            elif register_password != register_password_confirm:
                st.warning("The passwords do not match.")

            else:
                try:
                    auth_client = new_supabase_client()

                    signup_result = auth_client.auth.sign_up(
                        {
                            "email": register_email.strip(),
                            "password": register_password,
                            "options": {
                                "data": {"full_name": register_name.strip()}
                            },
                        }
                    )

                    if signup_result.session and signup_result.user:
                        st.session_state.student = signup_result.user
                        st.session_state.supabase_client = auth_client
                        st.session_state.authenticated = True
                        st.session_state.navigation = "🏠 Dashboard"

                        load_or_create_profile()

                        st.success("Account created successfully.")
                        st.rerun()

                    else:
                        st.success(
                            "Registration submitted. Check your email to "
                            "confirm your account, then return to the Login tab."
                        )

                except Exception as exc:
                    st.error(
                        "Registration failed. Check the email and try again."
                    )
                    st.caption(str(exc)[:500])

    st.stop()


# =========================================================
# 6. LOAD AUTHENTICATED PROFILE
# =========================================================

if (
    st.session_state.student is None
    or st.session_state.supabase_client is None
):
    st.session_state.authenticated = False
    st.rerun()

if st.session_state.profile is None:
    try:
        load_or_create_profile()
    except Exception as exc:
        st.error("Could not load your student profile.")
        st.caption(str(exc)[:500])
        st.stop()


# =========================================================
# 7. SIDEBAR
# =========================================================

profile = st.session_state.profile or {}
student = st.session_state.student

student_email = getattr(student, "email", "") or ""
student_name = profile.get("full_name") or student_email.split("@")[0]

with st.sidebar:
    st.title("🎯 AI Interview Studio")
    st.caption(f"Signed in as {student_name}")

    st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💬 Interview Practice",
            "📊 Performance",
            "👤 My Profile",
        ],
        key="navigation",
    )

    st.divider()
    st.subheader("Interview Settings")

    roles = [
        "Java Developer",
        "Python Developer",
        "AI/ML Engineer",
        "Data Analyst",
        "Software Engineer",
    ]

    levels = ["Beginner", "Intermediate", "Advanced"]

    if "current_role" not in st.session_state:
        st.session_state.current_role = "Java Developer"

    if "current_level" not in st.session_state:
        st.session_state.current_level = "Beginner"

    st.session_state.current_role = st.selectbox(
        "Target role",
        roles,
        index=(
            roles.index(st.session_state.current_role)
            if st.session_state.current_role in roles
            else 0
        ),
    )

    st.session_state.current_level = st.selectbox(
        "Difficulty",
        levels,
        index=(
            levels.index(st.session_state.current_level)
            if st.session_state.current_level in levels
            else 0
        ),
    )

    st.divider()
    st.subheader("Resume PDF")

    uploaded_resume = st.file_uploader(
        "Upload a resume",
        type=["pdf"],
        key="resume_uploader",
    )

    if uploaded_resume is not None:
        if st.session_state.resume_name != uploaded_resume.name:
            st.session_state.resume_text = ""
            st.session_state.resume_name = ""

            try:
                pdf_bytes = uploaded_resume.getvalue()

                with fitz.open(
                    stream=pdf_bytes,
                    filetype="pdf",
                ) as pdf:
                    extracted_text = "\n".join(
                        page.get_text() for page in pdf
                    )

                if not extracted_text.strip():
                    with st.spinner("Reading scanned resume with OCR..."):
                        try:
                            import easyocr
                            import numpy as np
                            from PIL import Image

                            reader = easyocr.Reader(["en"], gpu=False)
                            page_texts = []

                            with fitz.open(
                                stream=pdf_bytes,
                                filetype="pdf",
                            ) as pdf:
                                for pdf_page in pdf:
                                    pix = pdf_page.get_pixmap(
                                        matrix=fitz.Matrix(2, 2),
                                        alpha=False,
                                    )

                                    image = Image.open(
                                        io.BytesIO(pix.tobytes("png"))
                                    )

                                    words = reader.readtext(
                                        np.array(image),
                                        detail=0,
                                        paragraph=True,
                                    )

                                    page_texts.append("\n".join(words))

                            extracted_text = "\n".join(page_texts)

                        except Exception:
                            st.warning(
                                "OCR was unavailable. Install easyocr, numpy, "
                                "and Pillow, or upload a text-based PDF."
                            )

                if extracted_text.strip():
                    st.session_state.resume_text = extracted_text[:12000]
                    st.session_state.resume_name = uploaded_resume.name
                    st.success("Resume text extracted.")
                else:
                    st.warning("No readable text found in this PDF.")

            except Exception as exc:
                st.error("Could not read this PDF.")
                st.caption(str(exc)[:400])

    if st.session_state.resume_text:
        st.caption(f"Loaded: {st.session_state.resume_name}")

        if st.button("Remove Resume", use_container_width=True):
            st.session_state.resume_text = ""
            st.session_state.resume_name = ""
            st.rerun()

    st.divider()

    if st.button("Start New Interview", use_container_width=True):
        clear_current_interview()
        st.session_state.navigation = "💬 Interview Practice"
        st.rerun()

    if st.button("Log Out", use_container_width=True):
        try:
            st.session_state.supabase_client.auth.sign_out()
        except Exception:
            pass

        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.rerun()


# =========================================================
# 8. LOAD THIS STUDENT'S SAVED HISTORY
# =========================================================

try:
    saved_history = load_history()
except Exception as exc:
    st.error("Could not load your saved interview history.")
    st.caption(str(exc)[:500])
    saved_history = []


# =========================================================
# 9. STUDENT DASHBOARD
# =========================================================

page = st.session_state.navigation

if page == "🏠 Dashboard":
    st.title(f"👋 Welcome, {student_name}!")

    st.write(
        "Your personal space to practise interviews, improve your skills, "
        "and track your progress."
    )

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            f"""
            <div class="profile-card">
                <h2>{student_name}</h2>
                <p>{profile.get("course") or "Add your course in My Profile"}</p>
                <p>{profile.get("college") or "Add your college in My Profile"}</p>
                <p>Graduation year: {profile.get("graduation_year") or "Not set"}</p>
                <p>Career goal: {profile.get("career_goal") or "Not set"}</p>
                <p>{student_email}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="profile-card">
                <h3>🎯 Your Preparation Goals</h3>
                <p>Practise consistently.</p>
                <p>Review feedback after each answer.</p>
                <p>Improve your technical explanations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scores = [int(item["score"]) for item in saved_history]
    average_score = sum(scores) / len(scores) if scores else 0
    best_score = max(scores, default=0)

    st.subheader("📈 Your Progress")

    c1, c2, c3 = st.columns(3)
    c1.metric("Questions Evaluated", len(saved_history))
    c2.metric("Average Score", f"{average_score:.1f}%")
    c3.metric("Best Score", f"{best_score}%")

    st.divider()
    st.subheader("🚀 Ready to practise?")

    st.write(
        "Generate role-specific interview questions, answer them, "
        "and save your scores to your student account."
    )

    if st.button(
        "▶ Start Interview Practice",
        use_container_width=True,
    ):
        st.session_state.navigation = "💬 Interview Practice"
        st.rerun()

    st.subheader("💡 Quick Tips")

    st.markdown(
        """
        - Explain technical concepts in your own words.
        - Use examples from your projects.
        - Keep answers clear and structured.
        - Review your feedback and practise regularly.
        """
    )


# =========================================================
# 10. INTERVIEW PRACTICE
# =========================================================

elif page == "💬 Interview Practice":
    st.title("💬 Interview Practice")

    st.write(
        f"Role: **{st.session_state.current_role}** · "
        f"Difficulty: **{st.session_state.current_level}**"
    )

    if st.session_state.resume_text:
        st.success("Resume-based interview mode is enabled.")
    else:
        st.info(
            "General interview mode is active. Upload a PDF resume "
            "in the sidebar to personalize your questions."
        )

    if st.button("✨ Generate Interview Question"):
        resume_text = st.session_state.resume_text

        if resume_text:
            resume_context = (
                "Use the candidate's resume to personalize the question. "
                "Do not invent skills or projects.\n\n"
                f"RESUME:\n{resume_text}"
            )
        else:
            resume_context = (
                "Generate a general interview question for the selected role."
            )

        prompt = f"""
You are an experienced technical interviewer.
Role: {st.session_state.current_role}
Difficulty: {st.session_state.current_level}

{resume_context}

Generate exactly ONE relevant interview question.
Do not provide the answer. Return only the question.
"""

        with st.spinner("Generating your interview question..."):
            try:
                question, used_model = generate_with_fallback(prompt)

                st.session_state.question = question
                st.session_state.question_model = used_model
                st.session_state.evaluation = None
                st.session_state.answer = ""

                st.rerun()

            except Exception as exc:
                st.error("Gemini could not generate a question.")
                st.caption(str(exc)[:1000])

    if st.session_state.question:
        st.subheader("Your Question")
        st.write(st.session_state.question)

        if st.session_state.question_model:
            st.caption(
                f"Generated with {st.session_state.question_model}"
            )

        answer = st.text_area(
            "Write your answer",
            value=st.session_state.answer,
            height=180,
            key="current_answer_input",
            placeholder="Type your answer here...",
        )

        st.session_state.answer = answer

        if st.button("🧠 Evaluate My Answer"):
            if not answer.strip():
                st.warning("Please enter your answer first.")
            else:
                evaluation_prompt = f"""
You are a fair technical interviewer.

Role: {st.session_state.current_role}
Difficulty: {st.session_state.current_level}
Question: {st.session_state.question}
Candidate answer: {answer}

Evaluate accuracy, relevance, clarity, and completeness.

Return ONLY valid JSON:
{{
  "score": 0,
  "verdict": "Needs Improvement",
  "feedback": "Constructive feedback",
  "strengths": ["strength"],
  "missing_concepts": ["concept"],
  "sample_answer": "Example of a strong answer"
}}

Score must be an integer from 0 to 100.
Return JSON only, without Markdown fences.
"""

                with st.spinner("Evaluating your answer..."):
                    try:
                        result_text, used_model = generate_with_fallback(
                            evaluation_prompt
                        )

                        match = re.search(
                            r"\{.*\}",
                            result_text,
                            flags=re.DOTALL,
                        )

                        if not match:
                            raise ValueError(
                                "The model did not return valid JSON."
                            )

                        evaluation = json.loads(match.group(0))

                        score = max(
                            0,
                            min(100, int(evaluation.get("score", 0))),
                        )

                        evaluation["score"] = score
                        evaluation["verdict"] = str(
                            evaluation.get("verdict", "Evaluated")
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

                        # Save through the authenticated Supabase client.
                        save_interview(evaluation, answer)

                        st.session_state.evaluation = evaluation

                        st.success(
                            f"Answer evaluated and saved using {used_model}."
                        )

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            "Could not evaluate or save the answer. "
                            "Check your Gemini and Supabase configuration."
                        )
                        st.caption(str(exc)[:1000])

        if st.session_state.evaluation:
            evaluation = st.session_state.evaluation

            st.divider()
            st.header("📊 Your Evaluation")

            c1, c2 = st.columns(2)
            c1.metric("Score", f"{evaluation['score']}/100")
            c2.metric("Verdict", evaluation["verdict"])

            st.subheader("📝 Feedback")
            st.write(evaluation.get("feedback", ""))

            st.subheader("✅ Strengths")

            strengths = evaluation.get("strengths", [])

            if strengths:
                for strength in strengths:
                    st.markdown(f"- {strength}")
            else:
                st.write("No specific strengths were listed.")

            st.subheader("📌 Areas to Improve")

            missing = evaluation.get("missing_concepts", [])

            if missing:
                for concept in missing:
                    st.markdown(f"- {concept}")
            else:
                st.write("No specific missing concepts were listed.")

            st.subheader("💡 Example Answer")
            st.write(evaluation.get("sample_answer", ""))


# =========================================================
# 11. PERFORMANCE PAGE
# =========================================================

elif page == "📊 Performance":
    st.title("📊 Performance Dashboard")

    if not saved_history:
        st.info(
            "No saved interview attempts yet. Complete an interview "
            "to see your results here."
        )
    else:
        scores = [int(item["score"]) for item in saved_history]
        average_score = sum(scores) / len(scores)

        c1, c2, c3 = st.columns(3)
        c1.metric("Saved Attempts", len(saved_history))
        c2.metric("Average Score", f"{average_score:.1f}%")
        c3.metric("Best Score", f"{max(scores)}%")

        st.divider()
        st.subheader("📋 Saved Interview History")

        for index, item in enumerate(saved_history, start=1):
            created = item.get("created_at", "")

            with st.expander(
                f"{index}. {item.get('role', 'Interview')} · "
                f"{item.get('score', 0)}/100 · {created}"
            ):
                st.write("**Difficulty:**", item.get("difficulty", ""))
                st.write("**Question:**", item.get("question", ""))
                st.write("**Your answer:**", item.get("answer", ""))
                st.write("**Verdict:**", item.get("verdict", ""))
                st.write("**Feedback:**", item.get("feedback", ""))
                st.write("**Strengths:**", item.get("strengths", []))

                st.write(
                    "**Areas to improve:**",
                    item.get("missing_concepts", []),
                )

                st.write(
                    "**Example answer:**",
                    item.get("sample_answer", ""),
                )

        st.download_button(
            "⬇️ Download My Interview History",
            data=json.dumps(
                saved_history,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="my_interview_history.json",
            mime="application/json",
        )


# =========================================================
# 12. PROFILE PAGE
# =========================================================

elif page == "👤 My Profile":
    st.title("👤 My Student Profile")
    st.write("Update your academic and career details.")

    current_profile = st.session_state.profile or {}

    with st.form("profile_form"):
        full_name = st.text_input(
            "Full name",
            value=current_profile.get("full_name", ""),
        )

        course = st.text_input(
            "Course",
            value=current_profile.get("course", ""),
        )

        college = st.text_input(
            "College",
            value=current_profile.get("college", ""),
        )

        graduation_year = st.number_input(
            "Graduation year",
            min_value=2000,
            max_value=2100,
            value=int(current_profile.get("graduation_year") or 2027),
        )

        career_goal = st.text_input(
            "Career goal",
            value=current_profile.get("career_goal", ""),
        )

        save_profile = st.form_submit_button(
            "Save Profile",
            use_container_width=True,
        )

    if save_profile:
        try:
            updated_profile = {
                "id": get_user_id(),
                "full_name": full_name.strip(),
                "course": course.strip(),
                "college": college.strip(),
                "graduation_year": int(graduation_year),
                "career_goal": career_goal.strip(),
            }

            result = (
                st.session_state.supabase_client
                .table("student_profiles")
                .upsert(updated_profile, on_conflict="id")
                .execute()
            )

            st.session_state.profile = (
                result.data[0] if result.data else updated_profile
            )

            st.success("Your profile has been saved.")
            st.rerun()

        except Exception as exc:
            st.error("Could not save your profile.")
            st.caption(str(exc)[:800])


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Interview Studio · Student data is stored in Supabase. "
    "AI-generated feedback is guidance, not a hiring decision."
)