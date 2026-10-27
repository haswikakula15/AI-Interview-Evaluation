
import os
import json
import streamlit as st
import fitz
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.8-flash"

st.set_page_config(
    page_title="AI Interview Studio",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# DARK NAVY + PURPLE DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0B1020, #111827, #17132B);
    color: #F8FAFC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #E9D5FF !important;
}

[data-testid="stSidebar"] {
    background: #0D1425;
    border-right: 1px solid #273449;
}

.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #A855F7);
    color: white !important;
    border: 1px solid #A78BFA;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background: #6D28D9;
    border-color: #C4B5FD;
}

.stTextArea textarea {
    background: #151E30 !important;
    color: #F8FAFC !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1E293B, #252044);
    border: 1px solid #55417D;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #C4B5FD !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎯 AI Interview Studio")
st.write(
    "Practice interviews, receive AI-powered feedback, "
    "and prepare confidently for your next opportunity."
)

st.caption("Personalized questions • Intelligent evaluation • Actionable feedback")

if not API_KEY:
    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY to your "
        "local .env file or Streamlit Cloud Secrets."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "history": [],
    "question": None,
    "evaluation": None,
    "answer": "",
    "current_role": None,
    "current_level": None,
    "resume_text": "",
    "resume_name": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# SIDEBAR: SETTINGS AND RESUME
# --------------------------------------------------

st.sidebar.title("Interview Setup")

role = st.sidebar.selectbox(
    "Choose your target role",
    [
        "Java Developer",
        "Python Developer",
        "AI/ML Engineer",
        "Data Analyst",
        "Software Engineer"
    ]
)

level = st.sidebar.selectbox(
    "Difficulty level",
    ["Beginner", "Intermediate", "Advanced"]
)

st.sidebar.divider()
st.sidebar.subheader("Resume-Based Interview")

uploaded_resume = st.sidebar.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

if uploaded_resume is not None:
    if st.session_state.resume_name != uploaded_resume.name:
        try:
            with fitz.open(
                stream=uploaded_resume.getvalue(),
                filetype="pdf"
            ) as pdf:
                extracted_text = "\n".join(
                    page.get_text() for page in pdf
                )

            if extracted_text.strip():
                st.session_state.resume_text = extracted_text[:12000]
                st.session_state.resume_name = uploaded_resume.name
                st.sidebar.success("Resume text extracted!")
            else:
                st.session_state.resume_text = ""
                st.session_state.resume_name = None
                st.sidebar.warning(
                    "No readable text found. Try a text-based PDF."
                )

        except Exception as e:
            st.sidebar.error(f"Could not read the PDF: {e}")

if st.session_state.resume_text:
    st.sidebar.caption(
        f"Resume loaded: {st.session_state.resume_name}"
    )
    if st.sidebar.button("Remove Resume"):
        st.session_state.resume_text = ""
        st.session_state.resume_name = None
        st.session_state.question = None
        st.session_state.evaluation = None
        st.rerun()

st.sidebar.divider()

if st.sidebar.button("Start New Interview"):
    st.session_state.history = []
    st.session_state.question = None
    st.session_state.evaluation = None
    st.session_state.answer = ""
    st.session_state.current_role = None
    st.session_state.current_level = None
    st.rerun()

# --------------------------------------------------
# INTERVIEW MODE
# --------------------------------------------------

if st.session_state.resume_text:
    st.info(
        "Resume-based mode is enabled. Questions can be tailored "
        "to the skills and projects identified in your PDF."
    )
else:
    st.info(
        "General interview mode is enabled. Upload a PDF resume "
        "in the sidebar for personalized questions."
    )

# --------------------------------------------------
# GENERATE QUESTION
# --------------------------------------------------

if st.button("✨ Generate Interview Question", type="primary"):
    resume_context = ""

    if st.session_state.resume_text:
        resume_context = f"""
        Candidate resume:
        {st.session_state.resume_text}

        Ask a question related to a skill, project, education,
        or experience actually mentioned in the resume.
        Do not invent qualifications or experience.
        """

    previous_questions = [
        item["question"] for item in st.session_state.history
    ]

    prompt = f"""
    Act as a professional technical interviewer.

    Target role: {role}
    Difficulty: {level}

    {resume_context}

    Generate one clear, relevant interview question.
    Avoid repeating these previous questions:
    {previous_questions}

    Return only the question, without its answer.
    """

    with st.spinner("Generating your interview question..."):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            if not response.text:
                st.error("The AI returned an empty question. Try again.")
            else:
                st.session_state.question = response.text.strip()
                st.session_state.evaluation = None
                st.session_state.answer = ""
                st.session_state.current_role = role
                st.session_state.current_level = level
                st.rerun()

        except Exception as e:
            st.error(f"Could not generate a question: {e}")

# --------------------------------------------------
# DISPLAY QUESTION AND COLLECT ANSWER
# --------------------------------------------------

if st.session_state.question:
    st.divider()
    st.subheader("📝 Your Interview Question")
    st.write(st.session_state.question)

    answer = st.text_area(
        "Your answer",
        value=st.session_state.answer,
        height=180,
        placeholder="Explain your answer in your own words...",
        key="candidate_answer"
    )

    if st.button("Evaluate My Answer", type="primary"):
        if not answer.strip():
            st.warning("Please enter your answer first.")
        else:
            evaluation_prompt = f"""
            You are a fair and professional technical interviewer.

            Role: {st.session_state.current_role}
            Difficulty: {st.session_state.current_level}
            Question: {st.session_state.question}
            Candidate answer: {answer}

            Evaluate technical correctness, understanding,
            completeness and clarity. Accept valid alternative
            explanations. Do not require exact wording.

            Return ONLY valid JSON using this structure:
            {{
              "score": 0,
              "verdict": "Incorrect",
              "feedback": "Constructive feedback",
              "strengths": ["A strength"],
              "missing_concepts": ["A concept to improve"],
              "sample_answer": "A clear model answer"
            }}

            Score must be an integer from 0 to 10.
            Use verdict Correct, Partially Correct, or Incorrect.
            Award 0 for empty or irrelevant answers.
            Do not give full marks to an incorrect answer.
            """

            with st.spinner("Evaluating your answer with AI..."):
                try:
                    result = client.models.generate_content(
                        model=MODEL,
                        contents=evaluation_prompt
                    )

                    raw = (result.text or "").strip()
                    raw = raw.removeprefix("```json").removeprefix("```")
                    raw = raw.removesuffix("```").strip()

                    evaluation = json.loads(raw)

                    score = max(
                        0, min(10, int(evaluation["score"]))
                    )

                    verdict = evaluation.get(
                        "verdict", "Partially Correct"
                    )

                    if verdict not in [
                        "Correct", "Partially Correct", "Incorrect"
                    ]:
                        verdict = "Partially Correct"

                    evaluation["score"] = score
                    evaluation["verdict"] = verdict

                    st.session_state.evaluation = evaluation
                    st.session_state.answer = answer

                    st.session_state.history.append({
                        "role": st.session_state.current_role,
                        "question": st.session_state.question,
                        "answer": answer,
                        "score": score,
                        "verdict": verdict,
                    })

                    st.rerun()

                except json.JSONDecodeError:
                    st.error(
                        "The AI returned an unexpected format. "
                        "Please try evaluating again."
                    )
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

# --------------------------------------------------
# DISPLAY EVALUATION
# --------------------------------------------------

if st.session_state.evaluation:
    evaluation = st.session_state.evaluation

    st.divider()
    st.subheader("📊 Evaluation Results")

    col1, col2 = st.columns(2)

    col1.metric("Your Score", f'{evaluation["score"]}/10')
    col2.metric("Verdict", evaluation["verdict"])

    st.markdown("### 💬 Feedback")
    st.write(evaluation.get("feedback", "No feedback provided."))

    st.markdown("### ✅ Strengths")
    strengths = evaluation.get("strengths", [])
    if strengths:
        for item in strengths:
            st.write(f"• {item}")
    else:
        st.write("No strengths were listed.")

    st.markdown("### 📚 Concepts to Improve")
    missing = evaluation.get("missing_concepts", [])
    if missing:
        for item in missing:
            st.write(f"• {item}")
    else:
        st.write("No major missing concepts identified.")

    st.markdown("### 💡 Sample Answer")
    st.write(
        evaluation.get("sample_answer", "No sample answer provided.")
    )

# --------------------------------------------------
# PERFORMANCE DASHBOARD
# --------------------------------------------------

if st.session_state.history:
    st.divider()
    st.subheader("📈 Interview Performance Dashboard")

    scores = [
        item["score"] for item in st.session_state.history
    ]

    average_score = sum(scores) / len(scores)

    col1, col2, col3 = st.columns(3)
    col1.metric("Questions Attempted", len(scores))
    col2.metric("Average Score", f"{average_score:.1f}/10")
    col3.metric("Highest Score", f"{max(scores)}/10")

    st.markdown("### Interview History")

    st.dataframe(
        [
            {
                "Role": item["role"],
                "Question": item["question"],
                "Score": f'{item["score"]}/10',
                "Verdict": item["verdict"]
            }
            for item in st.session_state.history
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "History is kept for this active Streamlit session only. "
        "It is not permanently stored."
    )

st.divider()
st.caption("AI Interview Studio | Powered by Streamlit and Gemini")
