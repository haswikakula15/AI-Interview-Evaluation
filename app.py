
import os
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Interview Evaluation",
    page_icon="🎯",
    layout="wide"
)

st.title("AI Interview Preparation & Evaluation System")
st.write("Practice interviews and receive AI-powered feedback on your answers.")

if not API_KEY:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.5-flash-lite"

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "question" not in st.session_state:
    st.session_state.question = None

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "answer" not in st.session_state:
    st.session_state.answer = ""

# Sidebar settings
st.sidebar.header("Interview Settings")

role = st.sidebar.selectbox(
    "Choose your role",
    [
        "Java Developer",
        "Python Developer",
        "AI/ML Engineer",
        "Data Analyst",
        "Software Engineer"
    ]
)

level = st.sidebar.selectbox(
    "Difficulty",
    ["Beginner", "Intermediate", "Advanced"]
)

if st.sidebar.button("Start New Interview"):
    st.session_state.history = []
    st.session_state.question = None
    st.session_state.evaluation = None
    st.session_state.answer = ""
    st.rerun()

# Generate question
if st.button("Generate Interview Question", type="primary"):
    with st.spinner("Generating your question..."):
        try:
            prompt = f"""
            Act as a professional technical interviewer.
            Generate one unique interview question for the role: {role}.
            Difficulty: {level}.
            Avoid repeating these previous questions:
            { [item["question"] for item in st.session_state.history] }

            Return only the question text, without an answer.
            """

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            st.session_state.question = response.text.strip()
            st.session_state.evaluation = None
            st.session_state.answer = ""

        except Exception as e:
            st.error(f"Could not generate a question: {e}")

if st.session_state.question:
    st.subheader("Your Interview Question")
    st.write(st.session_state.question)

    answer = st.text_area(
        "Type your answer here",
        value=st.session_state.answer,
        height=180,
        key="answer_input",
        placeholder="Explain your answer clearly..."
    )

    if st.button("Evaluate My Answer", type="primary"):
        if not answer.strip():
            st.warning("Please enter an answer first.")
        else:
            with st.spinner("AI is evaluating your answer..."):
                try:
                    prompt = f"""
                    You are a fair and professional technical interviewer.

                    Role: {role}
                    Difficulty: {level}
                    Question: {st.session_state.question}
                    Candidate answer: {answer}

                    Evaluate technical correctness, understanding,
                    completeness, and clarity. Accept valid alternative
                    explanations. Do not require exact wording.

                    Return ONLY valid JSON with these keys:
                    {{
                      "score": integer from 0 to 10,
                      "verdict": "Correct", "Partially Correct", or "Incorrect",
                      "feedback": "constructive explanation",
                      "strengths": ["strength 1", "strength 2"],
                      "missing_concepts": ["concept 1", "concept 2"],
                      "sample_answer": "a clear model answer"
                    }}

                    Award 0 if the answer is empty or irrelevant.
                    Do not give full marks to an incorrect answer.
                    """

                    result = client.models.generate_content(
                        model=MODEL,
                        contents=prompt
                    )

                    text = result.text.strip()
                    text = text.removeprefix("```json").removeprefix("```")
                    text = text.removesuffix("```").strip()

                    evaluation = json.loads(text)
                    evaluation["score"] = max(
                        0, min(10, int(evaluation["score"]))
                    )

                    st.session_state.evaluation = evaluation
                    st.session_state.answer = answer

                    st.session_state.history.append({
                        "role": role,
                        "question": st.session_state.question,
                        "answer": answer,
                        "score": evaluation["score"],
                        "verdict": evaluation["verdict"]
                    })

                except json.JSONDecodeError:
                    st.error(
                        "The AI returned an unexpected format. "
                        "Please try evaluating again."
                    )
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

# Display evaluation
if st.session_state.evaluation:
    evaluation = st.session_state.evaluation

    st.divider()
    st.subheader("Evaluation Results")

    col1, col2 = st.columns(2)
    col1.metric("Score", f'{evaluation["score"]}/10')
    col2.metric("Verdict", evaluation["verdict"])

    st.markdown("**Feedback**")
    st.write(evaluation["feedback"])

    st.markdown("**What you did well**")
    for item in evaluation.get("strengths", []):
        st.write(f"• {item}")

    st.markdown("**Concepts to improve**")
    missing = evaluation.get("missing_concepts", [])
    if missing:
        for item in missing:
            st.write(f"• {item}")
    else:
        st.write("No major missing concepts identified.")

    st.markdown("**Sample Answer**")
    st.write(evaluation["sample_answer"])

# Interview summary
if st.session_state.history:
    st.divider()
    st.subheader("Interview Performance Summary")

    scores = [item["score"] for item in st.session_state.history]
    average = sum(scores) / len(scores)

    col1, col2, col3 = st.columns(3)
    col1.metric("Questions Attempted", len(scores))
    col2.metric("Average Score", f"{average:.1f}/10")
    col3.metric("Highest Score", f"{max(scores)}/10")

    st.dataframe(
        [
            {
                "Question": item["question"],
                "Score": f'{item["score"]}/10',
                "Verdict": item["verdict"]
            }
            for item in st.session_state.history
        ],
        use_container_width=True
    )
