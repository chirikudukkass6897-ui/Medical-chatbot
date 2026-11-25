import os
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import fitz
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
load_dotenv()

APP_NAME = "DIAGNOBOT"
DB_NAME = "diagnobot.db"

# Load API key from Streamlit Cloud Secrets or local .env
try:
    API_KEY = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

# Model configuration
try:
    MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --------------------------------------------------
# MEDICAL SAFETY
# --------------------------------------------------

EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "severe bleeding",
    "unconscious",
    "not waking up",
    "stroke",
    "face drooping",
    "slurred speech",
    "seizure",
    "overdose",
    "severe allergic reaction",
    "suicidal",
    "suicide",
    "severe headache",
    "blue lips"
]

def emergency_check(text):

    text = text.lower()

    for keyword in EMERGENCY_KEYWORDS:

        if keyword in text:

            return True

    return False


def emergency_message(language="English"):

    if language == "Malayalam":

        return """
🚨 **അടിയന്തര മുന്നറിയിപ്പ്**

നിങ്ങൾ വിവരിച്ച ലക്ഷണങ്ങൾ അടിയന്തര
വൈദ്യസഹായം ആവശ്യമായ അവസ്ഥയെ
സൂചിപ്പിച്ചേക്കാം.

ദയവായി ഉടൻ അടുത്തുള്ള ആശുപത്രിയിലെ
എമർജൻസി വിഭാഗത്തെ സമീപിക്കുക.

ഇന്ത്യയിൽ അടിയന്തര സഹായത്തിന് **112**
വിളിക്കുക.

AI പ്രതികരണത്തിനായി കാത്തിരിക്കരുത്.
"""

    return """
🚨 **POTENTIAL MEDICAL EMERGENCY**

Your symptoms may require immediate medical attention.

- Call emergency services immediately.
- In India, call **112**.
- Go to the nearest emergency department.
- Do not wait for a chatbot response.

This automated warning is not a validated
emergency detection system.
"""


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def initialize_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_notes (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date TEXT,

            category TEXT,

            details TEXT

        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date TEXT,

            question TEXT,

            answer TEXT

        )
    """)

    conn.commit()

    conn.close()


def save_note(category, details):

    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO health_notes
        (date, category, details)
        VALUES (?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        category,
        details
    ))

    conn.commit()

    conn.close()


def save_chat(question, answer):

    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO chat_history
        (date, question, answer)
        VALUES (?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        question,
        answer
    ))

    conn.commit()

    conn.close()


def get_notes():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        "SELECT * FROM health_notes ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


def get_chat_history():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        "SELECT * FROM chat_history ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


initialize_database()


# --------------------------------------------------
# AI CLIENT
# --------------------------------------------------
def get_ai_client():
    if not API_KEY:
        return None

    try:
        return OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error(f"OpenAI connection error: {e}")
        return None


SYSTEM_PROMPT = """
You are DIAGNOBOT, a responsible medical information chatbot.

Your purpose:
- Explain general medical topics.
- Help users understand symptoms.
- Explain medical reports in simple language.
- Explain general medicine information.
- Provide preventive health education.
- Help users prepare questions for doctors.

STRICT SAFETY RULES:

1. Never claim to provide a definitive diagnosis.
2. Never prescribe medications.
3. Never recommend starting, stopping or changing
   prescription medication.
4. Do not invent medical test results.
5. Clearly explain uncertainty.
6. Recommend professional consultation when appropriate.
7. If symptoms are severe, worsening or concerning,
   recommend medical attention.
8. For emergencies, advise immediate emergency services.
9. Do not fabricate references or scientific evidence.
10. Do not ask users to share unnecessary personal data.
11. Interpret reports cautiously. Reference ranges vary.
12. Explain medical information in simple language.
13. Do not claim a report proves a particular disease.
14. You are an educational assistant, not a doctor.
"""


def ask_ai(question, language="English", context=None):

    client = get_ai_client()

    if client is None:

        return """
⚠️ AI API is not configured.

Please add your API key to the `.env` file.

The chatbot interface and other basic modules
can still be used without the AI API.
"""

    messages = [

        {
            "role": "system",

            "content": SYSTEM_PROMPT +
            f"\nRespond in {language}."
        },

        {
            "role": "user",

            "content:": question
        }
    ]

    # Correct message structure
    messages[-1] = {
        "role": "user",
        "content": question
    }

    if context:

        messages.append({

            "role": "user",

            "content": (
                "Additional information to explain:\n"
                + context
            )

        })

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=0.2

        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI API Error: {type(e).__name__}: {e}"


# --------------------------------------------------
# REPORT ANALYSIS
# --------------------------------------------------

def extract_pdf_text(uploaded_file):

    try:

        pdf = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        text = ""

        for page in pdf:

            text += page.get_text() + "\n"

        return text

    except Exception:

        return ""


# --------------------------------------------------
# SYMPTOM CHECKER
# --------------------------------------------------

def analyze_symptoms(symptoms, duration, age_group, language):

    if emergency_check(symptoms):

        return emergency_message(language)

    prompt = f"""
    A user has described the following symptoms.

    Symptoms: {symptoms}

    Duration: {duration}

    Age group: {age_group}

    Provide a cautious educational response.

    Format your response as:

    1. Summary of reported symptoms
    2. Common possible explanations (not diagnosis)
    3. Questions the user may want to consider
    4. General self-care information, if appropriate
    5. When professional medical care is recommended
    6. Warning signs requiring urgent care

    Do not diagnose or prescribe.
    """

    return ask_ai(prompt, language)


# --------------------------------------------------
# MEDICINE INFORMATION
# --------------------------------------------------

def medicine_information(medicine, language):

    prompt = f"""
    Explain general educational information about:

    Medicine name: {medicine}

    Include:

    - Common uses
    - Common side effects
    - General precautions
    - Possible allergy concerns
    - Why professional advice matters
    - Important safety considerations

    Do not recommend a dosage.
    Do not prescribe.
    Do not recommend starting or stopping medication.
    Explain that the user should follow their clinician
    and the official medication leaflet.
    """

    return ask_ai(prompt, language)


# --------------------------------------------------
# REPORT EXPLAINER
# --------------------------------------------------

def explain_report(report_text, language):

    prompt = """
    Explain this medical report in simple language.

    Identify test names, values, units and reference ranges
    if clearly available in the supplied text.

    Explain:
    - What each test generally measures
    - Which values appear outside the stated reference range
    - Why context matters
    - Questions to ask a healthcare professional

    Do not diagnose.
    Do not invent missing values.
    Do not assume reference ranges.
    Do not claim that a result confirms a disease.

    REPORT:
    """

    return ask_ai(
        prompt,
        language,
        context=report_text
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🩺 DIAGNOBOT")

    st.caption("AI Medical Information Assistant")

    language = st.selectbox(
        "Language",
        ["English", "Malayalam"]
    )

    st.divider()

    st.markdown("### Medical Safety")

    st.warning(
        "This application is not a medical "
        "diagnostic tool."
    )

    st.markdown(
        "Emergency number in India: **112**"
    )

    st.divider()

    st.markdown("### AI Status")

    if API_KEY:

        st.success("AI API configured")

    else:

        st.error("AI API not configured")


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.title("DIAGNOBOT")
st.subheader("Your AI Healthcare Information Assistant")

st.markdown("""
Understand your health information, explore symptoms,
learn about medicines and prepare for medical consultations.
""")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "💬 AI Chat",

    "🩹 Symptoms",

    "📄 Medical Reports",

    "💊 Medicines",

    "📊 Dashboard",

    "📚 Health History"

])


# ==================================================
# TAB 1 — AI CHATBOT
# ==================================================

with tab1:

    st.header("Medical AI Chat")

    st.info(
        "Ask general medical questions. "
        "Avoid sharing identifiable personal information."
    )

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask DIAGNOBOT a medical question..."
    )

    if user_question:

        with st.chat_message("user"):

            st.markdown(user_question)

        st.session_state.chat_messages.append({

            "role": "user",

            "content": user_question

        })

        with st.chat_message("assistant"):

            if emergency_check(user_question):

                answer = emergency_message(language)

            else:

                answer = ask_ai(
                    user_question,
                    language
                )

            st.markdown(answer)

        st.session_state.chat_messages.append({

            "role": "assistant",

            "content": answer

        })

        save_chat(user_question, answer)


# ==================================================
# TAB 2 — SYMPTOM CHECKER
# ==================================================

with tab2:

    st.header("AI Symptom Checker")

    st.caption(
        "This module provides educational guidance, "
        "not a diagnosis."
    )

    symptoms = st.text_area(

        "Describe your symptoms",

        placeholder=(
            "Example: headache, mild fever "
            "and tiredness..."
        ),

        height=120

    )

    duration = st.selectbox(

        "Duration",

        [
            "Less than 24 hours",
            "1–3 days",
            "4–7 days",
            "More than a week",
            "Not sure"
        ]

    )

    age_group = st.selectbox(

        "Age group",

        [
            "Under 5",
            "5–17",
            "18–39",
            "40–64",
            "65+",
            "Prefer not to say"
        ]

    )

    if st.button(
        "Analyze Symptoms",
        type="primary"
    ):

        if not symptoms.strip():

            st.error("Please enter symptoms.")

        elif emergency_check(symptoms):

            st.error(
                emergency_message(language)
            )

        else:

            with st.spinner(
                "Reviewing your symptoms..."
            ):

                result = analyze_symptoms(
                    symptoms,
                    duration,
                    age_group,
                    language
                )

            st.markdown(result)

            save_note(
                "Symptom Assessment",
                f"""
                Symptoms: {symptoms}

                Duration: {duration}

                Age group: {age_group}

                """
            )


# ==================================================
# TAB 3 — MEDICAL REPORT ANALYZER
# ==================================================

with tab3:

    st.header("Medical Report Explainer")

    st.write(
        "Upload a text-based medical PDF to "
        "understand the terminology."
    )

    uploaded_report = st.file_uploader(

        "Upload medical report",

        type=["pdf", "txt"]

    )

    if uploaded_report:

        if uploaded_report.name.endswith(".txt"):

            report_text = uploaded_report.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        else:

            report_text = extract_pdf_text(
                uploaded_report
            )

        if report_text.strip():

            with st.expander("Extracted text"):

                st.text(report_text[:15000])

            if st.button(
                "Explain My Report",
                type="primary"
            ):

                with st.spinner(
                    "Explaining report..."
                ):

                    result = explain_report(
                        report_text[:15000],
                        language
                    )

                st.markdown(result)

                save_note(
                    "Medical Report",
                    report_text[:10000]
                )

        else:

            st.warning(
                "Unable to extract text. "
                "This may be a scanned PDF."
            )


# ==================================================
# TAB 4 — MEDICINE INFORMATION
# ==================================================

with tab4:

    st.header("Medicine Information Assistant")

    st.caption(
        "Educational information only. "
        "Not a prescribing tool."
    )

    medicine_name = st.text_input(
        "Enter medicine name"
    )

    if st.button("Get Medicine Information"):

        if medicine_name.strip():

            if emergency_check(medicine_name):

                st.error(
                    emergency_message(language)
                )

            else:

                with st.spinner(
                    "Searching for general information..."
                ):

                    result = medicine_information(
                        medicine_name,
                        language
                    )

                st.markdown(result)

        else:

            st.warning(
                "Please enter a medicine name."
            )


# ==================================================
# TAB 5 — HEALTH DASHBOARD
# ==================================================

with tab5:

    st.header("Health Information Dashboard")

    notes = get_notes()

    history = get_chat_history()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Saved Health Notes",
        len(notes)
    )

    col2.metric(
        "Chat Conversations",
        len(history)
    )

    col3.metric(
        "Report / Symptom Records",
        len(notes)
    )

    if not notes.empty:

        st.subheader("Health Note Categories")

        chart = px.pie(

            notes,

            names="category",

            title="Saved Health Information"

        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        st.subheader("Recent Health Records")

        st.dataframe(
            notes[["date", "category"]],
            use_container_width=True
        )

    else:

        st.info(
            "Your health dashboard will display "
            "information after records are saved."
        )


# ==================================================
# TAB 6 — HEALTH HISTORY
# ==================================================

with tab6:

    st.header("Saved Health History")

    notes = get_notes()

    if notes.empty:

        st.info("No records available.")

    else:

        for _, row in notes.iterrows():

            with st.expander(
                f"{row['date']} — {row['category']}"
            ):

                st.write(row["details"])

    st.divider()

    st.subheader("Chat History")

    history = get_chat_history()

    if history.empty:

        st.info("No previous conversations.")

    else:

        for _, row in history.iterrows():

            with st.expander(
                f"{row['date']} — {row['question'][:70]}"
            ):

                st.markdown("**Question:**")

                st.write(row["question"])

                st.markdown("**Answer:**")

                st.write(row["answer"])


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption("""
DIAGNOBOT is an educational prototype.

AI-generated medical information may be incomplete or inaccurate.
It does not replace a qualified healthcare professional.
Do not delay emergency care to use this application.
""")
