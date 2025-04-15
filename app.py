import streamlit as st
import os
import pandas as pd
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import PyPDF2
import time
import random
from datetime import datetime, timedelta
import requests
import tempfile
import edge_tts
import asyncio
import base64
import time
import tempfile
import asyncio

try:
    from mutagen.mp3 import MP3
    mutagen_available = True
except ImportError:
    mutagen_available = False

def speak_text(text, voice="alpha"):
    voice_map = {
        "alpha": "en-US-JennyNeural",
        "beta": "en-GB-RyanNeural",
    }
    real_voice = voice_map.get(voice, "en-US-JennyNeural")

    async def run_tts():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            communicate = edge_tts.Communicate(text, real_voice)
            await communicate.save(tmpfile.name)

            with open(tmpfile.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
                b64_audio = base64.b64encode(audio_bytes).decode()

            st.markdown(f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)

            # Estimate or get actual duration
            duration = get_mp3_duration(tmpfile.name, text)
            time.sleep(duration)

    asyncio.run(run_tts())

def get_mp3_duration(file_path, text):
    if mutagen_available:
        audio = MP3(file_path)
        return audio.info.length
    else:
        # Basic estimate: ~2.5 words per second
        words = len(text.split())
        return max(words / 2.5, 2)  # minimum 2 seconds just in case


def get_user_ip_ad():
     try:
         response = requests.get('https://api.ipify.org?format=json', timeout=2)
         return response.json().get("ip", "")
     except Exception:
         return ""
 
def is_csusb(ip):
     return any([
         ip.startswith("138.23."),
         ip.startswith("139.182.")
     ])

def generate_alpha_question_intro(q_num, question):
    if q_num == 0:
        starters = [
            "Alright Beta, let's kick things off—",
            "Okay, first up—",
            "Let's dive into it—",
            "To get us started—",
            "Kicking things off, Beta—"
        ]
    else:
        starters = [
            f"Question {q_num + 1} coming at you—",
            "Alright, next up—",
            "Let's keep it rolling—",
            "Here's another one—",
            "This one's interesting—"
        ]
    return random.choice(starters) + question

st.set_page_config(
    page_title="CSUSB Study Podcast",
    page_icon="logo/csusb_logo.png",
)

# Custom CSS
st.markdown("""
<style>
div[data-testid="stFileUploader"] div[aria-live="polite"] {
    display: none !important;
}
.stButton button { width: 100%; }
.submit-button-container {
    display: flex;
    align-items: flex-end;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
.stTextInput > div > div > input {
    padding-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    col_img, col_text = st.columns([0.2, 1])
    with col_img:
        st.image("logo/csusb_logo.png", width=60)
    with col_text:
        st.markdown("<h3 style='font-size: 22px; margin: 0px;'>CSUSB Study Podcast Assistant</h3>", unsafe_allow_html=True)

apik = os.getenv("GROQ_API_KEY")
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()

chat_alpha = init_chat_model("llama3-8b-8192", model_provider="groq")
chat_beta = init_chat_model("llama3-70b-8192", model_provider="groq")
messages = [SystemMessage(content="You are an AI assistant that will help.")]

if "podcast_started" not in st.session_state:
    st.session_state["podcast_started"] = False
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])
if "last_upload_time" not in st.session_state:
    st.session_state["last_upload_time"] = None

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

uploaded_file = None
user_ip = get_user_ip_ad()


if not is_csusb(user_ip):
    st.warning("Access denied")
    st.stop()
else:
    uploaded_file = st.file_uploader("Upload a PDF document (Max: 10MB)", type=["pdf"])


if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])

def update_sidebar():
    with st.sidebar:
        st.markdown("### Confusion Matrix")
        st.write(pd.DataFrame(st.session_state.conf_matrix, 
                              columns=["Answered Correctly", "Not Answered"],
                              index=["Actual Yes", "Actual No"]))

        tp, fn = st.session_state.conf_matrix[0, 0], st.session_state.conf_matrix[1, 0]
        fp = st.session_state.conf_matrix[0, 1] if 0 in st.session_state.conf_matrix.shape else 0
        tn = st.session_state.conf_matrix[1, 1] if 1 in st.session_state.conf_matrix.shape else 0

        total = tp + tn + fp + fn
        model_accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

        st.markdown("### Model Metrics")
        st.write(f"**Model Accuracy:** {model_accuracy:.2f}%")
        st.write(f"**Precision:** {precision:.2f}%")
        st.write(f"**Recall (Sensitivity):** {recall:.2f}%")
        st.write(f"**Specificity:** {specificity:.2f}%")
        st.write(f"**F1 Score:** {f1_score:.2f}%")

        if st.button("🔄 Reset Metrics", key="reset_button"):
            st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])
            st.session_state["podcast_started"] = False
            st.success("Confusion Matrix & Model Metrics Reset!")
            st.experimental_rerun()

# AI Podcast Function (Modified to Use Uploaded Document) 
import time

def start_ai_podcast():
    st.markdown("## 🎙️ Welcome to the AI Podcast")

    messages.clear()
    start_time = time.time()
    max_duration = 180  # Total duration: 3 minutes

    # Load PDF chunks by page for question variety
    from PyPDF2 import PdfReader
    try:
        reader = PdfReader(uploaded_file)
        chunks = [page.extract_text() or "" for page in reader.pages if page.extract_text()]
    except Exception as e:
        st.error(f"Failed to extract document chunks: {e}")
        return

    if not chunks:
        st.warning("No readable content found in the uploaded PDF.")
        return

    chunk_index = 0

    def time_left():
        return max_duration - (time.time() - start_time)

    # Alpha's short intro
    intro_prompt = """
    You're Alpha, the podcast host. Generate a short and friendly podcast intro (1–2 sentences max) that:
    - Greets the audience
    - Introduces yourself as Alpha
    - Says you're discussing something interesting with Beta (no placeholder text like 'Insert Topic Name')
    Keep it casual and fun, like a real chill podcast.
    """
    intro = chat_alpha.invoke([HumanMessage(content=intro_prompt)]).content.strip()
    st.markdown(f"**Alpha:** {intro}")
    speak_text(intro, voice="alpha")

    st.markdown("---")
    time.sleep(0.2)

    q_num = 0
    while time_left() > 15:
        if time_left() < 15:
            break

        context = chunks[chunk_index][:1000]

        question_prompt = f"""
        You're Alpha, a podcast host. Based on the document section below, ask Beta a short, casual podcast-style question. Be specific.

        Document:
        {context}
        """
        question = chat_alpha.invoke([HumanMessage(content=question_prompt)]).content.strip()
        question = question.replace("→", "").replace("Alpha:", "").strip()

        alpha_q = generate_alpha_question_intro(q_num, question)
        st.markdown(f"**Alpha:** {alpha_q}")
        speak_text(alpha_q, voice="alpha")

        time.sleep(0.2)
        if time_left() < 10:
            break

        beta_prompt = f"""
        You're Beta, a podcast co-host. Respond to Alpha’s question using ONLY the document section below. If it's not there, say "I don't know." Keep your response brief and natural (2–3 lines).

        Document:
        {context}

        Question:
        {question}
        """
        ai_response = chat_beta.invoke([SystemMessage(content=beta_prompt)]).content.strip()

        # 🎙️ Make Beta's "I don't know" sound natural and casual
        if ai_response.lower() in ["i don't know", "i do not know", "i'm not sure"]:
            beta_natural_responses = [
                "Hmm, I'm not quite sure about that one.",
                "That's a bit outside my scope, Alpha!",
                "Good question, but I don’t think the document covers that.",
                "I wish I had the answer, but I don't have enough info from the doc."
            ]
            ai_response = random.choice(beta_natural_responses)

        messages.append(AIMessage(content=ai_response))

        st.markdown(f"**Beta:** {ai_response}")
        speak_text(ai_response, voice="beta")


        time.sleep(0.2)
        
        # 🎤 If Beta didn't really know the answer, Alpha reacts more casually
        if any(x in ai_response.lower() for x in [
            "i'm not quite sure", "i don't have", "outside my scope", "wish i had the answer"
        ]):
            alpha_follow_up = random.choice([
                "No problem, let's move on to something else.",
                "All good, Beta — not everything has a clear answer!",
                "Fair enough, let’s try something more direct.",
                "Haha, that’s okay — let’s switch gears!"
            ])
        else:
            follow_up_prompt = f"""
            You're Alpha, the podcast host. React briefly (1 casual sentence) to Beta’s answer.

            Beta said: {ai_response}
            """
            alpha_follow_up = chat_alpha.invoke([HumanMessage(content=follow_up_prompt)]).content.strip()

        messages.append(HumanMessage(content=alpha_follow_up))

        st.markdown(f"**Alpha:** {alpha_follow_up}")
        speak_text(alpha_follow_up, voice="alpha")

        st.markdown("---")
        time.sleep(0.2)
        q_num += 1
        chunk_index = (chunk_index + 1) % len(chunks)

    outro_prompt = """
    You're Alpha, the podcast host. End the podcast with a short outro (3-4 sentences), thanking Beta and the audience in a friendly tone.
    """
    outro = chat_alpha.invoke([HumanMessage(content=outro_prompt)]).content.strip()
    st.markdown(f"**Alpha:** {outro}")
    speak_text(outro, voice="alpha")


if uploaded_file:
    current_time = datetime.now()

    # ⛔ Block if last podcast was completed less than 2 minutes ago
    if (
        "last_upload_time" in st.session_state and
        st.session_state["last_upload_time"] is not None and
        current_time - st.session_state["last_upload_time"] < timedelta(minutes=2)
    ):
        wait_time = timedelta(minutes=2) - (current_time - st.session_state["last_upload_time"])
        st.warning(f"⚠️ Please wait {int(wait_time.total_seconds() // 60)} min {int(wait_time.total_seconds() % 60)} sec before uploading another file.")
        uploaded_file = None  # Ignore this file
    elif uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File size exceeds the 10MB limit. Please upload a smaller PDF.")
        uploaded_file = None
    else:
        # ✅ Extract the text but don't start podcast yet
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF uploaded successfully!")


# Function to test AI rephrasing and answering
def test_ai_rephrasing():
    if not uploaded_file or not extracted_text:
        st.warning("Please upload a PDF document before testing the AI.")
        return

    # 5 questions that CAN be answered based on the document
    answerable_questions = [
        "Summarize a key concept mentioned early in the document.",
        "What topic does the document mainly discuss?",
        "Name one specific detail or statistic mentioned in the document.",
        "What is one recommendation or conclusion from the document?",
        "Identify one author or source cited in the document."
    ]

    # 5 questions that are clearly unrelated and should not be answerable
    unanswerable_questions = [
        "What’s the capital of Iceland?",
        "Who won the Super Bowl in 2024?",
        "What’s the weather like in San Bernardino today?",
        "Who is the president of CSUSB?",
        "What’s the square root of 1,024?"
    ]

    all_questions = answerable_questions + unanswerable_questions
    random.shuffle(all_questions)

    for i, question in enumerate(all_questions):
        st.markdown(f"**Alpha:** {question}")
        speak_text(question, voice="alpha")

        if question in answerable_questions:
            beta_prompt = f"""
            You are Beta, a podcast co-host. Answer the question below using ONLY the uploaded document. If the document does not include an answer, say "I don't know."

            Document:
            {extracted_text[:4000]}

            Question: {question}
            """
        else:
            beta_prompt = f"""
            You are Beta, a podcast co-host. You can ONLY answer questions using the uploaded document. If the document does not provide the answer, say "I don't know."

            Document:
            {extracted_text[:4000]}

            Question: {question}
            """

        response = chat_beta.invoke([SystemMessage(content=beta_prompt)])
        ai_response = response.content.strip() if response else "I don't know"

        st.markdown(f"**Beta:** {ai_response}")
        speak_text(ai_response, voice="beta")

        # Update confusion matrix
        if "I don't know" in ai_response:
            st.session_state.conf_matrix[1, 1] += 1
        else:
            st.session_state.conf_matrix[0, 0] += 1

    update_sidebar()
      
# Create three buttons in separate columns above the text input box
col1, col2, col3 = st.columns(3)

user_ip = get_user_ip_ad()
if not is_csusb(user_ip):
    st.warning("Access denied")
    st.stop()
else:
    st.success("Welcome CSUSB User!")


with col1:
    if st.button(":material/voice_chat: Start AI Podcast", key="start_podcast_button_1"):
        if not uploaded_file:
            st.warning("Please upload a PDF before starting the podcast.")
        else:
            st.session_state["podcast_started"] = True
            st.session_state["last_upload_time"] = datetime.now()  # start cooldown after it runs

with col2:
    if st.button(":material/bug_report: Test AI", key="test_ai_button"):
        st.session_state["test_ai_clicked"] = True

# Chatbot Input for Document-based or Model-based Communication
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Ask the Chatbot a Question (Document-based or Model-based)", key="chat_input")
with col2:
    st.markdown('<div class="submit-button-container">', unsafe_allow_html=True)
    if st.button(":material/send: Submit", key="submit_button_1"):
        if extracted_text:
            prompt = f"""
            You are an AI assistant. The following is the content extracted from a document uploaded by the user. Use this extracted text to respond to the user's queries.

            Extracted Document Text:
            {extracted_text[:1000]}

            **User Query:** {user_input}

            Provide an accurate response in the requested format.
            """
        else:
            prompt = f"""
            You are an AI assistant with general knowledge. Please respond to the user's query based on your pre-trained knowledge.
            
            **User Query:** {user_input}

            Provide an accurate response in the requested format.
            """

        response = chat_alpha.invoke([SystemMessage(content=prompt)]) if uploaded_file else chat_beta.invoke([SystemMessage(content=prompt)])
        ai_response = response.content if response else "I do not know!"

        # Save the response into session_state so we can show it outside col2
        st.session_state.last_question = user_input
        st.session_state.last_answer = ai_response

        # Update confusion matrix
        if "I do not know!" in ai_response:
            st.session_state.conf_matrix[1, 0] += 1
        else:
            st.session_state.conf_matrix[0, 0] += 1

    st.markdown('</div>', unsafe_allow_html=True)

# 👇 Show the response in the main body (not col2)
if "last_question" in st.session_state and "last_answer" in st.session_state:
    st.markdown("### Chatbot Response")
    st.markdown(f"**🧑 User:** {st.session_state.last_question}")
    st.markdown(f"**🤖 AI:** {st.session_state.last_answer}")

# Unified Output Section (Ensuring Output Appears Below in a Single Row)
if st.session_state["podcast_started"]:
    st.write("---")  # Separator for clarity
    start_ai_podcast()  # Run the AI podcast function

if st.session_state.get("test_ai_clicked"):
    st.write("---")  # Separator for clarity
    test_ai_rephrasing()  # Run the AI rephrasing and answering function
