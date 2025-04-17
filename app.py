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

try:
    from mutagen.mp3 import MP3
    mutagen_available = True
except ImportError:
    mutagen_available = False

uploaded_file = None  # initialize early
extracted_text = ""

cooldown_active = False


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

            duration = get_mp3_duration(tmpfile.name, text)
            time.sleep(duration)

    asyncio.run(run_tts())


def get_mp3_duration(file_path, text):
    if mutagen_available:
        audio = MP3(file_path)
        return audio.info.length
    else:
        words = len(text.split())
        return max(words / 2.5, 2)


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


st.set_page_config(
    page_title="CSUSB Study Podcast",
    page_icon="logo/csusb_logo.png",
)

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

apik = os.environ["GROQ_API_KEY"]
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()

chat_alpha = init_chat_model("llama3-8b-8192", model_provider="groq")
chat_beta = init_chat_model("llama3-70b-8192", model_provider="groq")

if "podcast_started" not in st.session_state:
    st.session_state["podcast_started"] = False
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])
if "last_upload_time" not in st.session_state:
    st.session_state["last_upload_time"] = None
if "show_podcast_warning" not in st.session_state:
    st.session_state["show_podcast_warning"] = False
if "show_test_warning" not in st.session_state:
    st.session_state["show_test_warning"] = False


def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


user_ip = get_user_ip_ad()
if not is_csusb(user_ip):
    st.warning("Access denied")
    st.stop()
else:
    st.success("Welcome, CSUSB User!")

    current_time = datetime.now()
    if (
        st.session_state["last_upload_time"] is not None and
        current_time - st.session_state["last_upload_time"] < timedelta(minutes=5)
    ):
        wait_time = timedelta(minutes=5) - (current_time - st.session_state["last_upload_time"])
        cooldown_active = True
        countdown_placeholder = st.empty()
        while wait_time.total_seconds() > 0:
            minutes, seconds = divmod(int(wait_time.total_seconds()), 60)
            countdown_placeholder.warning(f"⏳ Upload locked. Please wait {minutes:02d}:{seconds:02d} to upload a new document.")
            time.sleep(1)
            wait_time -= timedelta(seconds=1)
        countdown_placeholder.empty()
        st.rerun()
    else:
        potential_file = st.file_uploader("Upload a PDF document (Max: 10MB)", type=["pdf"])
        if potential_file:
            if potential_file.size > 10 * 1024 * 1024:
                st.error("❌ File size exceeds the 10MB limit. Please upload a smaller PDF.")
            else:
                uploaded_file = potential_file
                extracted_text = extract_text_from_pdf(uploaded_file)
                st.success("✅ PDF uploaded successfully!")

start_clicked = st.button(":material/voice_chat: Start AI Podcast", key="start_podcast_button_1")

if start_clicked:
    st.session_state["show_test_warning"] = False
    if not uploaded_file:
        st.session_state["show_podcast_warning"] = True
    else:
        st.session_state["podcast_started"] = True
        st.session_state["last_upload_time"] = datetime.now()
        st.session_state["show_podcast_warning"] = False

if st.session_state["show_podcast_warning"]:
    st.warning("🚨 Please upload a PDF before starting the podcast.")

if cooldown_active:
    uploaded_file = None
    extracted_text = ""


def start_ai_podcast():
    if not uploaded_file:
        st.warning("🛑 PDF has been removed. Ending podcast.")
        st.session_state["podcast_started"] = False
        return

    st.markdown("## 🎙️ Welcome to the AI Podcast")
    messages = []
    start_time = time.time()
    max_duration = 180

    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        chunks = [page.extract_text() or "" for page in reader.pages if page.extract_text()]
    except Exception as e:
        st.warning("🛑 PDF is no longer available or readable. Ending podcast.")
        st.session_state["podcast_started"] = False
        return

    if not chunks:
        st.warning("No readable content found in the uploaded PDF.")
        return

    chunk_index = 0
    context = extracted_text[:4000]

    intro_prompt = f"""
    You're Alpha, kicking off a casual podcast with your co-host Beta.
    Say hi, drop your name, and mention you're diving into a doc together.
    Keep it under 2 sentences. Make it relaxed, friendly, not robotic.

    Document:
    {context}
    """
    intro = chat_alpha.invoke([HumanMessage(content=intro_prompt)]).content.strip()
    st.markdown(f"**Alpha:** {intro}")
    speak_text(intro, voice="alpha")

    st.markdown("---")
    last_beta_response = None

    while time.time() - start_time < max_duration:
        context = chunks[chunk_index][:4000]

        if last_beta_response:
            tone = random.choice(["curious", "amused", "sarcastic", "chill", "deep thinker"])
            alpha_prompt = f"""
            You're Alpha, a podcast host with a {tone} vibe.
            React to what Beta just said or ask a follow-up. Keep it smooth and natural.
            Avoid names and labels.
            Be very brief, three sentences max.

            Context:
            {context}

            Beta just said:
            {last_beta_response}
            """
        else:
            alpha_prompt = f"""
            You're Alpha, podcast host. Ask a quick, casual question based on the doc.
            Keep it real and tight, with short, quick, questions.

            Document:
            {context}
            """

        alpha_question = chat_alpha.invoke([HumanMessage(content=alpha_prompt)]).content.strip()
        st.markdown(f"**Alpha:** {alpha_question}")
        speak_text(alpha_question, voice="alpha")

        beta_prompt = f"""
        You're Beta, the podcast co-host. Answer Alpha's question based ONLY on the doc below.
        Keep it short, conversational. If unsure, say something casual like 'Not sure on that one.'

        Document:
        {context}

        Alpha asked:
        {alpha_question}
        """

        beta_response = chat_beta.invoke([SystemMessage(content=beta_prompt)]).content.strip()

        if beta_response.lower() in ["i don't know", "i do not know", "i'm not sure"]:
            beta_response = random.choice([
                "Hmm, I’m not totally sure about that.",
                "Yeah, that’s not really clear in the doc.",
                "Hard to say, honestly.",
                "Not sure on that one."
            ])

        st.markdown(f"**Beta:** {beta_response}")
        speak_text(beta_response, voice="beta")

        last_beta_response = beta_response
        chunk_index = (chunk_index + 1) % len(chunks)

    outro_prompt = f"""
    You're Alpha, the podcast host. End the podcast in one short sentence.
    Thank Beta and the audience, and casually mention the topic of the doc.
    Make it smooth, brief, and natural — max 20 words.

    Document context:
    {context[:2000]}
    """
    outro = chat_alpha.invoke([HumanMessage(content=outro_prompt)]).content.strip()

    st.markdown(f"**Alpha:** {outro}")
    speak_text(outro, voice="alpha")

    # Mark podcast as ended and trigger cooldown with adjusted time
    podcast_duration = time.time() - start_time
    cooldown_remaining = max(300 - podcast_duration, 0)  # 300 seconds = 5 minutes

    # Adjust last_upload_time so the remaining cooldown is honored  
    st.session_state["podcast_started"] = False
    st.session_state["last_upload_time"] = datetime.now() - timedelta(seconds=(300 - cooldown_remaining))
    st.rerun()

if st.session_state["podcast_started"]:
    st.write("---")
    start_ai_podcast()