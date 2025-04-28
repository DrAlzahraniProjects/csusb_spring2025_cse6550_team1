import streamlit as st
import os
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

CHUNK_SIZE = 1000  # characters
TOP_K = 3  # number of top matching chunks to pull
uploaded_file = None  # initialize early
extracted_text = ""

cooldown_active = False
test_mode = False


LOCK_FILE = "/tmp/streamlit_app.lock"
PODCAST_LOCK_TIMEOUT = 180  # 3 minutes
UPLOAD_COOLDOWN = 300       # 5 minutes
UPLOAD_TRACKER_DIR = "/tmp/upload_timestamps"

def is_locked():
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, "r") as f:
                timestamp = float(f.read().strip())
                if time.time() - timestamp < PODCAST_LOCK_TIMEOUT:
                    return True
                else:
                    os.remove(LOCK_FILE)
        return False
    except Exception as e:
        return False

def set_lock():
    with open(LOCK_FILE, "w") as f:
        f.write(str(time.time()))

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def is_upload_cooldown_active(ip):
    os.makedirs(UPLOAD_TRACKER_DIR, exist_ok=True)
    ip_file = os.path.join(UPLOAD_TRACKER_DIR, ip.replace(".", "_") + ".txt")

    if os.path.exists(ip_file):
        with open(ip_file, "r") as f:
            last_upload = float(f.read())
            if time.time() - last_upload < UPLOAD_COOLDOWN:
                return UPLOAD_COOLDOWN - (time.time() - last_upload)
    return 0

def update_upload_timestamp(ip):
    os.makedirs(UPLOAD_TRACKER_DIR, exist_ok=True)
    ip_file = os.path.join(UPLOAD_TRACKER_DIR, ip.replace(".", "_") + ".txt")
    with open(ip_file, "w") as f:
        f.write(str(time.time()))

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_chunks(question, chunks, top_k=3):
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    chunk_vectors = vectorizer.transform(chunks)
    question_vector = vectorizer.transform([question])
    sims = cosine_similarity(question_vector, chunk_vectors).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def speak_text(text, voice="Teacher"):
    voice_map = {
        "Teacher": "en-US-JennyNeural",
        "Student": "en-GB-RyanNeural",
    }
    real_voice = voice_map.get(voice, "en-US-JennyNeural")

    async def run_tts():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            communicate = edge_tts.Communicate(text, real_voice, rate="+10%")
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
    page_icon="./logo/csusb_logo.png",
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
        st.image("./logo/csusb_logo.png", width=60)
    with col_text:
        st.markdown("""
            <div style='line-height: 1.2;'>
                <h3 style='font-size: 22px; margin: 0px;'>CSUSB Study Podcast Assistant</h3>
                <p style='font-size: 14px; margin: 2px 0 0 0; color: gray;'>
                    Study Podcast Assistant that converts PDF documents into engaging AI conversations
                </p>
            </div>
        """, unsafe_allow_html=True)


apik = os.environ["GROQ_API_KEY"]

if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()

if "test_mode" not in st.session_state:
    test_mode = False
    if apik.endswith("_TEST"):
        test_mode = True
        os.environ["GROQ_API_KEY"] = apik[:-5]  # update env var to the clean key
    st.session_state["test_mode"] = test_mode
else:
    test_mode = st.session_state["test_mode"]

if "test_mode" not in st.session_state:
    st.info("🚀 Test Mode Active - No IP checks, manual Student questions enabled.")

chat_Teacher = init_chat_model("llama3-8b-8192", model_provider="groq")
chat_Student = init_chat_model("llama3-70b-8192", model_provider="groq")

if "podcast_started" not in st.session_state:
    st.session_state["podcast_started"] = False
if "last_upload_time" not in st.session_state:
    st.session_state["last_upload_time"] = None
if "show_podcast_warning" not in st.session_state:
    st.session_state["show_podcast_warning"] = False
if "show_test_warning" not in st.session_state:
    st.session_state["show_test_warning"] = False

user_ip = get_user_ip_ad()
if not test_mode:
    user_ip = get_user_ip_ad()
    if not is_csusb(user_ip):
        st.warning("Access denied")
        st.stop()

if not test_mode:
    cooldown_remaining = is_upload_cooldown_active(user_ip)
    if cooldown_remaining > 0:
        cooldown_active = True
        countdown_placeholder = st.empty()
        while cooldown_remaining > 0:
            minutes, seconds = divmod(int(cooldown_remaining), 60)
            countdown_placeholder.warning(f"\u23f3 Upload locked. Please wait {minutes:02d}:{seconds:02d} to upload a new document.")
            time.sleep(1)
            cooldown_remaining -= 1
        countdown_placeholder.empty()
        st.rerun()

if is_locked():
    placeholder = st.empty()
    with placeholder.container():
        waiting_text = st.empty()
        for i in range(PODCAST_LOCK_TIMEOUT):
            dots = '.' * ((i % 3) + 1)
            if uploaded_file is None:
                release_lock()
            waiting_text.warning(f"🚧 Server is busy. Waiting{dots}")
            time.sleep(1)
            if not is_locked():
                st.rerun()
        st.warning("Still busy. Try refreshing manually.")
    st.stop()

st.success("Welcome, CSUSB User!")

current_time = datetime.now()
if (
    st.session_state["last_upload_time"] is not None and
    current_time - st.session_state["last_upload_time"] < timedelta(seconds=UPLOAD_COOLDOWN)
):
    wait_time = timedelta(seconds=UPLOAD_COOLDOWN) - (current_time - st.session_state["last_upload_time"])
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

        progress = st.progress(0, text="Preparing to extract...")

        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(pdf_reader.pages)

            extracted_text = ""
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                extracted_text += text
                doc_chunks = chunk_text(extracted_text)

                progress.progress(int(((i + 1) / num_pages) * 100), text=f"Extracting page {i + 1} of {num_pages}...")

            progress.empty()
            st.success(f"✅ PDF extracted successfully with {num_pages} page(s)!")
        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")

start_clicked = st.button(":material/voice_chat: Start AI Podcast", key="start_podcast_button_1")

if test_mode:
    st.markdown("---")
    st.header("👨‍🏫 Test Mode Active: Ask the Student")

    if uploaded_file:
        if "test_session" not in st.session_state:
            st.session_state["test_session"] = []

        user_question = st.text_input("Enter your question for Student:")

        if st.button("Ask Student"):
            relevant_chunks = retrieve_chunks(user_question, doc_chunks)
            context = "\n\n".join(relevant_chunks)

            chat_Student = init_chat_model("llama3-70b-8192", model_provider="groq")

            student_prompt = f"""
            You are Student. Answer ONLY based on the document below.
            If the answer is unclear, say 'The document doesn't clearly say.'
            Prefer quoting the document where possible. No guessing allowed.

            Document:
            {context}

            Teacher asked:
            {user_question}
            """.strip()

            student_answer = chat_Student.invoke([SystemMessage(content=student_prompt)]).content.strip()

            if student_answer.lower() in ["i don't know", "i do not know", "i'm not sure"]:
                student_answer = random.choice([
                    "Hmm, not really sure.",
                    "Yeah, that’s kinda unclear.",
                    "Not sure on that one."
                ])

            st.session_state["test_session"].append((user_question, student_answer))

    if "test_session" in st.session_state:
        for q, a in st.session_state["test_session"]:
            st.markdown(f"**Teacher asked:** {q}")
            st.markdown(f"**Student answered:** {a}")

if start_clicked:
    st.session_state["show_test_warning"] = False
    if not uploaded_file:
        st.session_state["show_podcast_warning"] = True
    else:
        set_lock()
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
    update_upload_timestamp(user_ip)
    max_duration = 180

    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
        doc_chunks = chunk_text(extracted_text)
    except Exception as e:
        st.warning("🛑 PDF is no longer available or readable. Ending podcast.")
        st.session_state["podcast_started"] = False
        return

    if not doc_chunks:
        st.warning("No readable content found in the uploaded PDF.")
        return
    
    intro_context = extracted_text[:500]

    intro_prompt = f= f"""
            You're Teacher, opening a short 3-minute podcast. Greet Student quickly and dive into the topic.
            Mention that today's topic is based on an interesting paper.
            Keep it friendly and relaxed, no more than two sentences.
            Document:
            {intro_context}
            """.strip()
    
    intro = chat_Teacher.invoke([HumanMessage(content=intro_prompt)]).content.strip()
    st.markdown(f"**Teacher:** {intro}")
    speak_text(intro, voice="Teacher")

    chunk_index = 1

    good_chunks = [chunk for chunk in doc_chunks if len(chunk) > 400]
    context = good_chunks[chunk_index]

    st.markdown("---")
    last_Student_response = None

    while time.time() - start_time < max_duration:
        if last_Student_response:
            context = good_chunks[chunk_index]

            Teacher_prompt = f"""
            You're Teacher. Acknowledge Student's last answer briefly, then immediately pivot to a new topic from the document.
            Ask a direct and specific question based on the document content below.
            Keep it very short and casual. No long commentary
            Student said:
            {last_Student_response}
            Document:
            {context}
            """.strip()
        else:
            Teacher_prompt = f"""
            You are Teacher. You are already live on the podcast. 
            Immediately ask a direct, specific question based on the document below. 
            Do NOT say you are starting. 
            Keep it short, casual, and stay on topic.

            Document:
            {context}
            """.strip()

        Teacher_question = chat_Teacher.invoke([HumanMessage(content=Teacher_prompt)]).content.strip()
        st.markdown(f"**Teacher:** {Teacher_question}")
        speak_text(Teacher_question, voice="Teacher")

        student_context_chunks = retrieve_chunks(Teacher_question, good_chunks)
        student_context = "\n\n".join(student_context_chunks)


        Student_prompt = f"""
        You are Student. Answer ONLY based on the document below.
        Do NOT say phrases like 'according to the document' or 'the paper says'.
        Just answer naturally, like you know the material.
        If the answer is unclear, say 'The document doesn't clearly say.'
        Prefer quoting the document where possible. No guessing allowed.

        Document:
        {student_context}

        Teacher asked:
        {Teacher_question}
        """.strip()


        Student_response = chat_Student.invoke([SystemMessage(content=Student_prompt)]).content.strip()

        if Student_response.lower() in ["i don't know", "i do not know", "i'm not sure"]:
            Student_response = random.choice([
                "Hmm, I’m not totally sure about that.",
                "Yeah, that’s not really clear in the doc.",
                "Hard to say, honestly.",
                "Not sure on that one."
            ])

        st.markdown(f"**Student:** {Student_response}")
        speak_text(Student_response, voice="Student")

        last_Student_response = Student_response

        chunk_index = min(chunk_index + 1, len(doc_chunks) - 1)

    outro_prompt = f"""
    You're Teacher, the podcast host. End the podcast in one short sentence.
    Thank Student and the audience, and casually mention the topic of the doc.
    Make it smooth, brief, and natural — max 20 words.

    Document context:
    {context[:2000]}
    """
    outro = chat_Teacher.invoke([HumanMessage(content=outro_prompt)]).content.strip()

    st.markdown(f"**Teacher:** {outro}")
    speak_text(outro, voice="Teacher")

    # Mark podcast as ended and trigger cooldown with adjusted time
    podcast_duration = time.time() - start_time
    cooldown_remaining = max(300 - podcast_duration, 0)  # 300 seconds = 5 minutes

    # Adjust last_upload_time so the remaining cooldown is honored  
    st.session_state["podcast_started"] = False
    st.session_state["last_upload_time"] = datetime.now() - timedelta(seconds=(300 - cooldown_remaining))
    release_lock()
    st.rerun()

if st.session_state["podcast_started"]:
    st.write("---")
    start_ai_podcast()