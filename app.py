import streamlit as st
import speech_recognition as sr
import os
import pandas as pd
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import PyPDF2
import time
from streamlit_TTS import auto_play, text_to_audio  # ✅ Using streamlit-tts for real-time playback
from gtts.lang import tts_langs  # ✅ Importing available languages from gTTS
import random

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

# ✅ Function to Speak Text (Both Alpha & Beta Speak)
def speak_text(text, voice):
    # Set language/accent based on the voice
    language = "en" if voice == "alpha" else "en"  # Alpha uses US English, Beta uses British English
    # Generate audio for the text
    audio = text_to_audio(text, language=language)
    # Play the audio in the browser
    auto_play(audio)

# ✅ Update Streamlit App Header (Title, Icon, and Layout)
st.set_page_config(
    page_title="CSUSB Study Podcast",
    page_icon="logo/csusb_logo.png",
)

# Load custom CSS
st.markdown(
    """
    <style>
    div[data-testid="stFileUploader"] div[aria-live="polite"] {
        display: none !important;
    }

    .stButton button {
        width: 100%;
    }

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
    """,
    unsafe_allow_html=True
)

# Header Section
col1, col2, col3 = st.columns([1, 3, 1])  # Creates three equal columns for centering

with col2:  # Center content in the middle column
    col_img, col_text = st.columns([0.2, 1])  # Adjust spacing between logo & title
    with col_img:
        st.image("logo/csusb_logo.png", width=60)  # ✅ Local image method

    with col_text:
        st.markdown(
            "<h3 style='font-size: 22px; margin: 0px;'>CSUSB Study Podcast Assistant</h3>",
            unsafe_allow_html=True
        )
        
# Initialize API key
apik = os.getenv("GROQ_API_KEY")
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()
    
# Initialize two different Llama3 models
chat_alpha = init_chat_model("llama3-8b-8192", model_provider="groq")  # Alpha's model
chat_beta = init_chat_model("llama3-70b-8192", model_provider="groq")  # Beta's model
messages = [SystemMessage(content="You are an AI assistant that will help.")]

recognizer = sr.Recognizer()

# Function to listen to microphone input
def listen_to_microphone():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Sorry, I'm unable to process the request at the moment."


# Ensure session state is initialized
if "podcast_started" not in st.session_state:
    st.session_state["podcast_started"] = False

if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# File Upload Section (PDF Only, Max 10MB)
uploaded_file = st.file_uploader("Upload a PDF document (Max: 10MB)", type=["pdf"])

if uploaded_file:
    if uploaded_file.size > 10 * 1024 * 1024:  # Check if file exceeds 10MB
        st.error("❌ File size exceeds the 10MB limit. Please upload a smaller PDF.")
        uploaded_file = None  # Discard the file
    else:
        # Process the PDF
        extracted_text = extract_text_from_pdf(uploaded_file)


# Initialize confusion matrix
if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])


# ✅ Function to Display Confusion Matrix & Model Metrics
def update_sidebar():
    with st.sidebar:
        st.markdown("### Confusion Matrix")
        st.write(pd.DataFrame(st.session_state.conf_matrix, 
                              columns=["Answered Correctly", "Not Answered"],
                              index=["Actual Yes", "Actual No"]))

        # ✅ Extract and Display Updated Model Metrics
        tp = st.session_state.conf_matrix[0, 0]  # True Positives
        fn = st.session_state.conf_matrix[1, 0]  # False Negatives
        fp = st.session_state.conf_matrix[0, 1] if 0 in st.session_state.conf_matrix.shape else 0  # False Positives
        tn = st.session_state.conf_matrix[1, 1] if 1 in st.session_state.conf_matrix.shape else 0  # True Negatives

        total = tp + tn + fp + fn
        model_accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
        sensitivity = recall  # Sensitivity is the same as recall

        # ✅ Display Updated Model Metrics in Sidebar
        st.markdown("### Model Metrics")
        st.write(f"**Model Accuracy:** {model_accuracy:.2f}%")
        st.write(f"**Precision:** {precision:.2f}%")
        st.write(f"**Recall (Sensitivity):** {recall:.2f}%")
        st.write(f"**Specificity:** {specificity:.2f}%")
        st.write(f"**F1 Score:** {f1_score:.2f}%")

        # ✅ Reset Button to Clear Confusion Matrix and Model Metrics
        if st.button("🔄 Reset Metrics", key="reset_button"):
            st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])
            st.session_state["podcast_started"] = False
            st.success("Confusion Matrix & Model Metrics Reset!")
            st.experimental_rerun()  # Force UI refresh

# AI Podcast Function (Modified to Use Uploaded Document) 
import time
import time

def start_ai_podcast():
    st.markdown("## 🎙️ Welcome to the AI Podcast")

    messages.clear()
    start_time = time.time()
    max_duration = 180  # Total duration: 3 minutes

    # ⏱ Helper to check remaining time
    def time_left():
        return max_duration - (time.time() - start_time)

    # 🎙️ Alpha's intro
    intro_prompt = """
    You're Alpha, the podcast host. Generate a friendly, welcoming podcast intro (3-5 sentences) that:
    - Greets the audience
    - Introduces yourself as Alpha
    - Mentions that the podcast will be a discussion based on an uploaded document—without naming it
    - Welcomes your guest, Beta, the AI assistant
    Keep it casual and fun, like a chill podcast chat.
    """
    intro = chat_alpha.invoke([HumanMessage(content=intro_prompt)]).content.strip()
    alpha_placeholder = st.empty()
    alpha_placeholder.markdown(f"**Alpha:** {intro}")
    speak_text(intro, voice="alpha")

    st.markdown("---")
    time.sleep(0.2)

    q_num = 0
    while time_left() > 15:  # Save 15 seconds for outro
        # ⏰ Stop early if time is almost up
        if time_left() < 15:
            break

        # 💬 Alpha generates a question
        if extracted_text:
            context = extracted_text[:1000]
            question_prompt = f"""
            You're Alpha, a podcast host. Based on the document below, generate a short, casual podcast-style question (1–2 lines). 
            Do NOT include explanations—just the question.

            Document:
            {context}
            """
        else:
            question_prompt = "Generate a short, casual podcast-style question Alpha can ask Beta about CSUSB-related topics."

        question = chat_alpha.invoke([HumanMessage(content=question_prompt)]).content.strip()
        question = question.replace("→", "").replace("Alpha:", "").strip()

        # 🎙️ Alpha asks the question
        alpha_q = generate_alpha_question_intro(q_num, question)
        alpha_placeholder = st.empty()
        alpha_placeholder.markdown(f"**Alpha:** {alpha_q}")
        speak_text(alpha_q, voice="alpha")

        time.sleep(0.2)
        if time_left() < 10:
            break

        # 🧠 Beta responds shortly
        beta_prompt = f"""
        You're Beta, a podcast co-host. Respond to Alpha’s question using the document below. Your answer should be short and natural: no more than 2-3 lines.

        Document:
        {extracted_text[:1000]}
        Question:
        {question}
        """
        ai_response = chat_beta.invoke([SystemMessage(content=beta_prompt)]).content.strip()
        messages.append(AIMessage(content=ai_response))

        beta_placeholder = st.empty()
        beta_placeholder.markdown(f"**Beta:** {ai_response}")
        speak_text(ai_response, voice="beta")

        time.sleep(0.2)
        if time_left() < 10:
            break

        # 🎤 Alpha follow-up
        follow_up_prompt = f"""
        You're Alpha, the podcast host. React briefly (1 casual sentence) to Beta’s answer.

        Beta said: {ai_response}
        """
        alpha_follow_up = chat_alpha.invoke([HumanMessage(content=follow_up_prompt)]).content.strip()
        messages.append(HumanMessage(content=alpha_follow_up))

        alpha_placeholder = st.empty()
        alpha_placeholder.markdown(f"**Alpha:** {alpha_follow_up}")
        speak_text(alpha_follow_up, voice="alpha")

        st.markdown("---")
        time.sleep(0.2)
        q_num += 1

    # 🎉 Outro
    outro_prompt = """
    You're Alpha, the podcast host. End the podcast with a short outro (3-4 sentences), thanking Beta and the audience in a friendly tone.
    """
    outro = chat_alpha.invoke([HumanMessage(content=outro_prompt)]).content.strip()
    st.markdown(f"**Alpha:** {outro}")
    speak_text(outro, voice="alpha")

# Function to test AI rephrasing and answering
def test_ai_rephrasing():
    test_questions = [
        "What CSUSB class assists with creating a podcast?",
        "Who is the current president of CSUSB?",
        "What do I need to start a podcast?",
        "What is the deadline to apply to CSUSB for Fall 2025?",
        "What is the best mic to start a podcast?",
        "When do I need to submit my paper for my CSE 6550 class?",
        "What is the best way to format a podcast?",
        "When will a CSUSB podcast workshop be held?",
        "Where can I upload my podcast for listening?",
        "When is the next CSUSB Podcast class open?"
    ]

    for i, question in enumerate(test_questions):
        # Alpha rephrases the question
        rephrase_prompt = f"Rephrase the following question in your own words: {question}"
        messages.append(HumanMessage(content=rephrase_prompt))
        rephrase_response = chat_alpha.invoke(messages)
        rephrased_question = rephrase_response.content.strip() if rephrase_response else "Could not rephrase the question."

        # Beta answers the rephrased question
        if i % 2 == 0:
            ai_response = "I don't know"
        else:
            beta_prompt = f"""
            You are an AI assistant answering questions. Provide an accurate short response without mentioning this prompt.

            **Question:** {rephrased_question}
            """
            response = chat_beta.invoke([SystemMessage(content=beta_prompt)])
            ai_response = response.content.strip() if response else "I don't know"

        # st.write(f"**Original Question:** {question}")
        # ✅ Alpha's text and speech appear together
        alpha_text = f"**Alpha:** {question}"
        alpha_placeholder = st.empty()
        alpha_placeholder.markdown("")  # Start empty

        speak_text(question, voice="alpha")  # 🎙️ Alpha Speaks

        # Typing effect - Update text as it speaks
        for i in range(len(question)):
            alpha_placeholder.markdown(f"**Alpha:** {question[:i+1]}")
            time.sleep(0.05)  # Slow typing effect

        time.sleep(1)  # Small delay after Alpha finishes speaking
        # ✅ Beta's text and speech appear together
        beta_placeholder = st.empty()
        beta_placeholder.markdown("")  # Start empty

        speak_text(ai_response, voice="beta")  # 🎙️ Beta Speaks

        # Typing effect - Update text as it speaks
        for i in range(len(ai_response)):
            beta_placeholder.markdown(f"**Beta:** {ai_response[:i+1]}")
            time.sleep(0.05)  # Slow typing effect

        st.markdown("---")

        time.sleep(2)  # Small delay before the next question

        # Update confusion matrix
        if "I don't know" in ai_response:
            st.session_state.conf_matrix[1, 1] += 1
        else:
            st.session_state.conf_matrix[0, 0] += 1

    # Update Sidebar After Running Test AI
    update_sidebar()
      
# Create three buttons in separate columns above the text input box
col1, col2, col3 = st.columns(3)

with col1:
    if st.button(":material/voice_chat: Start AI Podcast", key="start_podcast_button_1"):
        st.session_state["podcast_started"] = True

with col2:
    if st.button(":material/mic: Start AI Podcast(Say: Podcast!)", key="start_podcast_voice_command_button"):
        podcast_command = listen_to_microphone()
        if "podcast" in podcast_command:
            st.session_state["podcast_started"] = True
        else:
            st.write(f"Command '{podcast_command}' not recognized for podcast start.")

with col3:
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
        
        st.write(f"**User:** {user_input}")
        st.write(f"**AI Response:** {ai_response}")

        # Update confusion matrix
        if "I do not know!" in ai_response:
            st.session_state.conf_matrix[1, 0] += 1
        else:
            st.session_state.conf_matrix[0, 0] += 1
    st.markdown('</div>', unsafe_allow_html=True)

# Unified Output Section (Ensuring Output Appears Below in a Single Row)
if st.session_state["podcast_started"]:
    st.write("---")  # Separator for clarity
    start_ai_podcast()  # Run the AI podcast function

if st.session_state.get("test_ai_clicked"):
    st.write("---")  # Separator for clarity
    test_ai_rephrasing()  # Run the AI rephrasing and answering function