import threading
import speech_recognition as sr
import streamlit as st
import os
import pyttsx3
import pandas as pd
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Initialize API key
apik = os.environ["GROQ_API_Key"] = "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"
    
# Initialize chat model
chat = init_chat_model("llama3-8b-8192", model_provider="groq")
messages = [SystemMessage(content="You are an AI assistant that will help.")] 

# Text-to-Speech Engine Setup
tts_engine_alpha = pyttsx3.init()
tts_engine_beta = pyttsx3.init()

# Set the voices (Male for Alpha, Female for Beta)
voices = tts_engine_alpha.getProperty('voices')
tts_engine_alpha.setProperty('voice', voices[0].id)  # Male voice for Alpha
tts_engine_beta.setProperty('voice', voices[1].id)   # Female voice for Beta

# Function to speak using Alpha voice
def speak_alpha(text):
    def run_speech():
        tts_engine_alpha.say(text)
        tts_engine_alpha.runAndWait()

    # Run the speech in a separate thread to avoid blocking
    speech_thread = threading.Thread(target=run_speech)
    speech_thread.start()

# Function to speak using Beta voice
def speak_beta(text):
    def run_speech():
        tts_engine_beta.say(text)
        tts_engine_beta.runAndWait()

    # Run the speech in a separate thread to avoid blocking
    speech_thread = threading.Thread(target=run_speech)
    speech_thread.start()

# Speech Recognition Setup
recognizer = sr.Recognizer()

# Function to listen to microphone
def listen_to_microphone(command_flag):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
        try:
            speech_text = recognizer.recognize_google(audio)
            command_flag.append(speech_text.lower())  # Append the command text to the flag
        except sr.UnknownValueError:
            command_flag.append("Sorry, I couldn't understand that.")
        except sr.RequestError:
            command_flag.append("Sorry, I'm unable to process the request at the moment.")

# Initialize session state for confusion matrix
if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = np.zeros((2, 2), dtype=int)  # [TP, FP], [FN, TN]

# Start AI Podcast (Separate from document interaction)
def start_ai_podcast():
    questions = [
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
    
    for i, question in enumerate(questions):
        # Alpha speaks the question
        speak_alpha(question)
        
        # AI Response generation (Dummy response simulation here)
        messages.append(HumanMessage(content=question))
        response = chat.invoke(messages)
        ai_response = response.content if i % 2 == 0 else "I do not know!"

        # Generate a short summary (simulated here by slicing the first 5-10 lines)
        ai_response_summary = " ".join(ai_response.splitlines()[:5])  # First 5 lines

        ai_message = AIMessage(content=ai_response_summary)
        messages.append(ai_message)
        
        # Beta speaks the answer
        speak_beta(ai_response_summary)
        
        # Display the conversation in a new row for better UI
        with st.container():
            st.write(f"**Alpha (Question):** {question}")
            st.write(f"**Beta (Answer):** {ai_response_summary}")
            st.write("---")  # Separator for better readability

        # Update confusion matrix based on the correctness of the answer
        # For simplicity, assume even-indexed questions are correctly answered
        if i % 2 == 0:
            st.session_state.conf_matrix[0, 0] += 1  # True Positive
        else:
            st.session_state.conf_matrix[1, 0] += 1  # False Positive

        # Display Confusion Matrix
        st.sidebar.write("### Confusion Matrix")
        conf_df = pd.DataFrame(
            st.session_state.conf_matrix,
            index=["Actual +", "Actual -"],
            columns=["Predicted +", "Predicted -"]
        )
        st.sidebar.table(conf_df)

# Streamlit UI setup (Logo + Title in a single column)
col1, col2 = st.columns([1, 3])  # Adjust column width to your preference

# Centered Logo in the first column
with col1:
    st.image("csusb_logo.png", width=100)  # Adjust path if necessary

# Title in the second column (exactly centered)
with col2:
    st.markdown("<h1 style='text-align: center;'>CSUSB Study Podcast Assistant</h1>", unsafe_allow_html=True)

# UI Layout Adjustments (One column with two rows)
col1 = st.columns(1)[0]  # Single column layout

# Prompt Section (Top part of screen)
with col1:
    user_input = st.text_input("Ask the Chatbot a Question (Document-based)", key="chat_input")
    if st.button("Submit", key="submit_button_1"):
        if user_input:
            messages.append(HumanMessage(content=user_input))
            response = chat.invoke(messages)
            ai_response = response.content if "?" in user_input else "I do not know!"

            # Generate a short summary (simulated here by slicing the first 5-10 lines)
            ai_response_summary = " ".join(ai_response.splitlines()[:5])  # First 5 lines

            ai_message = AIMessage(content=ai_response_summary)
            messages.append(ai_message)
            
            st.write(f"**Alpha:** {user_input}")
            st.write(f"**Beta:** {ai_response_summary}")
            speak_beta(ai_response_summary)
        else:
            st.warning("Please enter a question before submitting.")

# AI Podcast Section (Bottom part of screen with buttons)
with col1:
    col2, col3 = st.columns(2)  # Two buttons side by side

    with col2:
        if st.button("Start AI Podcast (Click or Speak)", key="start_podcast_button_1"):
            start_ai_podcast()

    with col3:
        # Create a list to hold the podcast command
        podcast_command_flag = []
        
        # Start the listening thread
        if st.button("Start Podcast with Voice Command", key="start_podcast_voice_command_button"):
            # Listen to microphone in a separate thread
            listening_thread = threading.Thread(target=listen_to_microphone, args=(podcast_command_flag,))
            listening_thread.start()

            # Wait for the command to be recognized and process
            listening_thread.join()
            podcast_command = podcast_command_flag[0] if podcast_command_flag else ""
            
            if "start podcast" in podcast_command:
                st.write("Podcast starting...")
                start_ai_podcast()
            else:
                st.write(f"Command '{podcast_command}' not recognized for podcast start.")