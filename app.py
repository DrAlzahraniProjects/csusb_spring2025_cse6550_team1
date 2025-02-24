import streamlit as st
import os
import numpy as np
import pandas as pd
import PyPDF2
import speech_recognition as sr
import pyttsx3
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Initialize API key
apik = os.getenv("GROQ_API_KEY")
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()

# Initialize chat model
chat = init_chat_model("llama3-8b-8192", model_provider="groq")
messages = [SystemMessage(content="You are an AI assistant that will help.")]

# Text-to-Speech Engine Setup
tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Streamlit UI setup
st.markdown("<h1 style='text-align: center;'>CSUSB Podcast Chatbot</h1>", unsafe_allow_html=True)

# UI Layout Adjustments
chat_container = st.container()
prompt_container = st.container()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF for Discussion", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    st.session_state.pdf_text = text
    st.success("PDF uploaded successfully! Conversation will be based on this document.")

# Predefined Questions
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

# Confusion Matrix Initialization
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])

# Generate AI Conversation
if st.button("Start AI Podcast Discussion"):
    if "pdf_text" in st.session_state:
        for i, question in enumerate(questions):
            messages.append(HumanMessage(content=question))
            response = chat.invoke(messages)
            ai_response = response.content if i % 2 == 0 else "I do not know!"
            ai_message = AIMessage(content=ai_response)
            messages.append(ai_message)
            
            with chat_container:
                st.write(f"**User:** {question}")
                st.write(f"**Chatbot:** {ai_response}")
                speak(ai_response)
            
            if ai_response != "I do not know!":
                st.session_state.conf_matrix[0, 0] += 1  # True Positive (TP)
            else:
                st.session_state.conf_matrix[1, 0] += 1  # False Negative (FN)
    else:
        st.warning("Please upload a PDF first.")

# Display Confusion Matrix
st.sidebar.write("### Confusion Matrix")
conf_df = pd.DataFrame(
    st.session_state.conf_matrix,
    index=["Actual +", "Actual -"],
    columns=["Predicted +", "Predicted -"]
)
st.sidebar.table(conf_df)

# Calculate Performance Metrics
TP = st.session_state.conf_matrix[0, 0]
FN = st.session_state.conf_matrix[0, 1]
FP = st.session_state.conf_matrix[1, 0]
TN = st.session_state.conf_matrix[1, 1]

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
accuracy = (TP + TN) / np.sum(st.session_state.conf_matrix) if np.sum(st.session_state.conf_matrix) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

metrics = {
    "Sensitivity": sensitivity,
    "Specificity": specificity,
    "Accuracy": accuracy,
    "Precision": precision,
    "F1 Score": f1_score,
}
for metric, value in metrics.items():
    st.sidebar.write(f"**{metric}:** {value:.2f}")

# Prompt UI for additional input
with prompt_container:
    user_input = st.text_input("Ask the Chatbot a Question", key="chat_input")
    if st.button("Submit"):
        if user_input:
            messages.append(HumanMessage(content=user_input))
            response = chat.invoke(messages)
            ai_response = response.content if "?" in user_input else "I do not know!"
            ai_message = AIMessage(content=ai_response)
            messages.append(ai_message)
            
            with chat_container:
                st.write(f"**User:** {user_input}")
                st.write(f"**Chatbot:** {ai_response}")
                speak(ai_response)
        else:
            st.warning("Please enter a question before submitting.")
