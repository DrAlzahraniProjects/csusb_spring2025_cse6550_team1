import threading
import speech_recognition as sr
import streamlit as st
import os
import pandas as pd
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import PyPDF2
import time
from docx import Document

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
def listen_to_microphone(command_flag):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command_flag.append(recognizer.recognize_google(audio).lower())
        except sr.UnknownValueError:
            command_flag.append("Sorry, I couldn't understand that.")
        except sr.RequestError:
            command_flag.append("Sorry, I'm unable to process the request at the moment.")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"

# Define prompts for both document-based and model-based communication
def get_document_based_prompt(extracted_text, user_query):
    return f"""
    You are an AI assistant. The following is the content extracted from a document uploaded by the user. Use this extracted text to respond to the user's queries. Please do not generate answers outside of the provided document text.

    Extracted Document Text:
    {extracted_text}

    User Query: {user_query}

    Please provide the most relevant response from the document above.
    """

def get_model_based_prompt(user_query):
    return f"""
    You are an AI assistant with general knowledge. Please respond to the user's query based on your pre-trained knowledge. Do not refer to any uploaded documents or extracted text unless explicitly instructed.

    User Query: {user_query}

    Please provide the most relevant and accurate response based on your training.
    """

# Initialize confusion matrix
if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])

# AI Podcast Conversation
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
        st.write(f"**Alpha:** {question}")
        time.sleep(1)

        # Show "Beta is thinking..." message
        thinking_text = st.empty()
        thinking_text.write("**Beta is thinking...**")
        time.sleep(2)

        # Select the model based on index
        messages.append(HumanMessage(content=question))
        response = chat_alpha.invoke(messages) if i % 2 == 0 else chat_beta.invoke(messages)
        ai_response = response.content if response else "I do not know!"
        ai_response_summary = " ".join(ai_response.splitlines()[:5])

        messages.append(AIMessage(content=ai_response_summary))
        thinking_text.empty()
        st.write(f"**Beta:** {ai_response_summary}")
        st.markdown("---")

        # Update confusion matrix based on correctness
        if ai_response != "I do not know!":
            st.session_state.conf_matrix[0, 0] += 1  # True Positive
        else:
            st.session_state.conf_matrix[1, 0] += 1  # False Negative

        time.sleep(5)

# Header Section
col1, col2 = st.columns([1, 3])
with col1:
    st.image(r"logo/csusb_logo.png", width=100)
with col2:
    st.markdown("<h1 style='text-align: center;'>CSUSB Study Podcast Assistant</h1>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX)", type=["pdf", "docx"])
extracted_text = ""

if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_docx(uploaded_file)

if st.button("Process Extracted Text"):
    if extracted_text:
        messages.append(HumanMessage(content=extracted_text))
        response = chat_alpha.invoke(messages)  # Using Alpha model
        ai_response = response.content if response else "I do not know!"
        ai_response_summary = " ".join(ai_response.splitlines()[:5])
        messages.append(AIMessage(content=ai_response_summary))
        st.write(f"**Extracted Text:** {extracted_text}")
        st.write(f"**AI Response:** {ai_response_summary}")
    else:
        st.warning("No text extracted to process.")

# Chatbot Input for Document-based or Model-based Communication
user_input = st.text_input("Ask the Chatbot a Question (Document-based or Model-based)", key="chat_input")
if st.button("Submit", key="submit_button_1"):
    prompt = get_document_based_prompt(extracted_text, user_input) if uploaded_file else get_model_based_prompt(user_input)
    response = chat_alpha.invoke([SystemMessage(content=prompt)]) if uploaded_file else chat_beta.invoke([SystemMessage(content=prompt)])
    ai_response = response.content if response else "I do not know!"
    st.write(f"**User:** {user_input}")
    st.write(f"**AI Response:** {ai_response}")

    # Update confusion matrix
    if "I do not know!" in ai_response:
        st.session_state.conf_matrix[1, 0] += 1
    else:
        st.session_state.conf_matrix[0, 0] += 1

# Podcast Start Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start AI Podcast (Click)", key="start_podcast_button_1"):
        start_ai_podcast()

with col2:
    podcast_command_flag = []
    if st.button("Start AI Podcast (Say: Start Podcast!)", key="start_podcast_voice_command_button"):
        listening_thread = threading.Thread(target=listen_to_microphone, args=(podcast_command_flag,))
        listening_thread.start()
        listening_thread.join()
        podcast_command = podcast_command_flag[0] if podcast_command_flag else ""
        if "start podcast" in podcast_command:
            st.write("Podcast starting...")
            start_ai_podcast()
        else:
            st.write(f"Command '{podcast_command}' not recognized for podcast start.")
