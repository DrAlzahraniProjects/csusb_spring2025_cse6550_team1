import streamlit as st
import speech_recognition as sr
import os
import pandas as pd
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import PyPDF2
import time
from docx import Document


# ✅ Update Streamlit App Header (Title, Icon, and Layout)
st.set_page_config(
    page_title="CSUSB Study Podcast",
    page_icon="logo/csusb_logo.png", 
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
apik = os.environ["GROQ_API_KEY"] = "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"
    
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
    

# Apply CSS to hide the misleading 200MB limit message
st.markdown(
    """
    <style>
        /* Hide the 200MB limit text */
        div[data-testid="stFileUploader"] div[aria-live="polite"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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

        # ✅ Display Updated Model Metrics in Sidebar
        st.markdown("### Model Metrics")
        st.write(f"**Model Accuracy:** {model_accuracy:.2f}%")
        st.write(f"**Precision:** {precision:.2f}%")
        st.write(f"**Recall:** {recall:.2f}%")
        st.write(f"**F1 Score:** {f1_score:.2f}%")

        # ✅ Reset Button to Clear Confusion Matrix and Model Metrics
        if st.button("🔄 Reset Metrics", key="reset_button"):
            st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])
            st.session_state["podcast_started"] = False
            st.success("Confusion Matrix & Model Metrics Reset!")
            st.experimental_rerun()  # Force UI refresh

# AI Podcast Function (Modified to Use Uploaded Document) 
def start_ai_podcast():
    for i in range(10):
        if extracted_text:
            question_prompt = f"""
            Generate a short, clear, and direct question (1-3 lines) based on the following document content:
            {extracted_text[:1000]}  # Use first 1000 characters for context
            """
        else:
            question_prompt = "Generate a short, clear, and direct question (1-3 lines) about CSUSB's podcast, events, admissions, or other CSUSB-related information."

        messages.append(HumanMessage(content=question_prompt))
        question_response = chat_alpha.invoke(messages)
        question = question_response.content.strip() if question_response else "Could not generate a question."

        # ✅ Ensure the question is properly formatted without unnecessary prefixes
        if question.startswith("→") or question.lower().startswith("alpha:") or "here is a short question" in question.lower():
            question = question.replace("→", "").replace("Alpha:", "").replace("here is a short question based on the document content:", "").strip()

        st.write(f"**Alpha:** {question}") 
        time.sleep(1)
        
        thinking_text = st.empty()
        thinking_text.write("**Beta is thinking...**")
        time.sleep(4)

        beta_prompt = f"""
        You are an AI assistant answering questions. Provide an accurate response in the following format:

        → (1-2 line summary of the answer)
        **Key Points:**
        - First key point
        - Second key point
        - Third key point

        **Question:** {question}
        """

        response = chat_beta.invoke([SystemMessage(content=beta_prompt)])
        ai_response = response.content.strip() if response else "I don't know"

        if i >= 5 or "I don't know" in ai_response or not ai_response:
            ai_response_summary = "I don't know"
        else:
            st.session_state.conf_matrix[0, 0] += 1

        messages.append(AIMessage(content=ai_response))
        thinking_text.empty()
        st.markdown(f"**Beta:**\n\n{ai_response}")
        st.markdown("---")

        time.sleep(1)

    # ✅ Update Sidebar After Running Podcast
    update_sidebar()
        
# Chatbot Input for Document-based or Model-based Communication
user_input = st.text_input("Ask the Chatbot a Question (Document-based or Model-based)", key="chat_input")

if st.button("Submit", key="submit_button_1"):
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


# Create two buttons in separate columns but trigger the same podcast function
col1, col2 = st.columns(2)

with col1:
    if st.button("Start AI Podcast (Click)", key="start_podcast_button_1"):
        st.session_state["podcast_started"] = True

with col2:
    if st.button("Start AI Podcast (Say: Podcast!)", key="start_podcast_voice_command_button"):
        podcast_command = listen_to_microphone()
        if "podcast" in podcast_command:
            st.session_state["podcast_started"] = True
        else:
            st.write(f"Command '{podcast_command}' not recognized for podcast start.")

# Unified Output Section (Ensuring Output Appears Below in a Single Row)
if st.session_state["podcast_started"]:
    st.write("---")  # Separator for clarity
    start_ai_podcast()  # Run the AI podcast function
            