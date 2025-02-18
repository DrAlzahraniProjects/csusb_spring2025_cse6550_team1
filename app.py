import streamlit as st
import os
import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Initialize API key
apik = os.getenv("GROQ_API_KEY")
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()

# Initialize chat model
chat = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
messages = [SystemMessage(content="You are an AI assistant that will help.")]

# Streamlit UI setup
st.markdown("<h1 style='text-align: center;'>CSUSB Podcast Chatbot</h1>", unsafe_allow_html=True)

# Left section: Confusion Matrix & Metrics
st.sidebar.write("### Performance Metrics")

# Initialize Confusion Matrix
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.array([[1, 1], [1, 1]])  # Default values from image

# Confusion Matrix Display
st.sidebar.write("### Confusion Matrix")
conf_df = pd.DataFrame(
    st.session_state.conf_matrix,
    index=["Actual +", "Actual -"],
    columns=["Predicted +", "Predicted -"],
)
st.sidebar.table(conf_df)

# Calculate Performance Metrics
TP = st.session_state.conf_matrix[0, 0]
FP = st.session_state.conf_matrix[1, 0]
FN = st.session_state.conf_matrix[0, 1]
TN = st.session_state.conf_matrix[1, 1]

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
accuracy = (TP + TN) / np.sum(st.session_state.conf_matrix)
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
    st.sidebar.write(f"**{metric}:** {value:.1f}")

# Chatbot Message Section
user_input = st.text_input("Message Chatbot", key="chat_input")

if st.button("Submit"):
    if user_input:
        messages.append(HumanMessage(content=user_input))
        response = chat.invoke(messages)
        ai_message = AIMessage(content=response.content)
        messages.append(ai_message)

        st.markdown(
            f"<div style='background-color:#4a4a4a; color:white; padding:10px; border-radius:5px; width:fit-content; margin:10px 0;'>"
            f"{user_input}</div>",
            unsafe_allow_html=True,
        )

        st.write("Sure! Please upload the podcast audio file or share a link to it, and let me know what kind of analysis you need—summary, key themes, sentiment analysis, or something else.")

        st.markdown(
            f"<div style='background-color:#333; color:white; padding:10px; border-radius:5px; width:fit-content;'>"
            f"CSUSB Chatbot response: {response.content}</div>",
            unsafe_allow_html=True,
        )

    else:
        st.warning("Please enter a message before submitting.")

# Upload Section
uploaded_file = st.file_uploader("", type=["mp3", "wav", "m4a"])

# Reset Button
if st.button("Reset"):
    st.session_state.conf_matrix = np.array([[1, 1], [1, 1]])  # Reset to default matrix
    st.experimental_rerun()
