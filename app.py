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

# Sidebar - Performance Metrics
st.sidebar.write("### Performance Metrics")

# Initialize Confusion Matrix in Session State
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])  # Start with zeros

# Chatbot Message Section
user_input = st.text_input("Ask the Chatbot a Question", key="chat_input")

if st.button("Submit"):
    if user_input:
        messages.append(HumanMessage(content=user_input))
        response = chat.invoke(messages)
        ai_message = AIMessage(content=response.content)
        messages.append(ai_message)

        # Display user input
        st.markdown(
            f"<div style='background-color:#4a4a4a; color:white; padding:10px; border-radius:5px; width:fit-content; margin:10px 0;'>"
            f"{user_input}</div>",
            unsafe_allow_html=True,
        )

        # Display chatbot response
        st.markdown(
            f"<div style='background-color:#333; color:white; padding:10px; border-radius:5px; width:fit-content;'>"
            f"CSUSB Chatbot response: {response.content}</div>",
            unsafe_allow_html=True,
        )

        # Store latest chatbot response in session state
        st.session_state.latest_response = response.content

    else:
        st.warning("Please enter a question before submitting.")

# User Feedback - Confusion Matrix Update
st.write("### Rate Chatbot's Response")
col1, col2 = st.columns(2)
if col1.button("Correct ✅"):
    st.session_state.conf_matrix[0, 0] += 1  # True Positive (TP)

if col2.button("Incorrect ❌"):
    st.session_state.conf_matrix[0, 1] += 1  # False Negative (FN)

# Display Confusion Matrix
st.sidebar.write("### Confusion Matrix")
conf_df = pd.DataFrame(
    st.session_state.conf_matrix,
    index=["Actual +", "Actual -"],
    columns=["Predicted +", "Predicted -"],
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

# Reset Button
if st.button("Reset Confusion Matrix"):
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])  # Reset to zeros
    st.rerun()
