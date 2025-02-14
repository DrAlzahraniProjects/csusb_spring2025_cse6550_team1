import streamlit as st
import os
from langchain.chat_models import init_chat_models
from langchain.schema import SystemMessage, HumanMessage, AIMessage



apik = os.getenv("GROQ_API_KEY")
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()

chat = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
messages = [SystemMessage(content="You are a AI assistant that will help.")]

# Streamlit app setup
st.title(":blue[CSUSB] Chatbot Assistant")
st.header("Spring 2025 Team 1", divider="blue")


user_input = st.text_input("Enter your Message:")

if st.button("Submit"):
    if user_input:
        messages.append(HumanMessage(content=user_input))
        response = chat.invoke(messages)
        ai_message = AIMessage(content=response.content)
        messages.append(ai_message)
        st.write(f"CSUSB Chatbot response: {response.content}")
    else:
        st.warning("Please enter a message before you submit.")