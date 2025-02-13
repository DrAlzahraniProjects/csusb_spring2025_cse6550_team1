import streamlit as st
# import random,time
from huggingface_hub import InferenceClient

global APIK  # Declare the global variable
APIK = None

def get_api_key():
    global APIK
    st.title("API Key Input")
    APIK = st.text_input("Enter your API Key:", type="password", label_visibility="hidden")
    if APIK:
        st.success("API Key set successfully!")

def response_generator(prompt):
    client = InferenceClient(provider="sambanova", api_key="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")
    
    messages = [{"role": "user", "content": prompt}]
    
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",  # Replace with your desired model
        messages=messages,
        max_tokens=500,
        stream=True
    )
    
    response = "".join(chunk.choices[0].delta.content for chunk in stream)
    return response


# Streamlit app setup
st.title("Chatbot Assistant") 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    response = response_generator(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})