import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# MUST BE FIRST Streamlit command
st.set_page_config(page_title="TinyLlama Chatbot", page_icon="🤖")

# Now your Streamlit UI code
st.title("💬 TinyLlama Chatbot")
# Load model and tokenizer once
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# User input box
user_input = st.text_input("You:", key="input")

if user_input:
    # Append user input to history
    st.session_state.chat_history += f"User: {user_input}\nAssistant:"

    # Tokenize + generate
    inputs = tokenizer(st.session_state.chat_history, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant reply
    assistant_reply = response.split("Assistant:")[-1].strip()
    st.session_state.chat_history += f" {assistant_reply}\n"

    # Display full chat
    for line in st.session_state.chat_history.strip().split("\n"):
        if line.startswith("User:"):
            st.markdown(f"**🧑‍💻 {line[5:].strip()}**")
        elif line.startswith("Assistant:"):
            st.markdown(f"**🤖 {line[10:].strip()}**")

