import streamlit as st
from streamlit_chat import message
from datetime import datetime
from gpt_index import GPTSimpleVectorIndex,LLMPredictor
import openai
import os
from dotenv import load_dotenv
from langchain import OpenAI

load_dotenv()

st.markdown("<h1 style='text-align: center; color: Blue;'>Plant Disease Classification Chat-Bot👋</h1>", unsafe_allow_html=True)


api_key_input = st.text_input("Enter your OpenAI API key:")
openai.api_key = api_key_input
llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))
vector_index_path = "vectorIndex.json"

def get_bot_response(user_query):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vector_index_path)
    response = vIndex.query(user_query, response_mode="compact")
    return str(response)

def display_messages(all_messages):
    for idx, msg in enumerate(all_messages):
        if msg['user'] == 'user':
            message(f"You ({msg['time']}): {msg['text']}", is_user=True, key=f"user-{idx}")
        else:
            message(f"Bot ({msg['time']}): {msg['text']}", key=f"bot-{idx}")

def send_message(user_query, all_messages):
    if user_query:
        current_time = datetime.now().strftime("%H:%M:%S")
        all_messages.append({'user': 'user', 'time': current_time, 'text': user_query})
        bot_response = get_bot_response(user_query)
        all_messages.append({'user': 'bot', 'time': current_time, 'text': bot_response})

        st.session_state.all_messages = all_messages
        display_messages(all_messages)

if 'all_messages' not in st.session_state:
    st.session_state.all_messages = []

user_query = st.text_input("You: ", "", key="input")
send_button = st.button("Send")

if send_button:
    send_message(user_query, st.session_state.all_messages)
