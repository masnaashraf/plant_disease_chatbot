import streamlit as st
from streamlit_chat import message
import json
import openai
from datetime import datetime
from  gpt_index import  GPTListIndex, GPTSimpleVectorIndex
import sys
import os
from dotenv import load_dotenv
load_dotenv()
#from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]

st.markdown("<h1 style='text-align: center; color: Blue;'>Plant Disease Classification Chat-Bot👋</h1>", unsafe_allow_html=True)

vector_index_path = "vectorIndex.json"

def get_bot_response(user_query):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vector_index_path)
    response = vIndex.query(user_query, response_mode="compact")
    return str(response)

def answerMe(vectorIndex):
  vIndex = GPTSimpleVectorIndex.load_from_disk(vectorIndex)
  while True:
    prompt = input("Please ask: ")
    if prompt == "exit":
      break
    response = vIndex.query(prompt, response_mode="compact")
    print(f"Response: {response} \n")

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
