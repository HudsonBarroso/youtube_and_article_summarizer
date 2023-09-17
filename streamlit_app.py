import streamlit as st
from st_chat_message import message
import pandas as pd
import utils
from datetime import datetime
from langchain.callbacks import get_openai_callback
from summarizer_agent import summary_agent

st.title("LangChain Chatapp :bird:")
tab1, tab2 = st.tabs(['Chat', "Usage Chart"])

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = summary_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

def append_state_messages(user_message, bot_message):
    """
    Appends a user message and a bot message to the app's state.
    Args:
        user_message (str): The user message.
        bot_message (str): The bot message.
    """
    st.session_state.messages.append({"user_message":user_message, "bot_message":bot_message})

def restore_history_message():
    """ Restores the app's history messages to the UI. """
    for history_message in st.session_state.messages:
        message(history_message['user_message'], is_user=True, key=str(datetime.now()))
        message(history_message['bot_message'], is_user=False, key=str(datetime.now()))

user_message = st.chat_input(placeholder="Type a message...")

# Chat tab
with tab1:
    st.header("OpenAI Chatapp")
    if user_message:
        restore_history_message()
        output = st.session_state.llm_chain.summarize(query=user_message)
        message(user_message, is_user=True, key="user_message")
        message(output, is_user=False, key="bot_message")
        append_state_messages(user_message, output)
        
# Usage Chart tab
with tab2:
    st.header("OpenAI Usage Cost")
    df = pd.read_csv("openai_api_usage.csv")
    st.bar_chart(df, x="us_date_format", y="total_cost_usd")                             
