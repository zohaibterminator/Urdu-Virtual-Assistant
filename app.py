import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import requests
load_dotenv()

user_id = "1" # example user id


def get_response(user_input):
    '''
        Takes user_input in english, and invokes the infer API for response.

        Parameters:
            user_input (string): User Query in english.
        Returns:
            res (string): Response from the LLM.
    '''
    url = f"http://127.0.0.1:8000/infer/{user_id}"

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"user_input": user_input}

    response = requests.post(url, headers=headers, data=data, stream=True)
    res = response.json()
    return res["data"]


st.set_page_config(page_title="Urdu Virtual Assistan", page_icon="🤖") # set the page title and icon
st.title("Urdu Virtual Assistant") # set the main title of the application

user_query = st.chat_input("Your message") # take input
if user_query: # if input is not an empty string
    with st.chat_message("Human"): # create the message box for human input
        st.markdown(user_query) # display the message as markdown

    response_text = get_response(user_input=user_query) # get response from the LLM.

    with st.chat_message("AI"): # create the message box for AI input
        st.markdown(response_text) 