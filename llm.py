from fastapi import FastAPI, Form
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
import requests
load_dotenv()
chat_histories = {} # for keeping track of users and their chat histories


app = FastAPI()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def translate(target, text):
  '''
    Translates given text into target language

    Parameters:
        target (string): 2 character code to specify the target language.
        text (string): Text to be translated.
    
    Returns:
        res (string): Translated text.
  '''
  url = "https://microsoft-translator-text.p.rapidapi.com/translate"

  querystring = {"api-version":"3.0","profanityAction":"NoAction","textType":"plain", "to":target} # parameters for the API (defines what language to target, take action on profanities if they are detected, etc)

  payload = [{ "Text": text }] # actual text to be converted to target language
  headers = {
    "x-rapidapi-key": os.getenv("RAPIDAPI_LANG_TRANS"),
    "x-rapidapi-host": os.getenv("RAPIDAPI_HOST"),
    "Content-Type": "application/json"
  }

  response = requests.post(url, json=payload, headers=headers, params=querystring) # call the API
  res = response.json() # convert response to JSON format
  return res[0]["translations"][0]["text"] # extract response 


def get_session_history(user_id: str):
    '''
        Saves history of users with respect to a particular User ID

        Parameters:
            user_id (string): User ID of a user.
        
        Returns:
            chat_history (ChatMessageHistory): A ChatMessageHistory object containing all the conversation history.
    '''
    if user_id not in chat_histories: # if it is an existing user
        memory = ChatMessageHistory(memory_key="chat_history") # extract chat history from memory
        chat_histories[user_id] = memory # save chat history in chat_histories dictionary
    return chat_histories[user_id] # return chat history


@app.get('/history/{user_id}')
def history(user_id: str):
    '''
        Returns chat history of a user.

        Parameters:
            user_id (string): User ID of a user.
        
        Returns:
            JSON Response (Dictionary): Returns a dictionary object with the chat history.
    '''
    return {
        'history': get_session_history(user_id)
    }


@app.post('/infer/{user_id}')
def infer_diagnosis(user_id: str, user_input: str = Form(...)):
    '''
        Returns the translated response from the LLM in response to a user query.

        Parameters:
            user_id (string): User ID of a user.
            user_input (string): User query.
        
        Returns:
            JSON Response (Dictionary): Returns a translated response from the LLM.
    '''

    user_input = translate("en", user_input) # translate user query to english

    prompt = ChatPromptTemplate.from_messages( # define a prompt
        [
            (
                "system",
                "You're a compassionate AI virtual Assistant"
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            ("human", "{user_input}")
        ]
    )

    runnable = prompt | llm | StrOutputParser() # define a chain

    conversation = RunnableWithMessageHistory( # wrap the chain along with chat history and user input
        runnable,
        get_session_history,
        history_messages_key="chat_history",
        input_messages_key="user_input"
    )

    response = conversation.invoke( # invoke the chain by giving the user input and the chat history
        {"user_input": user_input},
        config={"configurable": {"session_id": user_id}}
    )

    res = translate("ur", response) # translate the response to Urdu

    return {
        "data": res
    }