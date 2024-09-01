from fastapi import FastAPI, Form
from langchain_core.runnables.base import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import requests
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

app = FastAPI()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

search = TavilySearchResults(
      max_results=2,
    )
tools = [search]
memory = MemorySaver()

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

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

  querystring = {"api-version":"3.0","profanityAction":"NoAction","textType":"plain", "to":target}

  payload = [{ "Text": text }]
  headers = {
    "x-rapidapi-key": os.getenv("RAPIDAPI_LANG_TRANS"),
    "x-rapidapi-host": "microsoft-translator-text.p.rapidapi.com",
    "Content-Type": "application/json"
  }

  response = requests.post(url, json=payload, headers=headers, params=querystring)
  res = response.json()
  return res[0]["translations"][0]["text"]


@app.post('/infer/{user_id}')
def infer(user_id: str, user_input: str = Form(...)):
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
            ("human", "{user_input}")
        ]
    )

    runnable = prompt | agent_executor # define a chain

    conversation = RunnableSequence( # wrap the chain along with chat history and user input
        runnable,
    )

    response = conversation.invoke( # invoke the chain by giving the user input and the chat history
        {"user_input": user_input},
        config={"configurable": {"thread_id":user_id}}
    )

    res = translate("ur", response["messages"][-1].content) # translate the response to Urdu

    return {
        "data": res
    }