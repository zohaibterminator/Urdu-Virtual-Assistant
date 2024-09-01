import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from langchain_core.messages import HumanMessage, AIMessage
import requests
from transformers import pipeline
from gtts import gTTS
import io
from langchain_core.runnables.base import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import requests
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

st.set_page_config(page_title="Urdu Virtual Assistant", page_icon="🤖")  # set the page title and icon

# Load environment variables (if any)
load_dotenv()

user_id = "1"  # example user id

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

agent_executor = create_react_agent(llm, tools)

# Initialize the wav2vec2 model for Urdu speech-to-text
pipe = pipeline("automatic-speech-recognition", model="kingabzpro/wav2vec2-large-xls-r-300m-Urdu")

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


def infer(user_input: str):
    '''
        Returns the translated response from the LLM in response to a user query.

        Parameters:
            user_id (string): User ID of a user.
            user_input (string): User query.

        Returns:
            res (string): Returns a translated response from the LLM.
    '''

    user_input = translate("en", user_input) # translate user query to english

    prompt = ChatPromptTemplate.from_messages( # define a prompt
        [
            (
                "system",
                "You are a compassionate and friendly AI virtual assistant. You will provide helpful answers to user queries using the provided tool to ensure the accuracy and relevance of your responses."
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
    )

    res = translate("ur", response["messages"][-1].content) # translate the response to Urdu
    return res


def text_to_speech(text, lang='ur'):
    '''
        Converts text to speech using gTTS.

        Parameters:
            text (string): Text to be converted to speech.
            lang (string): Language for the speech synthesis. Default is 'ur' (Urdu).
        Returns:
            response_audio_io (BytesIO): BytesIO object containing the audio data.
    '''
    tts = gTTS(text, lang=lang)
    response_audio_io = io.BytesIO()
    tts.write_to_fp(response_audio_io)
    response_audio_io.seek(0)
    return response_audio_io


col1, col2 = st.columns([1, 5])  # Adjust the ratio to control the logo and title sizes

# Display the logo in the first column
with col1:
    st.image("bolo_logo-removebg-preview.png", width=100)  # Adjust the width as needed

# Display the title in the second column
with col2:
    st.title("Urdu Virtual Assistant") # set the main title of the application
st.write("This application is a comprehensive speech-to-speech model designed to understand and respond in Urdu. It not only handles natural conversations but also has the capability to access and provide real-time information by integrating with the Tavily search engine. Whether you're asking for the weather or engaging in everyday dialogue, this assistant delivers accurate and context-aware responses, all in Urdu.")

# Add a text input box
audio = audiorecorder()

if len(audio) > 0:
    # Save the audio to a file
    audio.export("audio.wav", format="wav")

    # Convert audio to text using the wav2vec2 model
    with open("audio.wav", "rb") as f:
        audio_bytes = f.read()

    # Process the audio file
    result = pipe("audio.wav")
    user_query = result["text"]

    with st.chat_message("Human"):  # create the message box for human input
        st.audio(audio.export().read())  # display the audio player
        st.markdown(user_query)

    # Get response from the LLM
    response_text = infer(user_input=user_query)
    response_audio = text_to_speech(response_text, lang='ur')

    # Play the generated speech in the app
    with st.chat_message("AI"):
      st.audio(response_audio.read(), format='audio/mp3')
      st.markdown(response_text)