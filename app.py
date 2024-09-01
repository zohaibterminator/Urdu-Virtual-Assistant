import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from langchain_core.messages import HumanMessage, AIMessage
import requests
from transformers import pipeline
from gtts import gTTS
import io

# Load environment variables (if any)
load_dotenv()

user_id = "1"  # example user id

# Initialize the wav2vec2 model for Urdu speech-to-text
pipe = pipeline("automatic-speech-recognition", model="kingabzpro/wav2vec2-large-xls-r-300m-Urdu")

def get_response(user_input):
    '''
        Takes user_input in English and invokes the infer API for response.

        Parameters:
            user_input (string): User Query in English.
        Returns:
            res (string): Response from the LLM.
    '''
    url = f"http://127.0.0.1/infer/{user_id}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"user_input": user_input}
    response = requests.post(url, headers=headers, data=data)
    res = response.json()
    return res["data"]


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


st.set_page_config(page_title="Urdu Virtual Assistant", page_icon="🤖")  # set the page title and icon
st.title("Urdu Virtual Assistant")  # set the main title of the application

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
    response_text = get_response(user_input=user_query)
    response_audio = text_to_speech(response_text, lang='ur')

    # Play the generated speech in the app
    with st.chat_message("AI"):
      st.audio(response_audio.read(), format='audio/mp3')
      st.markdown(response_text)