# Urdu Virtual Assistant

This project is a comprehensive speech-to-speech model that understands and responds in Urdu. It integrates natural language processing with real-time information retrieval to provide a seamless conversational experience. The assistant can handle everyday conversations in Urdu and access relevant real-time information through the Tavily search engine.

## Features

- **Speech-to-Speech Interaction**: Converts spoken Urdu queries into text, processes the input, and generates spoken Urdu responses.
- **Real-Time Information Retrieval**: Integrates with Tavily search engine to provide up-to-date information.
- **Language Translation**: Translates Urdu queries to English for processing and then translates responses back to Urdu.
- **Text-to-Speech**: Converts generated responses into speech using Google Text-to-Speech (gTTS).
- **Customizable**: Built with modular components, making it easy to extend or adapt to different use cases.

## Technologies Used

- **Streamlit**: Frontend for the application.
- **Google Text-to-Speech (gTTS)**: For converting text to speech.
- **Hugging Face Transformers**: Specifically, `wav2vec2` for Urdu speech recognition.
- **LangChain**: For building conversational AI with Groq's Llama model.
- **Tavily Search Engine**: For retrieving real-time information.

## Project Structure

```bash
├── app.py              # Main application code
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
├── bolo_logo.png       # Logo image
├── audio.wav           # Temporary audio file storage
└── README.md           # Project documentation
```

## Setup and Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/urdu-virtual-assistant.git
cd urdu-virtual-assistant
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Then, run:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root and add the following variables:
```bash
streamlit run app.py
```
The application should now be accessible at `http://localhost:8501`.

## Usage
* Launching the Application: Once the app is running, you will see a title and logo at the top of the page.
* Recording a Query: Click the "Click to record" button to start recording your Urdu query. Click again to stop.
* Processing the Query: The app will convert the spoken query to text, process it, and then generate a spoken response.
* Listening to the Response: The AI's response will be played back in Urdu.

## Acknowledgments
* Special thanks to the creators of Streamlit, LangChain, Hugging Face, and other open-source libraries used in this project.
* Thanks to Groq and Tavily for providing the necessary APIs and models.

## Contributors
* Zohaib Saqib
* Ghulam Abbas
* Adeena Khudadad
* Abdur Rahman
* Ryyan
