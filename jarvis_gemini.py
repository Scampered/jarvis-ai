import google.generativeai as genai
import speech_recognition as sr
from openai import OpenAI
import pyaudio
import os
import re
import time
import warnings
warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")
from faster_whisper import WhisperModel

wake_word = 'jarvis'
listening_for_wake_word = True

whisper_size = "base"
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores, num_workers=num_cores)

#open ai api key
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
# configure connection to api
client = OpenAI(api_key=OPENAI_KEY)
# google api key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
# configure connection to api
genai.configure(api_key=GOOGLE_API_KEY)


# configure the responses of ai
generation_config = {
    "temperature": 0.7, # how random the responses are, higher means more creative.
    "top_p": 1,
    "top_k": 1,
    # normal sized response from ai
    "max_output_tokens": 2048,
}

# allow ai to chat freely
safety_settings = [
    {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
    },
    {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
    },
    {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
    },
    {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
    },
]

#select the model of ai, and pass the response settings to the model
model = (genai.GenerativeModel
         ('gemini-1.5-pro-002',
                              generation_config=generation_config,
                              safety_settings=safety_settings
          )
         )

# start a converstation with ai
convo = model.start_chat()

system_msg = """ You are Jarvis, a knowledgeable and helpful voice assistant. 
Respond to user prompts in a short, clear, concise, and informative manner. Do not use contractions. 
Do not use any subscripts or superscripts. Do not use special characters.
Avoid excessive detail and fofeeedcus on providing the most relevant information. 
Always be polite and respectful.
"""
system_msg = system_msg.replace(f"\n", "")
convo.send_message(system_msg)

r = sr.Recognizer()
source = sr.Microphone()

def speak(text):
    """Generate and play TTS audio from OpenAI with proper playback settings."""
    try:
        # Initialize PyAudio for audio playback
        p = pyaudio.PyAudio()
        stream_start = False

        # OpenAI TTS likely uses 24 kHz; adjust this if necessary
        playback_rate = 24000
        playback_format = pyaudio.paInt16  # Ensure PCM 16-bit signed integers
        playback_channels = 1  # Mono audio

        player_stream = p.open(
            format=playback_format,
            channels=playback_channels,
            rate=playback_rate,
            output=True,
        )

        # Generate TTS audio using OpenAI
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",             # Ensure the model name is correct
            voice="onyx",              # Replace "echo" with your chosen voice
            response_format="pcm",     # PCM output format
            input=text,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                if not stream_start:
                    stream_start = True
                player_stream.write(chunk)

        # Close the audio stream
        player_stream.stop_stream()
        player_stream.close()
        p.terminate()
    except Exception as e:
        print("Error during text-to-speech playback:", e)

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    # Convert generator to list of transcripts
    text_list = [segment.text for segment in segments]
    text = "".join(text_list)
    return text

def convert_to_plain_text(text):
    """Converts formatted text to plain text by replacing subscripts and special characters."""
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    special_char_map = {
        '/': 'slash',
        '+': 'plus',
        '=': 'equals',
        '@': 'at',
        '#': 'hashtag',
        '$': 'dollar',
        '%': 'percent',
        '&': 'and',
        '^': 'caret',
        '~': 'tilde',
        '|': 'pipe'
    }

    # Replace subscripts
    for subscript, replacement in subscript_map.items():
        text = text.replace(subscript, replacement)

    # Replace special characters
    for char, word in special_char_map.items():
        text = text.replace(char, f" {word} ")

    # Remove any remaining special characters
    text = re.sub(r'[^A-Za-z0-9 ,.!?]', '', text)
    return text

# Continuously prompt user for input
def continuous_prompt(source):
    global listening_for_wake_word

    while not listening_for_wake_word:
        print("Listening for your question...")
        try:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            prompt_gpt(audio)
        except sr.WaitTimeoutError:
            print("No input detected.")
            speak("I didn't hear anything. Please try again.")
        except Exception as e:
            print("Error in continuous_prompt:", e)
            speak("An error occurred. Please try again.")

# Listen for wake word
def listen_for_wake_word(audio):
    global listening_for_wake_word

    wake_audio_path = "wake_detect.wav"
    with open(wake_audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    text_input = wav_to_text(wake_audio_path)
    if wake_word in text_input.lower():
        print("Wake word detected. Welcome to ", wake_word)
        speak("Jarvis is activated. How can I help?")
        listening_for_wake_word = False
        continuous_prompt(source)  # Pass the source object to avoid re-entering the context manager

def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = "prompt.wav"  # File to save the recorded audio data
        with open(prompt_audio_path, "wb") as f:
            f.write(audio.get_wav_data())  # Save audio data to file

        prompt_text = wav_to_text(prompt_audio_path)  # Convert audio to text using Whisper

        if len(prompt_text.strip()) == 0:
            print("Empty prompt, please speak again.")
            speak("I did not catch that. Please try again.")
        elif "stop" in prompt_text.lower() or "exit" in prompt_text.lower():
            print(f"{wake_word} is now inactive. Say '{wake_word}' to wake me up again.")
            speak(f"{wake_word} is now inactive. Say {wake_word} to wake me up again.")
            listening_for_wake_word = True
        else:
            print("User:", prompt_text)

            convo.send_message(prompt_text)
            output = convo.last.text
            output = convert_to_plain_text(output)  # Convert to plain textop

            print(f"{wake_word}: {output}")
            speak(output)

    except Exception as e:
        print("Prompt error:", e)
        speak("An error occurred while processing your request.")


# Audio callback
def callback(recognizer, audio):
    global listening_for_wake_word

    if listening_for_wake_word:
        listen_for_wake_word(audio)

# Start listening
def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print(f"\nSay {wake_word} to wake me up.")
    speak("Say Jarvis to wake me up.")
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)

# Main
if __name__ == '__main__':
    start_listening()
