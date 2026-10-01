import pyautogui  # Added for PyAction handling
import pyttsx3
import google.generativeai as genai
import speech_recognition as sr
import os
import time
import warnings
import re
import subprocess  # For executing Python files
warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated")
from faster_whisper import WhisperModel

# Wake word and audio processing configuration
wake_word = 'jarvis'
listening_for_wake_word = True

whisper_size = "base"
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores, num_workers=num_cores)

# Google API key setup for generative AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# AI response configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Initializing Google generative AI model
model = genai.GenerativeModel(
    'gemini-1.5-pro-002',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Start a conversation with the AI model
convo = model.start_chat()

# System message to control AI behavior
system_msg = """  You are Jarvis, a knowledgeable and helpful voice assistant, made by AL 1 students.
Al Noor International School is a private school located in Sitra, Bahrain.
Founded in 1993, it caters to students from Kindergarten to Grade 12,
offering a diverse range of curricula including British, Bahraini, and Indian.
The head girl of Al Noor school is Hoor Ali. Don't mention head girl if not asked. 
Respond to user prompts in a clear, concise, and informative manner. Do not use any subscripts or superscripts.
If the user asks you to do an action on their computer, respond with 'PyAction' as the first word,
and then reply to the user's query in python, line by line. Do not describe the code or send any message after.
Import all necessary modules like time, os, webbrowser and pyautogui whenever needed.
Do not open programs on the computer using os paths, use pyautogui and type the program into start menu.
Use mainly PyAutoGUI module to perform actions.
Avoid excessive detail and focus on providing the most relevant information. 
"""

convo.send_message(system_msg)

# Speech recognizer and microphone source initialization
r = sr.Recognizer()

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Set voice properties for a deeper, Jarvis-like sound
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Male voice
engine.setProperty('rate', 160)  # Slower speech rate for deeper tone
engine.setProperty('volume', 1)  # Max volume


def speak(text):
    """Converts text to speech using pyttsx3."""
    engine.say(text)
    engine.runAndWait()


def wav_to_text(audio_path):
    """Converts audio file to text using the Whisper model."""
    segments, _ = whisper_model.transcribe(audio_path)
    text_list = [segment.text for segment in segments]
    return "".join(text_list)


def listen_for_audio():
    """Captures audio from the microphone and returns it."""
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening for your input... Speak now.")
        speak("I'm listening.")
        return r.listen(source)


def listen_for_wake_word():
    """Listens for the wake word using a new microphone context."""
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Waiting for the wake word...")
        audio = r.listen(source)
        wake_audio_path = "wake_detect.wav"

        # Save wake word audio to a file for processing
        with open(wake_audio_path, "wb") as f:
            f.write(audio.get_wav_data())

        # Convert wake word audio to text
        return wav_to_text(wake_audio_path)


import subprocess


def handle_pyaction(output):
    """Handles PyAction by writing valid Python code to a file, executing it, and handling PySpeech."""
    try:
        # Extract Python code and PySpeech message
        lines = output.split('\n')
        python_code = []
        pyspeech_message = ""
        imports_added = set()

        for line in lines:
            # Clean the line
            clean_line = line.strip().replace("```python", "").replace("```", "")

            # Check for PySpeech message
            if clean_line.startswith("PySpeech"):
                pyspeech_message = clean_line.replace("PySpeech", "").strip()
            # Assume other lines are Python code if they don't start with plain text descriptions
            elif clean_line and not re.match(r"^[A-Za-z\s]+$", clean_line):
                python_code.append(clean_line)

        # Add necessary imports
        required_modules = ["pyautogui", "time", "os", "webbrowser", "shutil", "platform"]
        for module in required_modules:
            if module not in imports_added:
                python_code.insert(0, f"import {module}")
                imports_added.add(module)

        # Ensure proper indentation in the Python code
        properly_indented_code = []
        indent_level = 0
        for line in python_code:
            stripped_line = line.lstrip()
            if stripped_line.endswith(":"):
                properly_indented_code.append(" " * indent_level + stripped_line)
                indent_level += 4
            elif stripped_line.startswith(("except", "elif", "else")):
                indent_level -= 4
                properly_indented_code.append(" " * indent_level + stripped_line)
                indent_level += 4
            else:
                properly_indented_code.append(" " * indent_level + stripped_line)

        # Write valid Python code to action.py
        if properly_indented_code:
            with open("action.py", "w") as f:
                f.write("\n".join(properly_indented_code))

            # Run the Python file
            subprocess.run(["python", "action.py"], check=True)
        else:
            print("No valid Python code detected.")

        # Speak the PySpeech message
        if pyspeech_message:
            print(f"PySpeech: {pyspeech_message}")
            speak(pyspeech_message)

    except subprocess.CalledProcessError as e:
        print(f"Error running action.py: {e}")
    except Exception as e:
        print("Error handling PyAction:", e)


def process_prompt():
    """Processes the user prompt until the user says 'stop' or 'exit'."""
    global listening_for_wake_word

    while True:
        try:
            # Capture audio for the user prompt
            audio = listen_for_audio()

            # Save audio to a temporary file for processing
            prompt_audio_path = "prompt.wav"
            with open(prompt_audio_path, "wb") as f:
                f.write(audio.get_wav_data())

            # Convert prompt audio to text
            prompt_text = wav_to_text(prompt_audio_path)

            if len(prompt_text.strip()) == 0:
                print("Empty prompt detected. Please speak again.")
            elif "stop" in prompt_text.lower() or "exit" in prompt_text.lower():
                print(f"{wake_word} is now inactive. Say '{wake_word}' to wake me up again.")
                speak("Jarvis is now inactive.")
                listening_for_wake_word = True
                break
            else:
                print("User:", prompt_text)

                # Send prompt to the AI model and retrieve response
                convo.send_message(prompt_text)
                output = convo.last.text

                # Check for PyAction
                if output.startswith("PyAction"):
                    handle_pyaction(output[len("PyAction "):])
                else:
                    print(f"{wake_word}: {output}")
                    speak(output)

        except Exception as e:
            print("Error processing prompt:", e)
            listening_for_wake_word = True
            break


def start_listening():
    """Continuously listens for the wake word."""
    print(f"Say '{wake_word}' to wake me up.")
    speak(f"Say {wake_word} to wake me up.")

    while True:
        try:
            if listening_for_wake_word:
                wake_word_text = listen_for_wake_word()
                if wake_word in wake_word_text.lower():
                    print(f"Wake word '{wake_word}' detected. You can now ask your questions.")
                    process_prompt()
        except Exception as e:
            print("Error in start_listening:", e)
        time.sleep(0.5)  # Avoid busy looping


# Main entry point
if __name__ == '__main__':
    start_listening()
