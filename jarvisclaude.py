"""
JARVIS AI ASSISTANT
===================
pip install google-generativeai faster-whisper speechrecognition pyautogui
pip install pystray pillow edge-tts pygame pygetwindow

Free Gemini API key: https://aistudio.google.com
Model: gemini-2.5-flash  (250 req/day free tier)
"""

import os, time, warnings, re, subprocess, threading, json, asyncio, base64, tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import google.generativeai as genai
import pyautogui, pystray
from PIL import Image, ImageDraw
import pygame, edge_tts

warnings.filterwarnings("ignore")
from faster_whisper import WhisperModel

pyautogui.FAILSAFE = True   # move mouse to top-left to abort any runaway action

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG_FILE = "jarvis_config.json"
DEFAULT_CONFIG = {
    "api_key":      "",
    "wake_word":    "jarvis",
    "volume":       80,
    "whisper_size": "tiny",
    "tts_voice":    "en-US-ChristopherNeural",
    "model":        "gemini-2.5-flash",   # 250 req/day free — do NOT use flash-lite (only 20/day)
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

config = load_config()

# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────
listening_for_wake_word = True
assistant_active        = True
whisper_model           = None
convo                   = None
tray_icon               = None
bar_window              = None
status_var              = None
_settings_open          = False

# ─────────────────────────────────────────────────────────────
#  WHISPER STT
# ─────────────────────────────────────────────────────────────
def load_whisper():
    global whisper_model
    n = os.cpu_count()
    whisper_model = WhisperModel(
        config["whisper_size"], device="cpu",
        compute_type="int8", cpu_threads=n, num_workers=n
    )

def wav_to_text(path):
    segs, _ = whisper_model.transcribe(path)
    return "".join(s.text for s in segs).strip()

# ─────────────────────────────────────────────────────────────
#  TEXT CLEANING — strip ALL markdown before TTS
# ─────────────────────────────────────────────────────────────
def clean_for_speech(text: str) -> str:
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)        # bold/italic
    text = re.sub(r'_{1,3}(.*?)_{1,3}',   r'\1', text)        # underscore emphasis
    text = re.sub(r'`{1,3}.*?`{1,3}',     '',    text, flags=re.DOTALL)  # code blocks
    text = re.sub(r'^#{1,6}\s*',           '',    text, flags=re.MULTILINE)  # headers
    text = re.sub(r'^\s*[\*\-\+]\s+',     '',    text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'^\s*\d+\.\s+',        '',    text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)     # markdown links
    text = re.sub(r'<https?://[^>]+>',    '',    text)         # bare URLs
    text = re.sub(r'\n+',                 ' ',   text)
    text = re.sub(r'  +',                 ' ',   text)
    return text.strip()

# ─────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_MSG = """You are {name}, a voice-controlled AI assistant running on the user's Windows PC.

PERSONALITY:
- Maximum 1-2 short sentences per reply. Never more.
- Brief apology when something fails. Brief greeting on wake. Short acknowledgements only.
- CRITICAL: Plain text only. Zero markdown. No asterisks, no hyphens as lists, no hash symbols, no backticks, no bold, no italics, no numbered lists. Write exactly as you would speak out loud.

COMPUTER ACTIONS:
When asked to do anything on the computer, output ONLY:
  Line 1:  PyAction
  Line 2+: Pure Python code
  Last line (optional): PySpeech Your spoken confirmation here.

Rules for the Python code:
- Do NOT add import statements — they are added automatically. Just write the action code directly.
- Do NOT add comments, docstrings, or any natural language lines.
- Do NOT wrap code in functions or classes.
- To open an installed app: press win key, type name, wait, press enter.
  Example:
    pyautogui.press('win')
    time.sleep(1.5)
    pyautogui.write('discord')
    time.sleep(1.5)
    pyautogui.press('enter')
- To open a website: webbrowser.open('https://...')
- To type text somewhere: pyautogui.click(x, y) then pyautogui.write('text') or pyautogui.hotkey('ctrl','v')
- To click UI elements: pyautogui.click(x, y) with screen coordinates
- To scroll: pyautogui.scroll(amount)
- To use keyboard shortcuts: pyautogui.hotkey('ctrl', 'c')

SCREEN-AWARE ACTIONS:
When you receive a SCREEN_CONTEXT message, you know the full layout of the screen including element positions.
Use those coordinates for click/type actions on visible UI elements.

If you receive ERROR_LOG: your previous code failed. Fix it and output corrected PyAction code only.

SCREEN VISION:
When given a screenshot, respond with one short plain sentence describing what is visible."""

def build_system_msg():
    return SYSTEM_MSG.format(name=config["wake_word"].capitalize())

# ─────────────────────────────────────────────────────────────
#  GEMINI — 429 retry backoff
# ─────────────────────────────────────────────────────────────
def init_gemini():
    global convo
    if not config["api_key"]:
        return False
    try:
        genai.configure(api_key=config["api_key"])
        mdl = genai.GenerativeModel(
            config["model"],
            generation_config={"temperature": 0.2, "max_output_tokens": 300},
            safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in [
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]],
        )
        convo = mdl.start_chat()
        convo.send_message(build_system_msg())
        return True
    except Exception as e:
        print(f"Gemini init error: {e}")
        return False

def ask_gemini(prompt, image=None, retries=3):
    for attempt in range(retries):
        try:
            if image:
                genai.configure(api_key=config["api_key"])
                vm = genai.GenerativeModel(config["model"])
                resp = vm.generate_content([prompt, image])
                return resp.text.strip()
            else:
                convo.send_message(prompt)
                return convo.last.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 20
                m = re.search(r'seconds:\s*(\d+)', err) or re.search(r'retry.*?(\d+)', err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/{retries})")
                set_status(f"Rate limit — {wait}s")
                speak(f"Rate limit reached. Waiting {wait} seconds.")
                time.sleep(wait)
            else:
                print(f"Gemini error: {e}")
                return None
    speak("API limit reached. Please try again in a minute.")
    return None

# ─────────────────────────────────────────────────────────────
#  TTS — interrupt support + Windows file-lock fix
# ─────────────────────────────────────────────────────────────
pygame.mixer.init()
_tts_lock    = threading.Lock()
_stop_speech = threading.Event()
_TTS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tts_out.mp3")

def stop_speaking():
    _stop_speech.set()
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

def speak(text):
    if not text or not text.strip():
        return
    text = clean_for_speech(text)
    if not text:
        return
    with _tts_lock:
        _stop_speech.clear()
        try:
            async def _gen():
                await edge_tts.Communicate(text, voice=config["tts_voice"]).save(_TTS_FILE)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            asyncio.run(_gen())
            pygame.mixer.music.load(_TTS_FILE)
            pygame.mixer.music.set_volume(config["volume"] / 100)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if _stop_speech.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"TTS error: {e}")

# ─────────────────────────────────────────────────────────────
#  MICROPHONE
# ─────────────────────────────────────────────────────────────
r = sr.Recognizer()

def listen_for_audio():
    with sr.Microphone() as src:
        r.adjust_for_ambient_noise(src, duration=0.4)
        return r.listen(src, timeout=10, phrase_time_limit=15)

def capture_audio_to_text(filename="prompt.wav"):
    audio = listen_for_audio()
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    return wav_to_text(filename)

# ─────────────────────────────────────────────────────────────
#  SCREEN CAPTURE
# ─────────────────────────────────────────────────────────────
SCREEN_TRIGGERS = [
    "look at my screen", "what do you see", "what's on my screen",
    "see my screen", "check my screen", "read my screen",
    "what is on my screen", "describe my screen", "can you see",
]

def get_screenshot_image():
    try:
        import PIL.ImageGrab
        return PIL.ImageGrab.grab()
    except Exception as e:
        print(f"Screenshot error: {e}")
        return None

def get_screenshot_with_coords():
    """
    Returns a PIL Image annotated with a grid of coordinate markers.
    This helps the AI reference exact pixel positions for click actions.
    """
    try:
        import PIL.ImageGrab, PIL.ImageDraw, PIL.ImageFont
        img = PIL.ImageGrab.grab()
        draw = PIL.ImageDraw.Draw(img)
        w, h = img.size
        step = 100
        for x in range(0, w, step):
            for y in range(0, h, step):
                draw.ellipse([x-3, y-3, x+3, y+3], fill=(255, 0, 0, 180))
                draw.text((x+5, y), f"{x},{y}", fill=(255, 255, 0))
        return img
    except Exception as e:
        print(f"Annotated screenshot error: {e}")
        return get_screenshot_image()

# ─────────────────────────────────────────────────────────────
#  PYACTION PARSER — fixed deduplication + PySpeech detection
# ─────────────────────────────────────────────────────────────

# Modules we always auto-import so the AI never needs to write them
AUTO_IMPORTS = ["pyautogui", "time", "os", "webbrowser", "subprocess", "shutil"]

# Regex: a line that looks like plain English (not Python)
_NATURAL_LANG = re.compile(
    r'^[A-Z][a-zA-Z\s\',\.\!\?]+$'   # starts uppercase, only word chars + punctuation
)

def extract_code_and_speech(raw: str):
    """
    Parse the AI's PyAction response.
    Returns (code_lines: list[str], speech: str)
    - Strips all import lines (we add them ourselves)
    - Strips markdown fences
    - Detects PySpeech line for spoken confirmation
    - Strips any natural-language lines that leaked in
    """
    speech = ""
    code_lines = []

    for line in raw.strip().splitlines():
        stripped = line.strip()

        # Remove markdown fences
        if stripped.startswith("```"):
            continue

        # PySpeech line — capture spoken text, do NOT put in code
        if re.match(r'^PySpeech\b', stripped, re.IGNORECASE):
            speech = re.sub(r'^PySpeech\s*:?\s*', '', stripped, flags=re.IGNORECASE).strip()
            continue

        # Skip ALL import lines — we add our own clean set
        if re.match(r'^import\s+\S', stripped) or re.match(r'^from\s+\S+\s+import', stripped):
            continue

        # Skip blank lines at this stage (we'll join cleanly later)
        if not stripped:
            continue

        # Skip lines that are clearly natural language sentences (AI slipping in prose)
        # Heuristic: no Python operators/brackets/colons and starts with capital letter
        has_python_chars = bool(re.search(r'[=\(\)\[\]\{\}:\'\"\\+\*/%<>!&|,]', stripped))
        is_keyword = stripped.split()[0] in (
            'if','else','elif','for','while','try','except','finally',
            'with','def','class','return','import','from','pass','break',
            'continue','raise','yield','print','pyautogui','time','os',
            'webbrowser','subprocess','shutil','True','False','None'
        ) if stripped else False

        if not has_python_chars and not is_keyword and _NATURAL_LANG.match(stripped):
            print(f"[Parser] Skipping natural-language line: {stripped}")
            continue

        code_lines.append(stripped)

    return code_lines, speech


def build_action_script(code_lines: list) -> str:
    """Prepend clean auto-imports and fix indentation."""
    # Build import header — clean, one per line, no duplicates
    header = [f"import {mod}" for mod in AUTO_IMPORTS]

    all_lines = header + [""] + code_lines  # blank line between imports and code

    indented, indent = [], 0
    for line in all_lines:
        s = line  # already stripped in extract step; preserve blank lines
        stripped = s.lstrip()
        if not stripped:
            indented.append("")
            continue
        if stripped.startswith(("except", "elif", "else", "finally")):
            indented.append(" " * max(0, indent - 4) + stripped)
        elif stripped.endswith(":"):
            indented.append(" " * indent + stripped)
            indent += 4
        else:
            indented.append(" " * indent + stripped)

    return "\n".join(indented)


def run_action_script(script: str):
    """Write action.py and execute it. Returns (success, stderr)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    try:
        result = subprocess.run(["python", path], capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired: script took too long."
    except Exception as e:
        return False, str(e)


MAX_ACTION_RETRIES = 2

def handle_pyaction(ai_response_text: str, original_request: str, retries_left=MAX_ACTION_RETRIES):
    code_lines, speech = extract_code_and_speech(ai_response_text)

    if not code_lines:
        speak("Sorry, no valid action was generated.")
        return

    script = build_action_script(code_lines)
    print(f"--- action.py ---\n{script}\n-----------------")

    success, stderr = run_action_script(script)

    if success:
        if speech:
            speak(speech)
        return

    # ── Failed ──
    print(f"PyAction failed:\n{stderr}")

    if retries_left <= 0:
        speak("Sorry, I kept running into errors and could not complete that.")
        return

    speak("Apologies, that failed. Let me try again.")
    set_status("Retrying...")

    fixed = ask_gemini(
        f"ERROR_LOG: Your previous PyAction code failed.\n"
        f"Original request: {original_request}\n"
        f"Python error:\n{stderr}\n\n"
        f"Rules reminder: Do NOT write any import statements. Do NOT write natural language lines. "
        f"Output PyAction then pure Python code only, then optionally PySpeech <message>."
    )

    if fixed and fixed.strip().startswith("PyAction"):
        handle_pyaction(fixed[len("PyAction"):].strip(), original_request, retries_left - 1)
    else:
        speak(f"Sorry, could not complete that.")

# ─────────────────────────────────────────────────────────────
#  STATUS / INDICATOR
# ─────────────────────────────────────────────────────────────
def set_status(text):
    print(f"[{text}]")
    if status_var:
        try:
            status_var.set(text)
        except Exception:
            pass

def set_indicator(active: bool):
    if not bar_window:
        return
    try:
        bar_window.nametowidget("indicator").config(bg="#00ff88" if active else "#444444")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
#  CORE CONVERSATION LOOP
# ─────────────────────────────────────────────────────────────
def manual_wake():
    global listening_for_wake_word
    if not listening_for_wake_word:
        return
    listening_for_wake_word = False
    set_status("Active")
    set_indicator(True)
    threading.Thread(target=process_prompt, daemon=True).start()

def process_prompt():
    global listening_for_wake_word
    speak(f"{config['wake_word'].capitalize()} online. Ready.")

    while True:
        try:
            set_status("Listening...")
            prompt_text = capture_audio_to_text()

            stop_speaking()  # interrupt if still talking

            if not prompt_text:
                speak("Didn't catch that.")
                continue

            lower = prompt_text.lower()
            print(f"You: {prompt_text}")

            # Sleep
            if any(w in lower for w in ["stop", "sleep", "go to sleep", "goodbye", "shut down"]):
                speak("Going to sleep.")
                listening_for_wake_word = True
                set_status("Sleeping")
                set_indicator(False)
                break

            set_status("Thinking...")

            # Determine if this needs screen context
            needs_screen = any(t in lower for t in SCREEN_TRIGGERS)

            # Action requests that benefit from seeing the screen
            # (clicking, typing into specific UI elements, finding buttons, etc.)
            needs_screen_for_action = any(t in lower for t in [
                "click", "open the", "press", "type", "write", "find", "select",
                "scroll", "close the", "the button", "that tab", "that window",
                "the input", "the field", "the box", "that message", "that chat",
            ])

            if needs_screen or needs_screen_for_action:
                set_status("Capturing screen...")
                img = get_screenshot_with_coords() if needs_screen_for_action else get_screenshot_image()
                if img:
                    if needs_screen:
                        # Vision request: get full desc silently, speak short summary
                        full_desc = ask_gemini(
                            f"Describe this screen in full detail, including positions of UI elements: {prompt_text}",
                            image=img
                        )
                        if full_desc:
                            convo.send_message(
                                f"SCREEN_CONTEXT (silent — use for follow-up actions, do not read aloud): {full_desc}"
                            )
                        output = ask_gemini(
                            "In one short spoken sentence, tell the user what you can see. Plain text only."
                        )
                    else:
                        # Action request with screen context
                        screen_prompt = (
                            f"The user wants to: {prompt_text}\n"
                            f"Here is the current screen with coordinate markers. "
                            f"Use the visible coordinates to click the correct elements. "
                            f"Output PyAction code only."
                        )
                        output = ask_gemini(screen_prompt, image=img)
                else:
                    output = ask_gemini(prompt_text)
            else:
                output = ask_gemini(prompt_text)

            if output is None:
                set_status("Listening...")
                continue

            print(f"AI: {output}")

            if output.strip().startswith("PyAction"):
                set_status("Running action...")
                handle_pyaction(output[len("PyAction"):].strip(), prompt_text)
            else:
                set_status("Speaking...")
                speak(output)

        except sr.WaitTimeoutError:
            speak("Timed out. Say my name to wake me.")
            listening_for_wake_word = True
            set_status("Sleeping")
            set_indicator(False)
            break
        except Exception as e:
            print(f"Prompt error: {e}")
            speak("Something went wrong.")
            listening_for_wake_word = True
            set_status("Sleeping")
            set_indicator(False)
            break

def start_listening_loop():
    global listening_for_wake_word
    set_status("Sleeping")

    while assistant_active:
        try:
            if listening_for_wake_word:
                with sr.Microphone() as src:
                    r.adjust_for_ambient_noise(src, duration=0.4)
                    audio = r.listen(src)
                with open("wake_detect.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                text = wav_to_text("wake_detect.wav")
                if config["wake_word"].lower() in text.lower():
                    listening_for_wake_word = False
                    set_status("Active")
                    set_indicator(True)
                    process_prompt()
        except Exception as e:
            print(f"Wake loop error: {e}")
        time.sleep(0.2)

# ─────────────────────────────────────────────────────────────
#  TRAY ICON
# ─────────────────────────────────────────────────────────────
def _tray_image():
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, 60, 60], fill=(0, 180, 255))
    d.text((18, 16), "AI", fill=(0, 0, 0))
    return img

def setup_tray():
    global tray_icon
    menu = pystray.Menu(
        pystray.MenuItem("Settings", lambda i, it: threading.Thread(
            target=open_settings_window, daemon=True).start()),
        pystray.MenuItem("Quit", lambda i, it: os._exit(0)),
    )
    tray_icon = pystray.Icon("Jarvis", _tray_image(),
                              f"{config['wake_word'].capitalize()} AI", menu)
    tray_icon.run()

# ─────────────────────────────────────────────────────────────
#  MINI BAR
# ─────────────────────────────────────────────────────────────
def open_mini_bar():
    global bar_window, status_var
    BG = "#111111"; ACCENT = "#00c8ff"; FG = "#aaaaaa"
    F = ("Segoe UI", 9); FB = ("Segoe UI", 9, "bold")

    win = tk.Tk()
    win.title("")
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.attributes("-alpha", 0.93)
    win.configure(bg=BG)
    bar_window = win

    W, H = 380, 36
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{W}x{H}+{sw-W-10}+{sh-H-48}")

    _d = {"x": 0, "y": 0}
    def ds(e): _d["x"] = e.x; _d["y"] = e.y
    def dm(e): win.geometry(f"+{win.winfo_x()+e.x-_d['x']}+{win.winfo_y()+e.y-_d['y']}")
    def bd(w): w.bind("<ButtonPress-1>", ds); w.bind("<B1-Motion>", dm)

    dot = tk.Label(win, name="indicator", bg="#444444", width=2)
    dot.pack(side="left", padx=(6, 0), pady=5, fill="y")
    bd(dot)

    nm = tk.Label(win, text=config["wake_word"].upper(), bg=BG, fg=ACCENT, font=FB)
    nm.pack(side="left", padx=(6, 2)); bd(nm)

    status_var = tk.StringVar(value="Initializing...")
    sl = tk.Label(win, textvariable=status_var, bg=BG, fg=FG, font=F, width=16, anchor="w")
    sl.pack(side="left"); bd(sl)

    tk.Button(win, text="▶ WAKE",
              command=lambda: threading.Thread(target=manual_wake, daemon=True).start(),
              bg="#1a1a2e", fg=ACCENT, font=FB, relief="flat", padx=8, cursor="hand2",
              activebackground=ACCENT, activeforeground="#000"
              ).pack(side="left", padx=4)

    tk.Button(win, text="⚙",
              command=lambda: threading.Thread(target=open_settings_window, daemon=True).start(),
              bg=BG, fg="#666", font=FB, relief="flat", padx=5, cursor="hand2",
              activebackground="#222", activeforeground=ACCENT
              ).pack(side="left", padx=2)

    tk.Button(win, text="✕", command=win.withdraw,
              bg=BG, fg="#444", font=F, relief="flat", padx=5, cursor="hand2",
              activebackground="#200", activeforeground="#f55"
              ).pack(side="right", padx=4)

    win.mainloop()

# ─────────────────────────────────────────────────────────────
#  SETTINGS WINDOW
# ─────────────────────────────────────────────────────────────
def open_settings_window():
    global _settings_open
    if _settings_open:
        return
    _settings_open = True

    BG = "#0d0d0d"; FG = "#e0e0e0"; ACCENT = "#00c8ff"; EB = "#1e1e1e"; EF = "#fff"
    F = ("Segoe UI", 10); FB = ("Segoe UI", 10, "bold")

    parent = bar_window if (bar_window and bar_window.winfo_exists()) else None
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Settings"); win.geometry("460x520")
    win.resizable(False, False); win.configure(bg=BG)
    win.attributes("-topmost", True)

    def on_close():
        global _settings_open; _settings_open = False; win.destroy()
    win.protocol("WM_DELETE_WINDOW", on_close)

    tk.Label(win, text="SETTINGS", font=("Segoe UI", 13, "bold"),
             bg=BG, fg=ACCENT).pack(pady=(18, 4))

    fr = tk.Frame(win, bg=BG); fr.pack(padx=28, fill="both", expand=True)

    def lbl(t):
        tk.Label(fr, text=t, font=FB, bg=BG, fg=ACCENT, anchor="w").pack(fill="x", pady=(10, 2))

    lbl("Gemini API Key  (aistudio.google.com)")
    api_v = tk.StringVar(value=config["api_key"])
    tk.Entry(fr, textvariable=api_v, show="*", bg=EB, fg=EF,
             insertbackground=ACCENT, font=F, relief="flat", bd=4).pack(fill="x", ipady=5)

    lbl("Wake Word  (single English word)")
    wake_v = tk.StringVar(value=config["wake_word"])
    tk.Entry(fr, textvariable=wake_v, bg=EB, fg=EF,
             insertbackground=ACCENT, font=F, relief="flat", bd=4).pack(fill="x", ipady=5)

    lbl("Volume")
    vrow = tk.Frame(fr, bg=BG); vrow.pack(fill="x")
    vol_v = tk.IntVar(value=config["volume"])
    vol_n = tk.Label(vrow, text=str(config["volume"]), bg=BG, fg=FG, font=F, width=4)
    def on_vol(v):
        vol_n.config(text=str(int(float(v))))
        pygame.mixer.music.set_volume(int(float(v)) / 100)
    tk.Scale(vrow, from_=0, to=100, orient="horizontal", variable=vol_v, command=on_vol,
             bg=BG, fg=FG, troughcolor="#333", activebackground=ACCENT, highlightthickness=0,
             sliderlength=14, length=310).pack(side="left", fill="x", expand=True)
    vol_n.pack(side="left", padx=6)

    lbl("Speech Recognition")
    srow = tk.Frame(fr, bg=BG); srow.pack(fill="x")
    sz_v = tk.StringVar(value=config["whisper_size"])
    for val, lab in [("tiny", "Tiny (fast)"), ("base", "Base"), ("small", "Small (best)")]:
        tk.Radiobutton(srow, text=lab, variable=sz_v, value=val, bg=BG, fg=FG,
                       selectcolor=EB, activebackground=BG, font=F).pack(side="left", padx=6)

    tk.Label(fr, text="Model: gemini-2.5-flash  |  250 free requests/day",
             bg=BG, fg="#444", font=("Segoe UI", 8), anchor="w").pack(fill="x", pady=(14, 0))

    brow = tk.Frame(win, bg=BG); brow.pack(pady=16)

    def save():
        global _settings_open
        wake = wake_v.get().strip().lower()
        if not re.match(r"^[a-z]+$", wake):
            messagebox.showerror("Invalid", "Wake word must be letters only."); return
        config["api_key"]      = api_v.get().strip()
        config["wake_word"]    = wake
        config["volume"]       = vol_v.get()
        config["whisper_size"] = sz_v.get()
        save_config(config)
        threading.Thread(target=load_whisper, daemon=True).start()
        if init_gemini():
            set_status("Sleeping")
            messagebox.showinfo("Saved", "Settings saved.")
            _settings_open = False; win.destroy()
        else:
            messagebox.showerror("Error", "Could not connect to Gemini. Check your API key.")

    def test():
        threading.Thread(
            target=speak,
            args=(f"Online. {wake_v.get().capitalize()} standing by.",),
            daemon=True
        ).start()

    tk.Button(brow, text="Save", command=save, bg=ACCENT, fg="#000", font=FB,
              relief="flat", padx=18, pady=7, cursor="hand2").pack(side="left", padx=8)
    tk.Button(brow, text="Test Voice", command=test, bg=EB, fg=FG, font=FB,
              relief="flat", padx=18, pady=7, cursor="hand2").pack(side="left", padx=8)

    if not parent:
        win.mainloop()

# ─────────────────────────────────────────────────────────────
#  FIRST-RUN SETUP
# ─────────────────────────────────────────────────────────────
def open_first_run_setup():
    global _settings_open
    if _settings_open:
        return
    _settings_open = True

    BG = "#0d0d0d"; FG = "#e0e0e0"; ACCENT = "#00c8ff"; EB = "#1e1e1e"; EF = "#fff"
    F = ("Segoe UI", 10); FB = ("Segoe UI", 10, "bold")

    win = tk.Tk()
    win.title("Setup"); win.geometry("460x300")
    win.resizable(False, False); win.configure(bg=BG)
    win.attributes("-topmost", True)

    def on_close():
        global _settings_open; _settings_open = False; win.destroy()
    win.protocol("WM_DELETE_WINDOW", on_close)

    tk.Label(win, text="FIRST-TIME SETUP", font=("Segoe UI", 13, "bold"),
             bg=BG, fg=ACCENT).pack(pady=(18, 4))
    tk.Label(win, text="Get a free API key at aistudio.google.com",
             bg=BG, fg=FG, font=F).pack(pady=(0, 10))

    fr = tk.Frame(win, bg=BG); fr.pack(padx=28, fill="x")

    tk.Label(fr, text="API Key", font=FB, bg=BG, fg=ACCENT, anchor="w").pack(fill="x", pady=(4, 2))
    api_v = tk.StringVar(value=config["api_key"])
    tk.Entry(fr, textvariable=api_v, show="*", bg=EB, fg=EF,
             insertbackground=ACCENT, font=F, relief="flat", bd=4).pack(fill="x", ipady=5)

    tk.Label(fr, text="Wake Word", font=FB, bg=BG, fg=ACCENT, anchor="w").pack(fill="x", pady=(10, 2))
    wake_v = tk.StringVar(value=config["wake_word"])
    tk.Entry(fr, textvariable=wake_v, bg=EB, fg=EF,
             insertbackground=ACCENT, font=F, relief="flat", bd=4).pack(fill="x", ipady=5)

    def save():
        global _settings_open
        wake = wake_v.get().strip().lower()
        if not re.match(r"^[a-z]+$", wake):
            messagebox.showerror("Invalid", "Wake word must be letters only."); return
        config["api_key"]   = api_v.get().strip()
        config["wake_word"] = wake
        save_config(config)
        _settings_open = False; win.destroy()

    tk.Button(win, text="Save and Start", command=save, bg=ACCENT, fg="#000", font=FB,
              relief="flat", padx=18, pady=8, cursor="hand2").pack(pady=20)
    win.mainloop()

# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────
def startup():
    print("Loading Whisper...")
    load_whisper()

    if not config["api_key"]:
        open_first_run_setup()

    print("Connecting to Gemini...")
    if not init_gemini():
        print("Gemini failed. Opening setup.")
        open_first_run_setup()
        if not init_gemini():
            print("Exiting — no valid API key.")
            return

    print(f"Ready. Say '{config['wake_word']}' or press WAKE.")
    threading.Thread(target=setup_tray, daemon=True).start()
    threading.Thread(target=start_listening_loop, daemon=True).start()
    open_mini_bar()


if __name__ == "__main__":
    startup()