"""
JARVIS AI ASSISTANT — Rebuilt
==============================
Models needed (pull with ollama):
  ollama pull qwen2.5:1.5b          # fast Q&A
  ollama pull qwen2.5:3b            # general actions
  ollama pull deepseek-r1:1.5b      # reasoning/code
  ollama pull qwen2.5-coder:3b      # coding tasks

Optional: Gemini API key at aistudio.google.com (free, for screen vision)

pip install google-generativeai faster-whisper speechrecognition pyautogui
pip install pystray pillow edge-tts pygame requests pyperclip PyQt6
pip install duckduckgo-search pygetwindow
"""

import os, sys, time, warnings, re, json, asyncio, subprocess, threading
import shutil, glob, math, tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
from pathlib import Path
import speech_recognition as sr
import pyautogui, pystray
from PIL import Image, ImageDraw
import pygame, edge_tts
import requests as http
import pyperclip

warnings.filterwarnings("ignore")
from faster_whisper import WhisperModel

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.06

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE    = os.path.join(BASE_DIR, "jarvis_config.json")
MEMORY_FILE    = os.path.join(BASE_DIR, "jarvis_memory.json")
SCRATCHPAD     = os.path.join(BASE_DIR, "jarvis_scratch.txt")
ACTION_PY      = os.path.join(BASE_DIR, "action.py")
TTS_FILE       = os.path.join(BASE_DIR, "_tts.mp3")
LOG_FILE       = os.path.join(BASE_DIR, "jarvis.log")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "wake_word":        "jarvis",
    "volume":           80,
    "whisper_size":     "tiny",
    "tts_voice":        "en-US-ChristopherNeural",
    "ollama_url":       "http://localhost:11434",
    "model_fast":       "qwen2.5:1.5b",
    "model_general":    "qwen2.5:3b",
    "model_reasoning":  "deepseek-r1:1.5b",
    "model_coder":      "qwen2.5-coder:3b",
    "gemini_api_key":   "",
    "gemini_model":     "gemini-2.5-flash",
    "use_gpu":          True,
    "search_engine":    "browser",   # browser = real chrome, ddg = duckduckgo
    "default_browser":  "chrome",
    "rate_limit_rpm":   20,
}

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

config = load_config()

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{level}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
#  LONG-TERM MEMORY
# ─────────────────────────────────────────────────────────────
def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {}

def save_memory(mem: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

def remember(key: str, value: str) -> str:
    mem = load_memory()
    mem[key] = {"value": value, "ts": datetime.now().isoformat()}
    save_memory(mem)
    return f"Remembered: {key} = {value}"

def recall(key: str) -> str:
    mem = load_memory()
    if key in mem:
        return mem[key]["value"]
    matches = [k for k in mem if key.lower() in k.lower()]
    return mem[matches[0]]["value"] if matches else ""

def memories_summary() -> str:
    mem = load_memory()
    if not mem:
        return ""
    return "\n".join(f"- {k}: {v['value']}" for k, v in list(mem.items())[-20:])

# ─────────────────────────────────────────────────────────────
#  SCRATCHPAD  (task working memory)
# ─────────────────────────────────────────────────────────────
def scratch_read() -> str:
    if os.path.exists(SCRATCHPAD):
        with open(SCRATCHPAD, encoding="utf-8") as f:
            return f.read(4000)
    return ""

def scratch_write(text: str):
    with open(SCRATCHPAD, "w", encoding="utf-8") as f:
        f.write(text)

def scratch_append(text: str):
    with open(SCRATCHPAD, "a", encoding="utf-8") as f:
        f.write("\n" + text)

def scratch_clear():
    if os.path.exists(SCRATCHPAD):
        os.remove(SCRATCHPAD)

# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────
listening_for_wake    = True
assistant_active      = True
whisper_model         = None
tray_icon             = None
orb_window            = None
_settings_open        = False
_speak_count          = 0
_energy_thresh        = 1500  # sensible default — avoids the near-zero calibration bug
_ollama_loaded        = {}   # model → bool
_last_screen_desc     = ""   # cached screen description
_last_screen_ts       = 0.0

# Rate limiting
_request_times: list  = []

# ─────────────────────────────────────────────────────────────
#  WHISPER
# ─────────────────────────────────────────────────────────────
def load_whisper():
    global whisper_model
    n = os.cpu_count()
    whisper_model = WhisperModel(
        config["whisper_size"], device="cpu",
        compute_type="int8", cpu_threads=n, num_workers=n
    )
    log(f"Whisper loaded: {config['whisper_size']}")

def wav_to_text(path: str) -> str:
    segs, _ = whisper_model.transcribe(path)
    return "".join(s.text for s in segs).strip()

# ─────────────────────────────────────────────────────────────
#  TEXT CLEANING
# ─────────────────────────────────────────────────────────────
def clean_speech(text: str) -> str:
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# ─────────────────────────────────────────────────────────────
#  RATE LIMITER
# ─────────────────────────────────────────────────────────────
def check_rate_limit() -> bool:
    global _request_times
    now = time.time()
    _request_times = [t for t in _request_times if now - t < 60]
    if len(_request_times) >= config["rate_limit_rpm"]:
        wait = 60 - (now - _request_times[0])
        speak(f"Rate limit reached. Waiting {int(wait)} seconds.")
        time.sleep(wait)
        _request_times = []
    _request_times.append(now)
    return True

# ─────────────────────────────────────────────────────────────
#  MODEL ROUTER  — pure string matching, zero latency
# ─────────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "vision": [
        "look at", "see my screen", "what's on", "what is on",
        "describe my screen", "read my screen", "check my screen",
        "can you see", "what do you see", "screen says",
    ],
    "reasoning": [
        "why does", "explain", "analyse", "analyze", "debug",
        "write code", "fix my code", "write a script", "write a program",
        "help me code", "what's wrong with", "review my", "improve my",
        "edit the file", "modify the file", "create a file",
    ],
    "browser": [
        "search for", "look up", "google", "find online",
        "what's the weather", "weather in", "latest news", "news about",
        "current price", "how much is", "what time is it in",
        "open youtube", "open instagram", "open", "go to",
    ],
    "system": [
        "shutdown", "restart", "turn off", "sleep mode",
        "volume up", "volume down", "mute", "brightness",
        "screenshot", "take a screenshot", "what time", "what's the time",
        "what's the date", "open app", "close app", "minimize",
    ],
    "memory": [
        "remember", "don't forget", "keep in mind", "recall",
        "what did i tell you", "do you remember", "forget",
    ],
    "file": [
        "read the file", "open the file", "create a file", "delete the file",
        "move the file", "copy the file", "list files", "find files",
        "save to", "write to", "show me the folder",
    ],
}

def route_intent(prompt: str) -> str:
    lower = prompt.lower()
    scores = {intent: 0 for intent in INTENT_PATTERNS}
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if p in lower:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "fast"
    return best

def model_for_intent(intent: str) -> str:
    mapping = {
        "vision":    config["model_general"],
        "reasoning": config["model_reasoning"],
        "browser":   config["model_general"],
        "system":    config["model_fast"],
        "memory":    config["model_fast"],
        "file":      config["model_coder"],
        "fast":      config["model_fast"],
    }
    return mapping.get(intent, config["model_fast"])

# ─────────────────────────────────────────────────────────────
#  OLLAMA BACKEND
# ─────────────────────────────────────────────────────────────
_ollama_histories: dict[str, list] = {}  # model → history list

def _system_prompt(model: str) -> str:
    ww  = config["wake_word"].capitalize()
    mem = memories_summary()
    mem_block = f"\nUSER FACTS:\n{mem}" if mem else ""
    scratch = scratch_read()
    scratch_block = f"\nWORKING NOTES:\n{scratch}" if scratch else ""

    return (
        f"You are {ww}, a voice AI agent controlling a Windows 11 PC.{mem_block}{scratch_block}\n"
        "Output ONLY a raw JSON object. Never write plain text outside JSON.\n"
        'Single: {"tool":"name","args":{}}\n'
        'Plan:   {"plan":[{"tool":"t","args":{}}]}\n'
        "ALWAYS use plan when speaking AND doing something.\n"
        "speak text = plain, max 2 sentences, no markdown.\n"
        "Use confirm before shutdown/delete. Use clarify if ambiguous.\n"
        "BROWSER ACTIONS: open apps and websites like a human would — "
        "use Win key to search and open apps, use keyboard to navigate. "
        "Never use webbrowser module. Always use pyautogui.\n"
        "TOOLS: speak,open_app,open_url_human,web_search,keyboard,type_text,"
        "mouse_click,mouse_move,scroll,get_clipboard,set_clipboard,"
        "screen_capture,find_on_screen,focus_window,shell,"
        "get_time,get_date,shutdown,restart,volume_set,notify,"
        "file_list,file_read,file_create,file_copy,file_move,file_delete,file_search,"
        "remember,recall,scratch_write,scratch_read,scratch_clear,"
        "clarify,confirm,reason,wait,run_script,self_read,self_edit"
    )

def ollama_available() -> bool:
    try:
        return http.get(f"{config['ollama_url']}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False

def _ensure_history(model: str):
    if model not in _ollama_histories:
        _ollama_histories[model] = [
            {"role": "system", "content": _system_prompt(model)}
        ]

def _warmup_model(model: str):
    try:
        http.post(f"{config['ollama_url']}/api/chat",
            json={"model": model,
                  "messages": [{"role":"user","content":"hi"}],
                  "stream": False,
                  "options": {"num_predict": 1}},
            timeout=60)
        _ollama_loaded[model] = True
        log(f"Warmed up: {model}")
    except Exception as e:
        log(f"Warmup failed ({model}): {e}", "WARN")

def ask_ollama(prompt: str, model: str, timeout: int = 45) -> str | None:
    check_rate_limit()
    _ensure_history(model)
    _ollama_histories[model].append({"role": "user", "content": prompt})

    # Rebuild system prompt with fresh memory/scratch on each call
    _ollama_histories[model][0] = {
        "role": "system",
        "content": _system_prompt(model)
    }

    try:
        resp = http.post(
            f"{config['ollama_url']}/api/chat",
            json={
                "model":   model,
                "messages": _ollama_histories[model],
                "stream":  True,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.1,
                    "top_k": 10,
                    "top_p": 0.9,
                }
            },
            stream=True,
            timeout=timeout
        )
        if resp.status_code != 200:
            log(f"Ollama {resp.status_code} for {model}", "ERR")
            return None

        full = ""
        for line in resp.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    full += chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        break
                except Exception:
                    continue

        reply = full.strip()
        _ollama_histories[model].append({"role": "assistant", "content": reply})

        # Keep history bounded
        if len(_ollama_histories[model]) > 24:
            _ollama_histories[model] = (
                [_ollama_histories[model][0]] +
                _ollama_histories[model][-22:]
            )
        return reply

    except http.exceptions.Timeout:
        log(f"Ollama timeout ({model})", "ERR")
        return None
    except Exception as e:
        log(f"Ollama error: {e}", "ERR")
        return None

# ─────────────────────────────────────────────────────────────
#  GEMINI BACKEND  (vision + fallback)
# ─────────────────────────────────────────────────────────────
_gemini_model_obj = None

def init_gemini() -> bool:
    global _gemini_model_obj
    if not config["gemini_api_key"]:
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=config["gemini_api_key"])
        _gemini_model_obj = genai.GenerativeModel(
            config["gemini_model"],
            generation_config={"temperature": 0.1, "max_output_tokens": 512},
        )
        log("Gemini ready")
        return True
    except Exception as e:
        log(f"Gemini init: {e}", "ERR")
        return False

def ask_gemini_vision(prompt: str, image) -> str | None:
    try:
        import google.generativeai as genai
        genai.configure(api_key=config["gemini_api_key"])
        vm = genai.GenerativeModel(config["gemini_model"])
        return vm.generate_content([prompt, image]).text.strip()
    except Exception as e:
        log(f"Gemini vision: {e}", "ERR")
        return None

def ask_gemini_text(prompt: str, retries: int = 2) -> str | None:
    for attempt in range(retries):
        try:
            resp = _gemini_model_obj.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                m = re.search(r'seconds:\s*(\d+)', err)
                wait = int(m.group(1)) + 2 if m else 20
                speak(f"Rate limit. Waiting {wait} seconds.")
                time.sleep(wait)
            else:
                log(f"Gemini text: {e}", "ERR")
                return None
    return None

# ─────────────────────────────────────────────────────────────
#  UNIFIED ASK
# ─────────────────────────────────────────────────────────────
def ask(prompt: str, intent: str = "fast", image=None) -> str | None:
    if image:
        if config["gemini_api_key"]:
            return ask_gemini_vision(prompt, image)
        return "Screen vision requires a Gemini API key in settings."

    model = model_for_intent(intent)
    result = ask_ollama(prompt, model)

    # Fallback chain: if primary model fails, try general, then gemini
    if result is None:
        log(f"Primary model {model} failed, trying general", "WARN")
        result = ask_ollama(prompt, config["model_general"])

    if result is None and config["gemini_api_key"] and _gemini_model_obj:
        log("Ollama failed, falling back to Gemini", "WARN")
        result = ask_gemini_text(prompt)

    if result is None:
        speak("I couldn't get a response. Please check that Ollama is running.")
    return result

# ─────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────
pygame.mixer.init()
_tts_lock    = threading.Lock()
_stop_speech = threading.Event()
_is_speaking = False

def stop_speaking():
    _stop_speech.set()
    try: pygame.mixer.music.stop()
    except Exception: pass

def speak(text: str):
    global _is_speaking
    if not text or not text.strip(): return
    text = clean_speech(text)
    if not text: return
    with _tts_lock:
        _stop_speech.clear()
        _is_speaking = True
        set_orb_state("speaking")
        try:
            async def _gen():
                await edge_tts.Communicate(
                    text, voice=config["tts_voice"]
                ).save(TTS_FILE)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            asyncio.run(_gen())
            pygame.mixer.music.load(TTS_FILE)
            pygame.mixer.music.set_volume(config["volume"] / 100)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if _stop_speech.is_set():
                    pygame.mixer.music.stop(); break
                time.sleep(0.05)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception as e:
            log(f"TTS: {e}", "ERR")
        finally:
            _is_speaking = False
            set_orb_state("idle")

# ─────────────────────────────────────────────────────────────
#  MICROPHONE
# ─────────────────────────────────────────────────────────────
_recognizer = sr.Recognizer()

def calibrate_mic():
    global _energy_thresh
    idx = _mic_index if _mic_index is not None else 1
    try:
        mic = sr.Microphone(device_index=idx)
        with mic as src:
            _recognizer.adjust_for_ambient_noise(src, duration=1.2)
            raw = _recognizer.energy_threshold
            _energy_thresh = max(raw * 1.5, 300)
            log(f"Mic calibrated (index {idx}): raw={raw:.0f} using={_energy_thresh:.0f}")
    except Exception as e:
        log(f"Mic calibration skipped: {e}", "WARN")

def _get_mic_index() -> int | None:
    """Find best available microphone index. Prefers real hardware over virtual."""
    try:
        names = sr.Microphone.list_microphone_names()
        # Prefer Intel Smart Sound or Realtek mic over virtual/mapper
        preferred = ["intel", "realtek", "microphone array", "microphone ("]
        for pref in preferred:
            for i, name in enumerate(names):
                if pref in name.lower() and "virtual" not in name.lower():
                    log(f"Using mic [{i}]: {name}")
                    return i
        # Fallback to index 1 (usually first real hardware mic)
        log(f"Using default mic index 1: {names[1] if len(names)>1 else 'unknown'}")
        return 1
    except Exception:
        return None  # let sr.Microphone use default

_mic_index = None  # cached after first successful use

def listen(timeout=20, phrase_limit=15):
    global _mic_index
    # Index 1 = Intel Smart Sound (confirmed working on this machine)
    # Try in order: cached → 1 → 7 → None(default)
    if _mic_index is None:
        _mic_index = 1
    indices_to_try = [_mic_index, 1, 7, None]
    # Deduplicate while preserving order
    seen = set()
    indices_to_try = [x for x in indices_to_try
                      if not (x in seen or seen.add(x))]
    last_err = None
    for idx in indices_to_try:
        try:
            mic = sr.Microphone(device_index=idx) if idx is not None else sr.Microphone()
            with mic as src:
                _recognizer.energy_threshold         = max(_energy_thresh, 300)
                _recognizer.dynamic_energy_threshold = False
                audio = _recognizer.listen(src, timeout=timeout,
                                            phrase_time_limit=phrase_limit)
            _mic_index = idx  # cache whichever worked
            return audio
        except sr.WaitTimeoutError:
            raise
        except Exception as e:
            last_err = e
            log(f"Mic index {idx} failed: {e}", "WARN")
            continue
    raise Exception(f"All mic indices failed. Last: {last_err}")

def capture_speech(filename: str = "prompt.wav") -> str:
    global _speak_count
    audio = listen()
    path = os.path.join(BASE_DIR, filename)
    with open(path, "wb") as f:
        f.write(audio.get_wav_data())
    _speak_count += 1
    if _speak_count % 3 == 0:
        threading.Thread(target=calibrate_mic, daemon=True).start()
    result = wav_to_text(path)
    log(f"Captured: '{result}'")
    return result

# ─────────────────────────────────────────────────────────────
#  SCREEN CAPTURE
# ─────────────────────────────────────────────────────────────
def get_screenshot():
    try:
        import PIL.ImageGrab
        return PIL.ImageGrab.grab()
    except Exception as e:
        log(f"Screenshot: {e}", "ERR"); return None

def get_screenshot_annotated():
    try:
        import PIL.ImageGrab, PIL.ImageDraw
        img = PIL.ImageGrab.grab()
        d   = PIL.ImageDraw.Draw(img)
        w, h = img.size
        for x in range(0, w, 150):
            for y in range(0, h, 150):
                d.ellipse([x-4,y-4,x+4,y+4], fill=(255,50,50))
                d.text((x+5,y-9), f"{x},{y}", fill=(255,255,0))
        return img
    except Exception:
        return get_screenshot()

def capture_and_describe(question: str = "Describe what is visible on the screen.") -> str:
    global _last_screen_desc, _last_screen_ts
    img = get_screenshot_annotated()
    if not img:
        return "Screen capture failed."
    desc = ask(
        f"Describe this screen in detail including all visible UI elements, "
        f"text, buttons, coordinates. Question: {question}",
        intent="vision",
        image=img
    ) or "Could not describe screen."
    _last_screen_desc = desc
    _last_screen_ts   = time.time()
    return desc

# ─────────────────────────────────────────────────────────────
#  HUMAN-LIKE BROWSER ACTIONS
# ─────────────────────────────────────────────────────────────
def open_app_human(name: str):
    """Open an app the way a human would — Win key, type, enter."""
    log(f"Opening app: {name}")
    pyautogui.press("win")
    time.sleep(0.9)
    pyautogui.write(name, interval=0.05)
    time.sleep(1.4)
    pyautogui.press("enter")
    time.sleep(1.2)

def open_url_human(url: str, browser: str = None):
    """Open a URL like a human — open browser, focus address bar, type URL."""
    browser = browser or config["default_browser"]
    log(f"Opening URL: {url}")
    open_app_human(browser)
    time.sleep(1.5)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.4)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyautogui.write(url, interval=0.04)
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(1.5)

def web_search_human(query: str):
    """Search Google like a human would."""
    log(f"Searching: {query}")
    open_url_human("google.com")
    time.sleep(1.5)
    # Google search box should be focused
    pyautogui.write(query, interval=0.04)
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(1.0)

def web_search_silent(query: str) -> str:
    """
    Get search results silently (no browser) for answering questions.
    Falls back to DuckDuckGo API, then Gemini.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if results:
            summary = "\n".join(
                f"{r['title']}: {r['body']}" for r in results
            )
            return summary[:1500]
    except Exception as e:
        log(f"DDG search: {e}", "WARN")

    # Fallback: Gemini web grounding if available
    if config["gemini_api_key"] and _gemini_model_obj:
        try:
            result = ask_gemini_text(
                f"Answer this question with current factual information: {query}"
            )
            return result or ""
        except Exception:
            pass
    return ""

# ─────────────────────────────────────────────────────────────
#  OVERLAY (PyQt6)
# ─────────────────────────────────────────────────────────────
_overlay_proc = None

def draw_overlay(shapes: list):
    global _overlay_proc
    clear_overlay()
    script = _make_overlay_script(json.dumps(shapes))
    tmp = os.path.join(BASE_DIR, "_overlay.py")
    with open(tmp, "w") as f:
        f.write(script)
    try:
        _overlay_proc = subprocess.Popen(["python", tmp])
    except Exception as e:
        log(f"Overlay: {e}", "ERR")

def clear_overlay():
    global _overlay_proc
    if _overlay_proc and _overlay_proc.poll() is None:
        _overlay_proc.terminate()
    _overlay_proc = None

def _make_overlay_script(shapes_json: str) -> str:
    return '''import sys, json
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush
shapes = ''' + shapes_json + '''
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint|Qt.WindowType.Tool|Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        s = QApplication.primaryScreen().size()
        self.setGeometry(0,0,s.width(),s.height())
        self.show()
        dur = shapes[0].get("duration",5)*1000 if shapes else 5000
        QTimer.singleShot(int(dur), QApplication.instance().quit)
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for s in shapes:
            c=QColor(s.get("color","#ff3030")); c.setAlpha(s.get("alpha",180))
            t=s.get("type","rect"); x,y,w,h=s.get("x",100),s.get("y",100),s.get("w",200),s.get("h",50)
            pen=QPen(c,s.get("thickness",3)); p.setPen(pen)
            if t=="rect":
                f=QColor(c); f.setAlpha(40); p.setBrush(QBrush(f)); p.drawRect(QRect(x,y,w,h)); p.setBrush(Qt.BrushStyle.NoBrush)
            elif t=="highlight":
                f=QColor(s.get("color","#ffff00")); f.setAlpha(100); p.setBrush(QBrush(f)); p.setPen(Qt.PenStyle.NoPen); p.drawRect(QRect(x,y,w,h))
            elif t=="text":
                p.setFont(QFont("Consolas",s.get("font_size",18))); p.drawText(x,y,s.get("text",""))
            elif t=="circle":
                p.drawEllipse(QRect(x,y,w,h))
app=QApplication(sys.argv); o=Overlay(); app.exec()
'''

# ─────────────────────────────────────────────────────────────
#  PYTHON SCRIPT FALLBACK
# ─────────────────────────────────────────────────────────────
def run_script(script: str) -> tuple[bool, str]:
    with open(ACTION_PY, "w", encoding="utf-8") as f:
        f.write(script)
    try:
        result = subprocess.run(
            ["python", ACTION_PY],
            capture_output=True, text=True, timeout=20
        )
        out = result.stdout.strip() or result.stderr.strip()
        return result.returncode == 0, out or "done"
    except subprocess.TimeoutExpired:
        return False, "Script timed out"
    except Exception as e:
        return False, str(e)

def ask_for_script(tool: str, args: dict, error: str = "") -> str | None:
    err_ctx = f" Error was: {error}" if error else ""
    prompt = (
        "Write a Python script to do: tool='" + tool +
        "', args=" + json.dumps(args) + "." + err_ctx +
        " Rules: print() the result. Import what you need "
        "(datetime,os,subprocess,pyautogui,webbrowser,ctypes etc). "
        "No input(). Output ONLY Python code."
    )
    raw = ask(prompt, intent="reasoning")
    if not raw: return None
    code = re.sub(r'^```python\s*', '', raw.strip(), flags=re.MULTILINE)
    code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
    return code.strip()


# ─────────────────────────────────────────────────────────────
#  TOOL CACHE
# ─────────────────────────────────────────────────────────────
_tool_cache: dict = {}
CACHE_TTL = {
    "get_clipboard": 5, "file_list": 10,
    "file_read": 30,    "recall": 60,
    "get_time": 30,     "get_date": 300,
}

def _cache_get(tool: str, args: dict) -> str | None:
    if tool not in CACHE_TTL: return None
    key = f"{tool}:{json.dumps(args,sort_keys=True)}"
    e   = _tool_cache.get(key)
    if e and time.time() - e["ts"] < CACHE_TTL[tool]:
        return e["result"]
    return None

def _cache_set(tool: str, args: dict, result: str):
    if tool not in CACHE_TTL: return
    key = f"{tool}:{json.dumps(args,sort_keys=True)}"
    _tool_cache[key] = {"result": result, "ts": time.time()}

# ─────────────────────────────────────────────────────────────
#  TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────
# Tools that need their result fed back to model for follow-up
CONTEXT_TOOLS = {
    "file_read","file_list","file_search","get_clipboard",
    "recall","screen_capture","find_on_screen","shell",
    "reason","web_search","scratch_read","self_read",
    "get_time","get_date",
}

def execute_tool(tool: str, args: dict) -> tuple[bool, str]:
    # Cache check
    cached = _cache_get(tool, args)
    if cached is not None:
        return True, cached

    ok, result = _execute_inner(tool, args)

    # Fallback: try Python script
    if not ok:
        log(f"Tool '{tool}' failed ({result[:60]}), trying script fallback")
        script = ask_for_script(tool, args, error=result)
        if script:
            ok2, result2 = run_script(script)
            if ok2:
                log(f"Script fallback OK: {result2[:60]}")
                _cache_set(tool, args, result2)
                return True, result2
            # Try once more with fixed script
            script2 = ask_for_script(tool, args, error=result2)
            if script2 and script2 != script:
                ok3, result3 = run_script(script2)
                if ok3:
                    _cache_set(tool, args, result3)
                    return True, result3
        return False, f"All fallbacks failed for '{tool}': {result}"

    _cache_set(tool, args, result)
    return ok, result


def _execute_inner(tool: str, args: dict) -> tuple[bool, str]:
    try:
        # ── SPEECH ──────────────────────────────────────────
        if tool == "speak":
            speak(args.get("text",""))
            return True, "spoken"

        # ── APP / URL / SEARCH ──────────────────────────────
        elif tool == "open_app":
            open_app_human(args.get("name",""))
            return True, f"Opened {args.get('name','')}"

        elif tool == "open_url_human":
            open_url_human(args.get("url",""), args.get("browser"))
            return True, "URL opened"

        elif tool == "web_search":
            mode = args.get("mode","silent")
            query = args.get("query","")
            if mode == "browser":
                web_search_human(query)
                return True, "Searched in browser"
            else:
                result = web_search_silent(query)
                return (True, result) if result else (False, "No results")

        # ── KEYBOARD / MOUSE ────────────────────────────────
        elif tool == "keyboard":
            keys = args.get("keys",[])
            if isinstance(keys, list): pyautogui.hotkey(*keys)
            else: pyautogui.hotkey(*str(keys).split("+"))
            return True, "Keys pressed"

        elif tool == "type_text":
            pyautogui.write(args.get("text",""), interval=0.04)
            return True, "Typed"

        elif tool == "mouse_click":
            x, y = args.get("x", 0), args.get("y", 0)
            btn  = args.get("button","left")
            pyautogui.click(x, y, button=btn)
            return True, f"Clicked {x},{y}"

        elif tool == "mouse_move":
            x, y = args.get("x",0), args.get("y",0)
            pyautogui.moveTo(x, y, duration=0.3)
            return True, f"Moved to {x},{y}"

        elif tool == "scroll":
            d = args.get("direction","down")
            a = int(args.get("amount",3))
            pyautogui.scroll(a if d=="up" else -a)
            return True, "Scrolled"

        # ── CLIPBOARD ───────────────────────────────────────
        elif tool == "get_clipboard":
            return True, pyperclip.paste()
        elif tool == "set_clipboard":
            pyperclip.copy(args.get("text",""))
            return True, "Clipboard set"

        # ── SCREEN ──────────────────────────────────────────
        elif tool == "screen_capture":
            desc = capture_and_describe(args.get("question","Describe the screen."))
            return True, desc

        elif tool == "find_on_screen":
            desc = args.get("description","")
            img  = get_screenshot_annotated()
            if not img: return False, "Screenshot failed"
            r = ask(
                f"Find UI element '{desc}'. Return JSON {{\"x\":N,\"y\":N}} only.",
                image=img
            )
            if r:
                m = re.search(r'\{[^}]+\}', r)
                if m: return True, m.group()
            return False, f"Could not find '{desc}'"

        elif tool == "highlight_text":
            text  = args.get("text","")
            color = args.get("color","#ffff00")
            dur   = args.get("duration",5)
            img   = get_screenshot_annotated()
            if not img: return False, "Screenshot failed"
            r = ask(
                f"Find '{text}' on screen. Return JSON {{\"x\":N,\"y\":N,\"w\":N,\"h\":N}}.",
                image=img
            )
            if r:
                m = re.search(r'\{[^}]+\}', r)
                if m:
                    c = json.loads(m.group())
                    draw_overlay([{"type":"highlight","x":c.get("x",0),
                                    "y":c.get("y",0),"w":c.get("w",200),
                                    "h":c.get("h",30),"color":color,"duration":dur}])
                    return True, f"Highlighted '{text}'"
            return False, f"Could not locate '{text}'"

        elif tool == "draw_on_screen":
            draw_overlay(args.get("shapes",[]))
            return True, "Overlay drawn"

        elif tool == "clear_overlay":
            clear_overlay()
            return True, "Cleared"

        # ── WINDOW ──────────────────────────────────────────
        elif tool == "focus_window":
            title = args.get("title","")
            try:
                import pygetwindow as gw
                wins = gw.getWindowsWithTitle(title)
                if wins: wins[0].activate(); return True, f"Focused {title}"
                return False, f"Window '{title}' not found"
            except Exception as e:
                return False, str(e)

        # ── SHELL ───────────────────────────────────────────
        elif tool == "shell":
            cmd = args.get("command","")
            blocked = ["format c:","rmdir /s /q c:\\","shutdown /r /f","del /f /s /q c:\\"]
            if any(b in cmd.lower() for b in blocked):
                return False, "Blocked: too destructive"
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            return r.returncode==0, (r.stdout or r.stderr).strip()[:500]

        # ── TIME / DATE ─────────────────────────────────────
        elif tool in ("get_time","time"):
            t = datetime.now().strftime("%I:%M %p")
            speak(f"The time is {t}.")
            return True, t

        elif tool in ("get_date","date"):
            d = datetime.now().strftime("%A, %B %d, %Y")
            speak(f"Today is {d}.")
            return True, d

        # ── POWER ───────────────────────────────────────────
        elif tool in ("shutdown","turn_off","power_off"):
            speak("Shutting down your computer.")
            time.sleep(2)
            subprocess.run(["shutdown","/s","/t","5"], shell=True)
            return True, "Shutdown initiated"

        elif tool in ("restart","reboot"):
            speak("Restarting.")
            time.sleep(2)
            subprocess.run(["shutdown","/r","/t","5"], shell=True)
            return True, "Restart initiated"

        # ── VOLUME ──────────────────────────────────────────
        elif tool == "volume_set":
            level = int(args.get("level", 50))
            script = (
                f"import ctypes\n"
                f"from ctypes import cast, POINTER\n"
                f"from comtypes import CLSCTX_ALL\n"
                f"from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume\n"
                f"devices = AudioUtilities.GetSpeakers()\n"
                f"interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)\n"
                f"volume = cast(interface, POINTER(IAudioEndpointVolume))\n"
                f"volume.SetMasterVolumeLevelScalar({level}/100, None)\n"
                f"print('Volume set to {level}')"
            )
            return run_script(script)

        # ── NOTIFY ──────────────────────────────────────────
        elif tool == "notify":
            msg = args.get("message","")
            try:
                subprocess.Popen(["powershell","-Command",
                    f'New-BurntToastNotification -Text "Jarvis","{msg}"'])
            except Exception: pass
            return True, "Notified"

        # ── FILE TOOLS ──────────────────────────────────────
        elif tool == "file_list":
            path    = os.path.expandvars(os.path.expanduser(
                        args.get("path", "~\\Desktop")))
            pattern = args.get("pattern","*")
            files   = glob.glob(os.path.join(path, pattern))
            names   = [os.path.basename(f) for f in files[:30]]
            return True, "\n".join(names) if names else "No files"

        elif tool == "file_read":
            path = os.path.expandvars(os.path.expanduser(args.get("path","")))
            if not os.path.exists(path): return False, f"Not found: {path}"
            with open(path, encoding="utf-8", errors="ignore") as f:
                return True, f.read(4000)

        elif tool == "file_create":
            path = os.path.expandvars(os.path.expanduser(args.get("path","")))
            os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
            with open(path, "w", encoding="utf-8") as f:
                f.write(args.get("content",""))
            return True, f"Created {path}"

        elif tool == "file_copy":
            src  = os.path.expandvars(os.path.expanduser(args.get("src","")))
            dest = os.path.expandvars(os.path.expanduser(args.get("dest","")))
            shutil.copy2(src, dest)
            return True, f"Copied to {dest}"

        elif tool == "file_move":
            src  = os.path.expandvars(os.path.expanduser(args.get("src","")))
            dest = os.path.expandvars(os.path.expanduser(args.get("dest","")))
            shutil.move(src, dest)
            return True, f"Moved to {dest}"

        elif tool == "file_delete":
            path = os.path.expandvars(os.path.expanduser(args.get("path","")))
            if not os.path.exists(path): return False, "Not found"
            if os.path.isdir(path): shutil.rmtree(path)
            else: os.remove(path)
            return True, f"Deleted {path}"

        elif tool == "file_search":
            root    = os.path.expandvars(os.path.expanduser(args.get("root","~")))
            pattern = args.get("pattern","*")
            results = []
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    if re.search(pattern.replace("*",".*"), fn, re.IGNORECASE):
                        results.append(os.path.join(dp, fn))
                    if len(results) >= 20: break
                if len(results) >= 20: break
            return True, "\n".join(results) if results else "No matches"

        # ── MEMORY ──────────────────────────────────────────
        elif tool == "remember":
            return True, remember(args.get("key",""), args.get("value",""))

        elif tool == "recall":
            v = recall(args.get("key",""))
            return True, v if v else f"Nothing stored for '{args.get('key','')}'"

        # ── SCRATCHPAD ──────────────────────────────────────
        elif tool == "scratch_write":
            scratch_write(args.get("text",""))
            return True, "Written to scratchpad"

        elif tool == "scratch_read":
            return True, scratch_read() or "Scratchpad is empty"

        elif tool == "scratch_clear":
            scratch_clear()
            return True, "Scratchpad cleared"

        # ── INTERACTION ─────────────────────────────────────
        elif tool == "clarify":
            speak(args.get("question",""))
            set_orb_state("listening")
            try:
                audio = listen(timeout=15)
                p = os.path.join(BASE_DIR, "clarify.wav")
                with open(p,"wb") as f: f.write(audio.get_wav_data())
                return True, wav_to_text(p)
            except sr.WaitTimeoutError:
                return False, "No answer"

        elif tool == "confirm":
            speak(f"{args.get('summary','')}. Say yes to confirm or no to cancel.")
            set_orb_state("listening")
            try:
                audio = listen(timeout=10)
                p = os.path.join(BASE_DIR, "confirm.wav")
                with open(p,"wb") as f: f.write(audio.get_wav_data())
                ans = wav_to_text(p).lower()
                ok  = any(w in ans for w in ["yes","yeah","yep","confirm","sure","do it"])
                return True, "confirmed" if ok else "cancelled"
            except sr.WaitTimeoutError:
                return False, "No response"

        elif tool == "reason":
            thought = ask(
                f"Think through: {args.get('problem','')} Give a short action plan.",
                intent="reasoning"
            )
            return True, thought or "No plan"

        elif tool == "wait":
            time.sleep(min(float(args.get("seconds",1)), 30))
            return True, "done"

        elif tool == "run_script":
            return run_script(args.get("code","print('no code')"))

        # ── SELF-AWARENESS ──────────────────────────────────
        elif tool == "self_read":
            try:
                with open(os.path.abspath(__file__), encoding="utf-8") as f:
                    return True, f.read(6000)
            except Exception as e:
                return False, str(e)

        elif tool == "self_edit":
            # Jarvis can propose edits to its own source
            section = args.get("section","")
            new_code = args.get("new_code","")
            try:
                with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:
                    src = f.read()
                if section not in src:
                    return False, "Section not found in source"
                new_src = src.replace(section, new_code, 1)
                backup = os.path.abspath(__file__) + ".bak"
                shutil.copy2(os.path.abspath(__file__), backup)
                with open(os.path.abspath(__file__), "w", encoding="utf-8") as f:
                    f.write(new_src)
                return True, "Self-edited. Restart to apply."
            except Exception as e:
                return False, str(e)

        else:
            return False, f"Unknown tool: {tool}"

    except Exception as e:
        return False, f"{tool} error: {e}"


# ─────────────────────────────────────────────────────────────
#  RESPONSE PARSER
# ─────────────────────────────────────────────────────────────
def _unwrap_double_encoded(parsed: dict) -> str | None:
    if parsed.get("tool") == "speak":
        text = parsed.get("args",{}).get("text","").strip()
        if text.startswith("{") or text.startswith("["):
            return text
    return None

def parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    # Strip <think>...</think> blocks (deepseek-r1 outputs these)
    raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()

    # Arrow-separated plans
    if "}>" in raw or ("}\n{" in raw):
        parts = re.split(r'\}>+|\}\s*\n\s*\{', raw)
        calls = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part.startswith("{"): part = "{" + part
            if not part.endswith("}"): part = part + "}"
            try:
                obj = json.loads(part)
                if "tool" in obj:
                    calls.append({"tool":obj["tool"],"args":obj.get("args",{})})
            except Exception: pass
        if calls: return calls

    m = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', raw)
    if not m:
        text = clean_speech(raw)
        return [{"tool":"speak","args":{"text":text}}] if text else []

    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError:
        cleaned = re.sub(r'\}[^}]*$', '}', m.group())
        try:
            parsed = json.loads(cleaned)
        except Exception:
            text = clean_speech(raw)
            return [{"tool":"speak","args":{"text":text}}] if text else []

    if isinstance(parsed, dict):
        inner = _unwrap_double_encoded(parsed)
        if inner: return parse_response(inner)

    if isinstance(parsed, list):
        return [{"tool":c["tool"],"args":c.get("args",{})} for c in parsed if "tool" in c]

    if isinstance(parsed, dict):
        if "plan" in parsed:
            steps = parsed["plan"]
            if isinstance(steps, list):
                return [{"tool":c["tool"],"args":c.get("args",{})} for c in steps if "tool" in c]
        if "tool" in parsed:
            return [{"tool":parsed["tool"],"args":parsed.get("args",{})}]

    text = clean_speech(raw)
    return [{"tool":"speak","args":{"text":text}}] if text else []

# ─────────────────────────────────────────────────────────────
#  TOOL PLAN RUNNER
# ─────────────────────────────────────────────────────────────
MAX_DEPTH = 3
PARALLEL_TOOLS = {"file_read","file_list","file_search","recall","get_clipboard"}

def _run_one(call: dict, original: str, depth: int,
             ctx: list, lock: threading.Lock):
    tool    = call.get("tool","")
    args    = call.get("args",{})
    thought = call.get("thought","")
    if thought: log(f"  [Thought] {thought}")
    log(f"  [Tool] {tool}({json.dumps(args)[:80]})")
    set_orb_state("thinking")

    ok, result = execute_tool(tool, args)
    log(f"  [{'OK' if ok else 'FAIL'}] {str(result)[:100]}")

    if ok and tool in CONTEXT_TOOLS and result not in ("spoken","done","","Cleared"):
        with lock:
            ctx.append(f"{tool}: {result[:600]}")

    if not ok and depth < MAX_DEPTH:
        speak("That failed. Trying another approach.")
        fix = ask(
            f"Tool '{tool}' failed: {result}\nOriginal: {original}\n"
            f"Return corrected JSON tool call.",
            intent=route_intent(original)
        )
        if fix:
            run_plan(parse_response(fix), original, depth+1)

    if ok and tool in ("clarify","confirm") and result not in ("confirmed","cancelled"):
        follow = ask(
            f"User answered: {result}\nOriginal: {original}\nContinue with JSON tool call.",
            intent=route_intent(original)
        )
        if follow:
            run_plan(parse_response(follow), original, depth)

    if ok and tool == "screen_capture":
        # Push screen context into conversation histories
        for h in _ollama_histories.values():
            h.append({"role":"user",
                       "content":f"SCREEN_CONTEXT (silent): {result}"})
            h.append({"role":"assistant","content":"{\"tool\":\"speak\",\"args\":{\"text\":\"Got it.\"}}"})


def run_plan(calls: list, original: str, depth: int = 0):
    if not calls or depth > MAX_DEPTH: return

    parallel   = [c for c in calls if c.get("tool","") in PARALLEL_TOOLS]
    sequential = [c for c in calls if c.get("tool","") not in PARALLEL_TOOLS]

    ctx  = []
    lock = threading.Lock()

    if parallel:
        threads = [threading.Thread(target=_run_one,
                    args=(c, original, depth, ctx, lock), daemon=True)
                   for c in parallel]
        for t in threads: t.start()
        for t in threads: t.join()

    for call in sequential:
        _run_one(call, original, depth, ctx, lock)

    if ctx and depth < MAX_DEPTH:
        follow = ask(
            "Tool results:\n" + "\n".join(ctx) +
            f"\n\nOriginal request: {original}\n"
            "Based on these results, what should happen next? "
            "If complete, use speak to tell the user. "
            "Otherwise output next JSON tool call.",
            intent=route_intent(original)
        )
        if follow:
            next_calls = parse_response(follow)
            if next_calls:
                run_plan(next_calls, original, depth+1)

# ─────────────────────────────────────────────────────────────
#  CORE CONVERSATION LOOP
# ─────────────────────────────────────────────────────────────
def manual_wake():
    global listening_for_wake
    if not listening_for_wake: return
    listening_for_wake = False
    set_orb_state("active")
    threading.Thread(target=process_prompt, daemon=True).start()

def process_prompt():
    global listening_for_wake
    log("process_prompt started")

    # Test TTS first — if this fails we know edge-tts is the issue
    try:
        speak(f"{config['wake_word'].capitalize()} online.")
        log("TTS OK")
    except Exception as e:
        log(f"TTS failed on startup: {e}", "ERR")

    while True:
        try:
            set_orb_state("listening")
            log("Waiting for speech input...")

            try:
                prompt = capture_speech()
            except sr.WaitTimeoutError:
                log("Listen timeout — going back to sleep")
                listening_for_wake = True
                set_orb_state("idle")
                break
            except OSError as e:
                log(f"Microphone error: {e}", "ERR")
                speak("Microphone not available.")
                listening_for_wake = True
                set_orb_state("idle")
                break
            except Exception as e:
                log(f"Listen error: {e}", "ERR")
                speak("Could not hear you.")
                listening_for_wake = True
                set_orb_state("idle")
                break
            stop_speaking()

            if not prompt or not prompt.strip():
                log("Empty prompt, retrying")
                speak("Didn't catch that.")
                continue

            lower = prompt.lower()
            log(f"You: {prompt}")

            if any(w in lower for w in ["stop","sleep","go to sleep","goodbye","shut down jarvis"]):
                speak("Going to sleep.")
                listening_for_wake = True
                set_orb_state("idle")
                break

            intent = route_intent(lower)
            log(f"  [Route] {intent}")

            # Real-time info that needs web search
            realtime_triggers = [
                "weather","news","price","score","latest","current",
                "today","right now","stock","rate","temperature outside"
            ]
            needs_web = any(t in lower for t in realtime_triggers)

            set_orb_state("thinking")

            # Screen context for vision/action tasks
            if intent == "vision" or any(t in lower for t in ["click","find the","the button"]):
                desc = capture_and_describe(prompt)
                prompt_with_ctx = (
                    f"Screen context: {desc}\n\nUser request: {prompt}\n"
                    f"Use the screen coordinates to perform precise actions."
                )
                raw = ask(prompt_with_ctx, intent=intent)
            elif needs_web:
                # Get info silently then answer
                results = web_search_silent(prompt)
                if results:
                    prompt_with_ctx = (
                        f"Web search results for '{prompt}':\n{results}\n\n"
                        f"Use this information to answer the user's question accurately. "
                        f"Respond with a speak tool call."
                    )
                    raw = ask(prompt_with_ctx, intent="fast")
                else:
                    raw = ask(prompt, intent=intent)
            else:
                mem = memories_summary()
                ctx_prompt = (f"User memories:\n{mem}\n\n" if mem else "") + prompt
                raw = ask(ctx_prompt, intent=intent)

            if raw is None:
                set_orb_state("idle"); continue

            log(f"AI: {raw[:200]}")
            calls = parse_response(raw)
            run_plan(calls, prompt)
            set_orb_state("listening")

        except sr.WaitTimeoutError:
            speak("Timed out.")
            listening_for_wake = True
            set_orb_state("idle"); break
        except Exception as e:
            import traceback
            log(f"Loop error: {e}", "ERR")
            log(traceback.format_exc(), "ERR")
            try:
                speak("Something went wrong.")
            except Exception:
                pass
            listening_for_wake = True
            set_orb_state("idle"); break

def wake_word_loop():
    global listening_for_wake
    set_orb_state("idle")
    while assistant_active:
        try:
            if listening_for_wake:
                try:
                    audio = listen(timeout=8, phrase_limit=4)
                except sr.WaitTimeoutError:
                    continue  # no sound, loop again
                wake_path = os.path.join(BASE_DIR, "wake.wav")
                with open(wake_path, "wb") as f:
                    f.write(audio.get_wav_data())
                heard = wav_to_text(wake_path)
                log(f"Heard: '{heard}'")
                if config["wake_word"].lower() in heard.lower():
                    log("Wake word detected!")
                    listening_for_wake = False
                    set_orb_state("active")
                    process_prompt()
        except Exception as e:
            log(f"Wake loop: {e}", "WARN")
        time.sleep(0.2)


# ─────────────────────────────────────────────────────────────
#  CIRCULAR ORB UI
# ─────────────────────────────────────────────────────────────
_orb_state    = "idle"   # idle | active | listening | thinking | speaking
_orb_win      = None
_orb_canvas   = None
_orb_angle    = 0.0
_orb_anim_id  = None
_menu_open    = False

ORB_COLORS = {
    "idle":      {"core":"#0a0f1a","ring":"#1a3050","pulse":"#1a3050"},
    "active":    {"core":"#0d1a2e","ring":"#0ef5c4","pulse":"#0ef5c4"},
    "listening": {"core":"#0a1520","ring":"#0ab8e8","pulse":"#0ab8e8"},
    "thinking":  {"core":"#0f1520","ring":"#e8b84b","pulse":"#e8b84b"},
    "speaking":  {"core":"#0d1520","ring":"#50c878","pulse":"#50c878"},
}

def set_orb_state(state: str):
    global _orb_state
    _orb_state = state
    if _orb_win:
        try:
            _orb_win.after(0, _redraw_orb)
        except Exception:
            pass

def _redraw_orb():
    if not _orb_canvas: return
    try:
        c = _orb_canvas
        w = h = 120
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill="black", outline="")
        colors = ORB_COLORS.get(_orb_state, ORB_COLORS["idle"])

        # Outer glow rings (animated when active)
        if _orb_state != "idle":
            for i in range(3):
                offset  = i * 8 + 4
                alpha_h = max(10, 60 - i*20)
                alpha_s = format(alpha_h, '02x')
                ring_c  = colors["ring"] + alpha_s if len(colors["ring"])==7 else colors["ring"]
                c.create_oval(
                    offset, offset,
                    w-offset, h-offset,
                    outline=colors["ring"],
                    width=1 if i>0 else 2
                )

        # Sound-wave arcs when speaking
        if _orb_state == "speaking":
            for i in range(4):
                angle_off = math.sin(_orb_angle + i * 0.8) * 15
                r = 35 + i * 6
                cx, cy = w//2, h//2
                c.create_arc(
                    cx-r, cy-r, cx+r, cy+r,
                    start=45+angle_off, extent=90,
                    outline=colors["ring"], width=2, style="arc"
                )

        # Thinking spinner
        if _orb_state == "thinking":
            cx, cy, r = w//2, h//2, 44
            for i in range(6):
                a = math.radians(_orb_angle * 3 + i * 60)
                x1 = cx + r * math.cos(a)
                y1 = cy + r * math.sin(a)
                x2 = cx + (r-8) * math.cos(a)
                y2 = cy + (r-8) * math.sin(a)
                alpha = int(255 * (i+1)/6)
                col = colors["ring"]
                c.create_line(x1,y1,x2,y2, fill=col, width=2)

        # Core circle
        pad = 18
        c.create_oval(pad,pad,w-pad,h-pad,
                       fill=colors["core"], outline=colors["ring"], width=2)

        # Wake word text
        name = config["wake_word"].upper()
        c.create_text(w//2, h//2, text=name,
                       fill=colors["ring"],
                       font=("Consolas", 9, "bold"))

        # Status dot
        dot_colors = {"idle":"#1a3050","active":"#0ef5c4",
                       "listening":"#0ab8e8","thinking":"#e8b84b","speaking":"#50c878"}
        dc = dot_colors.get(_orb_state,"#1a3050")
        c.create_oval(w//2-3, h-14, w//2+3, h-8, fill=dc, outline="")

    except Exception as e:
        pass

def _animate_orb():
    global _orb_angle, _orb_anim_id
    if not _orb_win: return
    _orb_angle += 0.08
    _redraw_orb()
    _orb_anim_id = _orb_win.after(50, _animate_orb)

def _open_orb_menu(event=None):
    global _menu_open
    if _menu_open: return
    _menu_open = True

    BG = "#080d14"; FG = "#c8ddf0"; ACC = "#0ef5c4"; BORDER = "#1a3050"
    F  = ("Consolas", 9, "bold")

    menu = tk.Toplevel(_orb_win)
    menu.overrideredirect(True)
    menu.attributes("-topmost", True)
    menu.attributes("-alpha", 0.97)
    menu.configure(bg=BG)

    # Position menu near orb
    ox = _orb_win.winfo_x()
    oy = _orb_win.winfo_y()
    menu.geometry(f"180x260+{ox-190}+{oy-70}")

    outer = tk.Frame(menu, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="both", expand=True)
    inner = tk.Frame(outer, bg=BG)
    inner.pack(fill="both", expand=True)

    tk.Label(inner, text=config["wake_word"].upper(),
             bg=BG, fg=ACC, font=("Consolas",11,"bold")).pack(pady=(10,2))
    tk.Label(inner, text=_orb_state.upper(),
             bg=BG, fg="#4a6888", font=("Consolas",8)).pack(pady=(0,8))
    tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=12)

    def close_menu():
        global _menu_open
        _menu_open = False
        menu.destroy()

    def btn(label, cmd, danger=False):
        b = tk.Button(inner, text=label, command=lambda: [cmd(), close_menu()],
                      bg=BG, fg=ACC if not danger else "#f04060",
                      font=F, relief="flat", padx=10, pady=5,
                      cursor="hand2", activebackground="#0d1520",
                      activeforeground=ACC, anchor="w")
        b.pack(fill="x", padx=10, pady=2)

    btn("▶  Wake",
        lambda: threading.Thread(target=manual_wake, daemon=True).start())
    btn("📷  Capture Screen",
        lambda: threading.Thread(target=lambda:
            speak(capture_and_describe()), daemon=True).start())
    btn("🎤  Calibrate Mic",
        lambda: threading.Thread(target=calibrate_mic, daemon=True).start())
    btn("🧠  Memories",
        lambda: threading.Thread(target=open_memories_window, daemon=True).start())
    btn("⚙   Settings",
        lambda: threading.Thread(target=open_settings_window, daemon=True).start())

    tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=12, pady=4)
    btn("✕  Quit", lambda: os._exit(0), danger=True)

    menu.bind("<FocusOut>", lambda e: close_menu())
    menu.focus_set()

def open_orb():
    global _orb_win, _orb_canvas

    win = tk.Tk()
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.attributes("-alpha", 0.95)
    win.configure(bg="black")
    # Use Windows transparentcolor to knock out the black background
    # giving a true circular orb appearance
    try:
        win.attributes("-transparentcolor", "black")
    except Exception:
        pass  # fallback — some Windows versions don't support this
    _orb_win = win

    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    W = H = 120
    win.geometry(f"{W}x{H}+{sw-W-20}+{sh-H-60}")

    canvas = tk.Canvas(win, width=W, height=H,
                        bg="black", highlightthickness=0)
    canvas.pack()
    _orb_canvas = canvas

    # Drag
    _d = {"x": 0, "y": 0}
    def ds(e): _d["x"] = e.x; _d["y"] = e.y
    def dm(e): win.geometry(f"+{win.winfo_x()+e.x-_d['x']}+{win.winfo_y()+e.y-_d['y']}")
    canvas.bind("<ButtonPress-1>", ds)
    canvas.bind("<B1-Motion>", dm)
    canvas.bind("<ButtonRelease-1>",
                lambda e: _open_orb_menu() if abs(e.x-_d["x"])<5 and abs(e.y-_d["y"])<5 else None)

    _animate_orb()
    win.mainloop()

# ─────────────────────────────────────────────────────────────
#  MEMORIES WINDOW
# ─────────────────────────────────────────────────────────────
def open_memories_window():
    BG="#0a0f1a"; FG="#c8ddf0"; ACC="#0ef5c4"; B="#1a3050"; EB="#141f30"
    F=("Consolas",9); FB=("Consolas",9,"bold")

    win = tk.Tk()
    win.title("Memories"); win.geometry("500x440")
    win.configure(bg=BG); win.attributes("-topmost",True); win.resizable(True,True)

    tk.Frame(win,bg=ACC,height=2).pack(fill="x")
    tk.Label(win,text="LONG-TERM MEMORY",font=("Consolas",12,"bold"),
             bg=BG,fg=ACC).pack(pady=(12,2))
    tk.Label(win,text="Persists across reboots. AI uses these automatically.",
             bg=BG,fg="#4a6888",font=F).pack()
    tk.Frame(win,bg=B,height=1).pack(fill="x",padx=16,pady=6)

    fr = tk.Frame(win,bg=EB); fr.pack(fill="both",expand=True,padx=14,pady=6)
    cv = tk.Canvas(fr,bg=EB,highlightthickness=0)
    sb = tk.Scrollbar(fr,orient="vertical",command=cv.yview)
    sf = tk.Frame(cv,bg=EB)
    sf.bind("<Configure>",lambda e:cv.configure(scrollregion=cv.bbox("all")))
    cv.create_window((0,0),window=sf,anchor="nw")
    cv.configure(yscrollcommand=sb.set)
    cv.pack(side="left",fill="both",expand=True); sb.pack(side="right",fill="y")

    def refresh():
        for w in sf.winfo_children(): w.destroy()
        mem = load_memory()
        if not mem:
            tk.Label(sf,text="No memories yet.",bg=EB,fg="#4a6888",font=F).pack(pady=20)
            return
        for key,data in mem.items():
            row=tk.Frame(sf,bg="#0d1825",pady=4); row.pack(fill="x",padx=4,pady=2)
            tk.Label(row,text=key,bg="#0d1825",fg=ACC,font=FB,width=16,anchor="w").pack(side="left",padx=8)
            tk.Label(row,text=data["value"],bg="#0d1825",fg=FG,font=F,anchor="w").pack(side="left",fill="x",expand=True)
            def dk(k=key):
                m=load_memory(); del m[k]; save_memory(m); refresh()
            tk.Button(row,text="×",command=dk,bg="#0d1825",fg="#f04060",
                       font=F,relief="flat",cursor="hand2").pack(side="right",padx=6)
    refresh()

    af=tk.Frame(win,bg=BG); af.pack(fill="x",padx=14,pady=(0,10))
    kv=tk.StringVar(); vv=tk.StringVar()
    tk.Label(af,text="Key:",bg=BG,fg="#4a6888",font=F).pack(side="left")
    tk.Entry(af,textvariable=kv,bg=EB,fg=FG,insertbackground=ACC,font=F,relief="flat",bd=3,width=14).pack(side="left",padx=4,ipady=3)
    tk.Label(af,text="Value:",bg=BG,fg="#4a6888",font=F).pack(side="left")
    tk.Entry(af,textvariable=vv,bg=EB,fg=FG,insertbackground=ACC,font=F,relief="flat",bd=3,width=20).pack(side="left",padx=4,ipady=3)
    def add():
        if kv.get().strip() and vv.get().strip():
            remember(kv.get().strip(),vv.get().strip()); kv.set(""); vv.set(""); refresh()
    tk.Button(af,text="ADD",command=add,bg=ACC,fg=BG,font=FB,relief="flat",padx=8,cursor="hand2").pack(side="left",padx=4)
    win.mainloop()

# ─────────────────────────────────────────────────────────────
#  SETTINGS WINDOW
# ─────────────────────────────────────────────────────────────
def open_settings_window():
    global _settings_open
    if _settings_open: return
    _settings_open = True

    BG="#0a0f1a"; FG="#c8ddf0"; ACC="#0ef5c4"; B="#1a3050"; EB="#141f30"
    F=("Consolas",9); FB=("Consolas",9,"bold")

    win = tk.Tk()
    win.title("Settings"); win.geometry("520x700")
    win.resizable(False,False); win.configure(bg=BG)
    win.attributes("-topmost",True)

    def on_close():
        global _settings_open; _settings_open=False; win.destroy()
    win.protocol("WM_DELETE_WINDOW",on_close)

    tk.Frame(win,bg=ACC,height=2).pack(fill="x")
    tk.Label(win,text="SETTINGS",font=("Consolas",13,"bold"),bg=BG,fg=ACC).pack(pady=(14,4))
    tk.Frame(win,bg=B,height=1).pack(fill="x",padx=20)

    # Scrollable content
    cv=tk.Canvas(win,bg=BG,highlightthickness=0)
    sb_s=tk.Scrollbar(win,orient="vertical",command=cv.yview)
    cv.configure(yscrollcommand=sb_s.set)
    cv.pack(side="left",fill="both",expand=True,pady=6)
    sb_s.pack(side="right",fill="y")
    fr=tk.Frame(cv,bg=BG); fid=cv.create_window((0,0),window=fr,anchor="nw")
    fr.bind("<Configure>",lambda e:cv.configure(scrollregion=cv.bbox("all")))
    cv.bind("<Configure>",lambda e:cv.itemconfig(fid,width=e.width))
    win.bind_all("<MouseWheel>",lambda e:cv.yview_scroll(-1*(e.delta//120),"units"))

    def sec(t):
        tk.Label(fr,text=f"── {t}",font=FB,bg=BG,fg=ACC,anchor="w").pack(fill="x",padx=20,pady=(14,2))
    def lbl(t,note=None):
        tk.Label(fr,text=t,font=FB,bg=BG,fg=FG,anchor="w").pack(fill="x",padx=20,pady=(6,1))
        if note: tk.Label(fr,text=note,font=F,bg=BG,fg="#4a6888",anchor="w").pack(fill="x",padx=20)
    def ent(var,show=None):
        kw={"show":show} if show else {}
        e=tk.Entry(fr,textvariable=var,bg=EB,fg=FG,insertbackground=ACC,
                    font=F,relief="flat",bd=4,**kw)
        e.pack(fill="x",ipady=5,padx=20); return e

    sec("ASSISTANT")
    lbl("Wake Word")
    wv=tk.StringVar(value=config["wake_word"]); ent(wv)
    lbl("TTS Voice","en-US-ChristopherNeural = deep American")
    tv=tk.StringVar(value=config["tts_voice"]); ent(tv)

    sec("MODELS")
    lbl("Fast model","Simple Q&A — qwen2.5:1.5b")
    mf=tk.StringVar(value=config["model_fast"]); ent(mf)
    lbl("General model","Actions — qwen2.5:3b")
    mg=tk.StringVar(value=config["model_general"]); ent(mg)
    lbl("Reasoning model","Code/debug — deepseek-r1:1.5b")
    mr=tk.StringVar(value=config["model_reasoning"]); ent(mr)
    lbl("Coder model","File ops — qwen2.5-coder:3b")
    mc=tk.StringVar(value=config["model_coder"]); ent(mc)

    sec("GEMINI (optional — for screen vision)")
    lbl("API Key","aistudio.google.com → free key")
    gk=tk.StringVar(value=config["gemini_api_key"]); ent(gk,show="*")

    sec("AUDIO")
    lbl("Volume")
    vrow=tk.Frame(fr,bg=BG); vrow.pack(fill="x",padx=20)
    vov=tk.IntVar(value=config["volume"])
    vn=tk.Label(vrow,text=str(config["volume"]),bg=BG,fg=ACC,font=FB,width=4)
    def ov(v): vn.config(text=str(int(float(v))))
    tk.Scale(vrow,from_=0,to=100,orient="horizontal",variable=vov,command=ov,
             bg=BG,fg=FG,troughcolor=EB,activebackground=ACC,
             highlightthickness=0,sliderlength=12,length=380).pack(side="left",fill="x",expand=True)
    vn.pack(side="left",padx=6)
    lbl("Whisper Speed")
    sz=tk.StringVar(value=config["whisper_size"])
    szr=tk.Frame(fr,bg=BG); szr.pack(fill="x",padx=20)
    for v,l in [("tiny","Tiny"),("base","Base"),("small","Small"),("medium","Medium")]:
        tk.Radiobutton(szr,text=l,variable=sz,value=v,bg=BG,fg=FG,
                        selectcolor=EB,activebackground=BG,font=F).pack(side="left",padx=6)

    brow=tk.Frame(win,bg="#0d1520"); brow.pack(fill="x",side="bottom")
    tk.Frame(brow,bg=B,height=1).pack(fill="x")
    bi=tk.Frame(brow,bg="#0d1520"); bi.pack(pady=12)

    def save():
        global _settings_open
        wake=wv.get().strip().lower()
        if not re.match(r"^[a-z]+$",wake):
            messagebox.showerror("Invalid","Wake word must be letters only."); return
        config.update({"wake_word":wake,"tts_voice":tv.get().strip(),
                        "model_fast":mf.get().strip(),"model_general":mg.get().strip(),
                        "model_reasoning":mr.get().strip(),"model_coder":mc.get().strip(),
                        "gemini_api_key":gk.get().strip(),"volume":vov.get(),
                        "whisper_size":sz.get()})
        save_config(config)
        threading.Thread(target=load_whisper,daemon=True).start()
        if config["gemini_api_key"]: init_gemini()
        messagebox.showinfo("Saved","Settings saved.")
        _settings_open=False; win.destroy()

    def test():
        threading.Thread(target=speak,
            args=(f"Online. {wv.get().capitalize()} ready.",),daemon=True).start()

    for lbl_t,cmd,col in [("SAVE",save,ACC),("TEST VOICE",test,"#0ab8e8")]:
        tk.Button(bi,text=lbl_t,command=cmd,bg=col,fg=BG,font=FB,
                   relief="flat",padx=18,pady=7,cursor="hand2").pack(side="left",padx=8)

    win.mainloop()

# ─────────────────────────────────────────────────────────────
#  TRAY ICON
# ─────────────────────────────────────────────────────────────
def _tray_img():
    img=Image.new("RGBA",(64,64),(0,0,0,0))
    d=ImageDraw.Draw(img)
    d.ellipse([4,4,60,60],fill=(14,245,196))
    d.text((16,18),"AI",fill=(5,20,20))
    return img

def setup_tray():
    menu=pystray.Menu(
        pystray.MenuItem("Settings",lambda i,it:threading.Thread(
            target=open_settings_window,daemon=True).start()),
        pystray.MenuItem("Memories",lambda i,it:threading.Thread(
            target=open_memories_window,daemon=True).start()),
        pystray.MenuItem("Quit",lambda i,it:os._exit(0)),
    )
    icon=pystray.Icon("Jarvis",_tray_img(),
                       f"{config['wake_word'].capitalize()} AI",menu)
    icon.run()

# ─────────────────────────────────────────────────────────────
#  FIRST-RUN SETUP
# ─────────────────────────────────────────────────────────────
def first_run_setup():
    BG="#0a0f1a"; FG="#c8ddf0"; ACC="#0ef5c4"; B="#1a3050"; EB="#141f30"
    F=("Consolas",10); FB=("Consolas",10,"bold")

    win=tk.Tk(); win.title("Setup")
    win.resizable(True,True); win.configure(bg=BG); win.attributes("-topmost",True)
    W,H=560,560; sw=win.winfo_screenwidth(); sh=win.winfo_screenheight()
    win.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    done={"v":False}
    def on_close():
        if not done["v"]: win.destroy()
    win.protocol("WM_DELETE_WINDOW",on_close)

    tk.Frame(win,bg=ACC,height=3).pack(fill="x")
    tk.Label(win,text="JARVIS SETUP",font=("Consolas",14,"bold"),bg=BG,fg=ACC).pack(pady=(18,2))
    tk.Label(win,text="Optimised for i5-1235U  /  8GB RAM  /  MX550",
             bg=BG,fg="#4a6888",font=F).pack(pady=(0,8))
    tk.Frame(win,bg=B,height=1).pack(fill="x",padx=24)

    cv=tk.Canvas(win,bg=BG,highlightthickness=0)
    sb=tk.Scrollbar(win,orient="vertical",command=cv.yview)
    cv.configure(yscrollcommand=sb.set)
    cv.pack(side="left",fill="both",expand=True,padx=(20,0),pady=8)
    sb.pack(side="right",fill="y",pady=8)
    fr=tk.Frame(cv,bg=BG); fid=cv.create_window((0,0),window=fr,anchor="nw")
    fr.bind("<Configure>",lambda e:cv.configure(scrollregion=cv.bbox("all")))
    cv.bind("<Configure>",lambda e:cv.itemconfig(fid,width=e.width))
    win.bind_all("<MouseWheel>",lambda e:cv.yview_scroll(-1*(e.delta//120),"units"))

    def lbl(t,note=None):
        tk.Label(fr,text=t,font=FB,bg=BG,fg=FG,anchor="w").pack(fill="x",pady=(10,1),padx=4)
        if note: tk.Label(fr,text=note,font=F,bg=BG,fg="#4a6888",anchor="w").pack(fill="x",padx=4)
    def mke(var,show=None):
        kw={"show":show} if show else {}
        e=tk.Entry(fr,textvariable=var,bg=EB,fg=FG,insertbackground=ACC,
                    font=F,relief="flat",bd=4,**kw)
        e.pack(fill="x",ipady=6,padx=4,pady=(0,2)); return e

    tk.Label(fr,text="Pull these models in your terminal first:",
             font=FB,bg=BG,fg=ACC,anchor="w").pack(fill="x",pady=(8,4),padx=4)
    for cmd in ["ollama pull qwen2.5:1.5b","ollama pull qwen2.5:3b",
                 "ollama pull deepseek-r1:1.5b","ollama pull qwen2.5-coder:3b"]:
        tk.Label(fr,text=f"  {cmd}",font=F,bg=EB,fg=ACC,anchor="w"
                  ).pack(fill="x",padx=4,pady=1)

    lbl("Wake Word","Single English word  e.g. jarvis, nova, aria")
    wv=tk.StringVar(value="jarvis"); mke(wv)

    lbl("Gemini API Key (optional)","aistudio.google.com → free key (needed for screen vision)")
    gv=tk.StringVar(); mke(gv,show="*")

    lbl("Fast Model","For simple Q&A — qwen2.5:1.5b")
    mf=tk.StringVar(value="qwen2.5:1.5b"); mke(mf)
    lbl("General Model","For actions — qwen2.5:3b")
    mg=tk.StringVar(value="qwen2.5:3b"); mke(mg)
    lbl("Reasoning Model","For code/analysis — deepseek-r1:1.5b")
    mr=tk.StringVar(value="deepseek-r1:1.5b"); mke(mr)

    tk.Frame(fr,bg=B,height=1).pack(fill="x",padx=4,pady=14)

    def save():
        wake=wv.get().strip().lower()
        if not re.match(r"^[a-z]+$",wake):
            messagebox.showerror("Invalid","Wake word must be letters only."); return
        config.update({"wake_word":wake,"gemini_api_key":gv.get().strip(),
                        "model_fast":mf.get().strip(),"model_general":mg.get().strip(),
                        "model_reasoning":mr.get().strip()})
        save_config(config)
        done["v"]=True; win.destroy()

    tk.Button(fr,text="SAVE AND START",command=save,
               bg=ACC,fg=BG,font=("Consolas",11,"bold"),
               relief="flat",padx=24,pady=10,cursor="hand2").pack(pady=(4,20))
    win.mainloop()
    return done["v"]

# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────
def startup():
    log("=== Jarvis Starting ===")

    # First run
    if not os.path.exists(CONFIG_FILE):
        if not first_run_setup():
            log("Setup cancelled."); return

    log("Loading Whisper...")
    load_whisper()

    if config["gemini_api_key"]:
        init_gemini()

    if not ollama_available():
        speak("Ollama is not running. Please start it and restart Jarvis.")
        log("Ollama not available", "ERR")

    # Warm up models in background
    models_to_warm = [config["model_fast"], config["model_general"]]
    for m in models_to_warm:
        threading.Thread(target=_warmup_model, args=(m,), daemon=True).start()

    # Warm TTS
    def _warm_tts():
        try:
            asyncio.run(
                edge_tts.Communicate(".", voice=config["tts_voice"]).save(TTS_FILE)
            )
        except Exception: pass
    threading.Thread(target=_warm_tts, daemon=True).start()

    # Initial mic calibration
    threading.Thread(target=calibrate_mic, daemon=True).start()

    log(f"Ready. Say '{config['wake_word']}' or click the orb.")

    # Tray
    threading.Thread(target=setup_tray, daemon=True).start()

    # Wake word loop
    threading.Thread(target=wake_word_loop, daemon=True).start()

    # Orb UI — runs on main thread
    open_orb()


if __name__ == "__main__":
    startup()
