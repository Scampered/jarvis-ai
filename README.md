# Jarvis AI Assistant

A desktop voice assistant: wake-word activation, speech-to-text, an LLM for reasoning, and text-to-speech for replies — with the ability to control your PC (opening apps/URLs, running scripts, screen vision, web search).

This repo holds four working variants, each a different take on the same idea:

| Script | Brain | Highlights |
|---|---|---|
| **[jarvis.py](jarvis.py)** | Local (Ollama) + optional Gemini | The flagship build. Routes prompts across multiple local models (fast Q&A, general, reasoning, coding), falls back to Gemini for text and screen-vision questions. Floating orb GUI + system tray, persistent memory/scratchpad, web search, app/URL launching, screenshot-and-describe, on-screen overlay drawing, and an agentic tool-calling/planning loop. |
| **[jarvisclaude.py](jarvisclaude.py)** | Gemini (cloud) | A lighter cloud-only build with the same voice loop, floating orb GUI, and tray icon, but no local models or memory system. |
| **[jarvis_gemini.py](jarvis_gemini.py)** | Gemini + OpenAI (Whisper) | Minimal command-line version — wake word, speech-to-text, one LLM call, spoken reply. No GUI. |
| **[jarvis_pc.py](jarvis_pc.py)** | Gemini | CLI version with **PyAction**: the model can write and execute Python code on your machine (via `action.py`) to carry out PC actions on request. |

> A newer, more advanced rebuild of this project — with far more capability — lives at [Jarvis-Automater](https://github.com/Scampered/Jarvis-Automater).

## Setup

1. Install dependencies for the variant you want to run, e.g.:
   ```bash
   pip install google-generativeai faster-whisper speechrecognition pyautogui pystray pillow edge-tts pygame pygetwindow
   ```
   `jarvis.py` additionally needs `requests pyperclip PyQt6 duckduckgo-search`, and `jarvis_gemini.py` additionally needs `openai pyaudio`.

2. For `jarvis.py` — install [Ollama](https://ollama.com) and pull the local models it routes to:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull qwen2.5:3b
   ollama pull deepseek-r1:1.5b
   ollama pull qwen2.5-coder:3b
   ```

3. Set API keys as environment variables rather than editing the source:
   - `jarvis.py` / `jarvisclaude.py` / `jarvis_pc.py` — enter your Gemini key in the app's settings UI on first run (saved to `jarvis_config.json`), or set `GOOGLE_API_KEY`.
   - `jarvis_gemini.py` — set `GOOGLE_API_KEY` and `OPENAI_API_KEY`.

   Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com).

4. Run whichever version you want:
   ```bash
   python jarvis.py
   ```

Say the wake word ("jarvis") to start listening.
