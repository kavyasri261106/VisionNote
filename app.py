"""
app.py
VisionNote - Streamlit front end.

Live webcam feed -> YOLOv8 currency detection -> on-screen box + offline
voice announcement. Run with:

    streamlit run app.py

Notes:
- Model weights should live at model/trained_model.pt (see detect.py).
- TTS runs in a background thread so it never blocks/freezes the video loop.
- Announcements are throttled (COOLDOWN_SECONDS) so the app doesn't repeat
  the same note over and over every frame.
"""

import time
import threading
import queue

import cv2
import streamlit as st
import pyttsx3

from detect import CurrencyDetector, draw_detection, get_phrase, total_amount

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "model/trained_model.pt"
COOLDOWN_SECONDS = 3.0   # min seconds between repeat announcements of same note
CAM_INDEX = 0

LANGUAGES = {"English": "en", "Hindi": "hi", "Tamil": "ta"}

# ---------------------------------------------------------------------------
# Text-to-speech worker (runs on its own thread so video stays smooth)
# ---------------------------------------------------------------------------
tts_queue: "queue.Queue[str]" = queue.Queue()


def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    while True:
        phrase = tts_queue.get()
        if phrase is None:
            break
        engine.say(phrase)
        engine.runAndWait()


threading.Thread(target=tts_worker, daemon=True).start()


def announce(phrase: str):
    tts_queue.put(phrase)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    return CurrencyDetector(model_path=MODEL_PATH)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="VisionNote", page_icon="💵", layout="centered")
st.title("💵 VisionNote")
st.caption("AI-powered currency note recognition for visually impaired users")

col1, col2 = st.columns(2)
with col1:
    lang_name = st.selectbox("Voice language", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_name]
with col2:
    multi_mode = st.checkbox("Multiple note mode (sum total)", value=False)

run = st.toggle("Start camera", value=False, key="camera_toggle")
stop_btn = st.button("Stop") if run else None
frame_slot = st.empty()
status_slot = st.empty()

if run:
    detector = load_detector()
    cap = cv2.VideoCapture(CAM_INDEX)

    last_label = None
    last_announced_at = 0.0

    try:
        while run and not stop_btn:
            ok, frame = cap.read()
            if not ok:
                status_slot.error("Could not read from camera.")
                break

            if multi_mode:
                detections = detector.all_detections(frame)
                for d in detections:
                    frame = draw_detection(frame, d, lang)

                if detections:
                    amount = total_amount(detections)
                    status_slot.success(f"Detected {len(detections)} note(s) — Total: ₹{amount}")
                    now = time.time()
                    if now - last_announced_at > COOLDOWN_SECONDS:
                        announce(f"Total amount is {amount} rupees")
                        last_announced_at = now
                else:
                    status_slot.info("No notes detected. Hold a note steady in frame.")

            else:
                result = detector.detect(frame)
                if result:
                    frame = draw_detection(frame, result, lang)
                    status_slot.success(
                        f"Detected: ₹{result.label} — Confidence: {result.confidence*100:.1f}%"
                    )
                    now = time.time()
                    if result.label != last_label or (now - last_announced_at > COOLDOWN_SECONDS):
                        announce(get_phrase(result.label, lang))
                        last_label = result.label
                        last_announced_at = now
                else:
                    status_slot.warning(
                        "Not confident yet — try moving the note closer or improving lighting."
                    )
                    last_label = None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, channels="RGB")

    finally:
        cap.release()
else:
    status_slot.info("Toggle 'Start camera' to begin scanning currency notes.")
