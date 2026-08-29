

import collections
import json
import os
import threading

import av
import cv2
import numpy as np
import pyttsx3
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

MODEL_PATH_TASK = "hand_landmarker.task"
MODEL_PATH_KERAS = "models/best_model.keras"
LABELS_PATH = "data/processed/labels.json"

SEQ_LEN = 40
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
EPS = 1e-6

CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW = 8
PREDICT_EVERY_N_FRAMES = 2
RESET_AFTER_MISMATCH_FRAMES = 6

st.set_page_config(page_title="ISL Recognition & Speech Assistant", layout="centered")


#  Cached, load-once resources 

@st.cache_resource
def load_model_and_labels():
    with open(LABELS_PATH) as f:
        label_maps = json.load(f)
    index_to_label = {int(k): v for k, v in label_maps["index_to_label"].items()}
    model = keras.models.load_model(MODEL_PATH_KERAS)
    return model, index_to_label


# (same logic as live_app.py) 

def corrected_label(raw_label):
    return "Right" if raw_label == "Left" else "Left"


def extract_landmarks(result):
    left = np.zeros(21 * 3)
    right = np.zeros(21 * 3)
    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            label = corrected_label(handedness[0].category_name)
            if label == "Left":
                left = coords
            else:
                right = coords
    return np.concatenate([left, right])


def draw_landmarks_simple(frame, result):
    if not result.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)


def normalize_hand(hand_vec63):
    if not np.any(hand_vec63):
        return hand_vec63
    points = hand_vec63.reshape(21, 3)
    wrist = points[WRIST_IDX]
    translated = points - wrist
    scale_ref = max(np.linalg.norm(translated[MIDDLE_MCP_IDX]), EPS)
    return (translated / scale_ref).flatten()


def normalize_sequence(seq):
    normalized = np.zeros_like(seq)
    for i, frame in enumerate(seq):
        normalized[i, :63] = normalize_hand(frame[:63])
        normalized[i, 63:] = normalize_hand(frame[63:])
    return normalized


class Speaker:
    """Same fresh-engine-per-call pattern as live_app.py -- avoids the
    pyttsx3 'works once then goes silent' issue on Windows."""

    def say(self, text):
        def _speak():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        threading.Thread(target=_speak, daemon=True).start()


# The video processor

class ISLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.index_to_label = load_model_and_labels()
        self.speaker = Speaker()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH_TASK)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.timestamp = 0
        self.frame_counter = 0
        self.frame_buffer = collections.deque(maxlen=SEQ_LEN)
        self.recent_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.sentence = []
        self.last_spoken_label = None
        self.mismatch_streak = 0
        self.current_display_text = "..."
        self.current_confidence = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp += 33
        result = self.detector.detect_for_video(mp_image, self.timestamp)
        draw_landmarks_simple(img, result)

        landmarks = extract_landmarks(result)
        self.frame_buffer.append(landmarks)

        if len(self.frame_buffer) == SEQ_LEN:
            self.frame_counter += 1
            if self.frame_counter % PREDICT_EVERY_N_FRAMES == 0:
                seq = np.array(self.frame_buffer)
                normalized_seq = normalize_sequence(seq)
                input_tensor = normalized_seq[np.newaxis, ...].astype(np.float32)
                probs = self.model(input_tensor, training=False).numpy()[0]
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])

                if top_conf >= CONFIDENCE_THRESHOLD:
                    top_label = self.index_to_label[top_idx]
                    self.recent_predictions.append(top_label)
                    self.current_display_text = top_label
                else:
                    top_label = None
                    self.recent_predictions.append(None)
                    self.current_display_text = "..."
                self.current_confidence = top_conf

                if top_label == self.last_spoken_label and top_conf >= CONFIDENCE_THRESHOLD:
                    self.mismatch_streak = 0
                else:
                    self.mismatch_streak += 1
                if self.mismatch_streak >= RESET_AFTER_MISMATCH_FRAMES:
                    self.last_spoken_label = None

                if (len(self.recent_predictions) == SMOOTHING_WINDOW
                        and None not in self.recent_predictions
                        and len(set(self.recent_predictions)) == 1):
                    confirmed_label = self.recent_predictions[0]
                    if confirmed_label != self.last_spoken_label:
                        self.sentence.append(confirmed_label)
                        self.speaker.say(confirmed_label)
                        self.last_spoken_label = confirmed_label
                        self.mismatch_streak = 0

        #Draw overlay directly on the frame
        cv2.putText(img, f"Prediction: {self.current_display_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Confidence: {self.current_confidence*100:.0f}%", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        sentence_text = " ".join(self.sentence) if self.sentence else "(empty)"
        cv2.putText(img, f"Sentence: {sentence_text}", (10, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit page

st.title("Real-Time ISL Recognition & Speech Assistant")
st.caption(
    "Recognizes a defined vocabulary of Indian Sign Language signs from live "
    "webcam video and speaks the result aloud."
)

if not os.path.exists(MODEL_PATH_KERAS):
    st.error(f"Model not found at {MODEL_PATH_KERAS}. Run train.py first.")
elif not os.path.exists(MODEL_PATH_TASK):
    st.error(f"MediaPipe model not found at {MODEL_PATH_TASK}.")
else:
    ctx = webrtc_streamer(
        key="isl-recognition",
        video_processor_factory=ISLVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

    st.caption(
        "Prediction, confidence, and your built sentence are shown directly "
        "on the video feed above."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear sentence", use_container_width=True):
            if ctx.video_processor:
                ctx.video_processor.sentence.clear()
                ctx.video_processor.recent_predictions.clear()
                ctx.video_processor.last_spoken_label = None
                ctx.video_processor.mismatch_streak = 0
    with col2:
        if st.button("Speak sentence", use_container_width=True):
            if ctx.video_processor and ctx.video_processor.sentence:
                ctx.video_processor.speaker.say(" ".join(ctx.video_processor.sentence))