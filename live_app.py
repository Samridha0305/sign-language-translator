
import collections
import json
import os
import threading

import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow import keras

MODEL_PATH_TASK = "hand_landmarker.task"
MODEL_PATH_KERAS = "models/best_model.keras"
LABELS_PATH = "data/processed/labels.json"

SEQ_LEN = 40
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
EPS = 1e-6

CONFIDENCE_THRESHOLD = 0.80   
SMOOTHING_WINDOW = 7          
COOLDOWN_FRAMES = 25          




class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()

    def say(self, text):
        threading.Thread(target=self._say_now, args=(text,), daemon=True).start()

    def _say_now(self, text):
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait()




def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH_TASK)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


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




def main():
    if not os.path.exists(MODEL_PATH_KERAS):
        print(f"Model not found at {MODEL_PATH_KERAS}. Run train.py first.")
        return

    with open(LABELS_PATH) as f:
        label_maps = json.load(f)
    index_to_label = {int(k): v for k, v in label_maps["index_to_label"].items()}
    num_classes = len(index_to_label)

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH_KERAS)
    speaker = Speaker()

    cap = cv2.VideoCapture(0)
    detector = build_detector()
    timestamp = 0

    frame_buffer = collections.deque(maxlen=SEQ_LEN)
    recent_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)

    last_spoken_label = None
    cooldown_counter = 0
    current_display_text = "..."
    current_confidence = 0.0

    print("Live app running. Press q to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp += 33
        result = detector.detect_for_video(mp_image, timestamp)
        draw_landmarks_simple(frame, result)

        landmarks = extract_landmarks(result)
        frame_buffer.append(landmarks)

        if cooldown_counter > 0:
            cooldown_counter -= 1

        if len(frame_buffer) == SEQ_LEN:
            seq = np.array(frame_buffer)
            normalized_seq = normalize_sequence(seq)
            probs = model.predict(normalized_seq[np.newaxis, ...], verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])

            if top_conf >= CONFIDENCE_THRESHOLD:
                recent_predictions.append(index_to_label[top_idx])
            else:
                recent_predictions.append(None)

            current_display_text = index_to_label[top_idx] if top_conf >= CONFIDENCE_THRESHOLD else "..."
            current_confidence = top_conf

            if (len(recent_predictions) == SMOOTHING_WINDOW
                    and len(set(recent_predictions)) == 1
                    and recent_predictions[0] is not None):
                confirmed_label = recent_predictions[0]

                if confirmed_label != last_spoken_label or cooldown_counter == 0:
                    speaker.say(confirmed_label)
                    last_spoken_label = confirmed_label
                    cooldown_counter = COOLDOWN_FRAMES
                    print(f"Confirmed + spoken: {confirmed_label} ({top_conf:.2f})")


        cv2.putText(frame, f"Prediction: {current_display_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {current_confidence*100:.0f}%", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "q = quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("ISL Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()