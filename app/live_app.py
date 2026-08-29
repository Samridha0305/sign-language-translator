
import collections
import json
import os
import queue
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
SMOOTHING_WINDOW = 8         
PREDICT_EVERY_N_FRAMES = 2    
 
 

 
class Speaker:

 
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
 
    def say(self, text):
        self.queue.put(text)
 
    def _worker(self):
        while True:
            text = self.queue.get()
     
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

 
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
 
 

 
class Predictor:

 
    def __init__(self, model, index_to_label):
        self.model = model
        self.index_to_label = index_to_label
        self.input_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_label = None
        self.latest_conf = 0.0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
 
    def submit(self, seq):
        # Keep only the newest buffer -- drop any stale one waiting in queue.
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        self.input_queue.put_nowait(seq)
 
    def get_latest(self):
        with self.lock:
            return self.latest_label, self.latest_conf
 
    def _worker(self):
        while True:
            seq = self.input_queue.get()
            normalized_seq = normalize_sequence(seq)
            input_tensor = normalized_seq[np.newaxis, ...].astype(np.float32)
            probs = self.model(input_tensor, training=False).numpy()[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            with self.lock:
                self.latest_label = self.index_to_label[top_idx]
                self.latest_conf = top_conf
 
 
# Main app 
 
def main():
    if not os.path.exists(MODEL_PATH_KERAS):
        print(f"Model not found at {MODEL_PATH_KERAS}. Run train.py first.")
        return

    if not os.path.exists(LABELS_PATH):
        print(f"Labels not found at {LABELS_PATH}.")
        return

    if not os.path.exists(MODEL_PATH_TASK):
        print(f"MediaPipe model not found at {MODEL_PATH_TASK}.")
        return

    with open(LABELS_PATH) as f:
        label_maps = json.load(f)

    index_to_label = {
        int(k): v for k, v in label_maps["index_to_label"].items()
    }

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH_KERAS)

    speaker = Speaker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    detector = build_detector()

    timestamp = 0
    predictor = Predictor(model, index_to_label)

    frame_buffer = collections.deque(maxlen=SEQ_LEN)
    recent_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)

    frame_counter = 0

    
    sentence = []


    last_spoken_label = None

    mismatch_streak = 0
    RESET_AFTER_MISMATCH_FRAMES = 6

    current_display_text = "..."
    current_confidence = 0.0

    print("Live app running.")
    print("Controls:")
    print("  q = quit")
    print("  c = clear sentence")
    print("  s = speak sentence")
    print()

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok:
            print("Could not read frame from webcam.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 33

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        draw_landmarks_simple(frame, result)



    

        landmarks = extract_landmarks(result)

        frame_buffer.append(landmarks)


        if len(frame_buffer) == SEQ_LEN:

            frame_counter += 1

            if frame_counter % PREDICT_EVERY_N_FRAMES == 0:

                predictor.submit(
                    np.array(frame_buffer)
                )

            top_label, top_conf = predictor.get_latest()

            if top_label is not None:

                current_confidence = top_conf


                if top_conf >= CONFIDENCE_THRESHOLD:

                    recent_predictions.append(top_label)

                    current_display_text = top_label

                else:

                    recent_predictions.append(None)

                    current_display_text = "..."


                if (
                    top_label == last_spoken_label
                    and top_conf >= CONFIDENCE_THRESHOLD
                ):

                    mismatch_streak = 0

                else:

                    mismatch_streak += 1

                if mismatch_streak >= RESET_AFTER_MISMATCH_FRAMES:

                    last_spoken_label = None

    

                if (
                    len(recent_predictions) == SMOOTHING_WINDOW
                    and None not in recent_predictions
                    and len(set(recent_predictions)) == 1
                ):

                    confirmed_label = recent_predictions[0]

                    if confirmed_label != last_spoken_label:

                        # Add sign to sentence
                        sentence.append(confirmed_label)

                        # Speak individual sign
                        speaker.say(confirmed_label)

                        last_spoken_label = confirmed_label
                        mismatch_streak = 0

                        print(
                            f"Confirmed: {confirmed_label} "
                            f"({top_conf:.2f})"
                        )

                        print(
                            "Sentence:",
                            " ".join(sentence)
                        )


        sentence_text = " ".join(sentence)

        cv2.putText(
            frame,
            f"Prediction: {current_display_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {current_confidence * 100:.0f}%",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )



        cv2.putText(
            frame,
            "Sentence:",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        words_per_line = 6

        sentence_lines = [
            sentence_text.split()[i:i + words_per_line]
            for i in range(
                0,
                len(sentence_text.split()),
                words_per_line
            )
        ]

        y = 135

        for line in sentence_lines:

            line_text = " ".join(line)

            cv2.putText(
                frame,
                line_text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y += 30


        cv2.putText(
            frame,
            "c = clear | s = speak sentence | q = quit",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )


        cv2.imshow(
            "ISL Live Recognition",
            frame
        )



        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


        elif key == ord("c"):

            sentence.clear()


            recent_predictions.clear()

            last_spoken_label = None
            mismatch_streak = 0

            print("Sentence cleared.")

        elif key == ord("s"):

            if sentence:

                complete_sentence = " ".join(sentence)

                print(
                    "Speaking sentence:",
                    complete_sentence
                )

                speaker.say(complete_sentence)

            else:

                print("Sentence is empty.")

    cap.release()
    cv2.destroyAllWindows()

    print("Application closed.")


if __name__ == "__main__":
    main()
 
