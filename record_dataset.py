"""
record_dataset.py


Records labeled sign-language gesture sequences using your webcam + MediaPipe
Hand Landmarker (Tasks API), and saves them as .npy landmark sequences.

"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SEQ_LEN = 40  
MODEL_PATH = "hand_landmarker.task"


def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Sign being recorded, e.g. HELLO")
    parser.add_argument("--participant", required=True, help="Participant ID, e.g. p01")
    parser.add_argument("--samples", type=int, default=25, help="How many samples to record this session")
    parser.add_argument("--out_dir", default="data/raw", help="Root output directory")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, args.participant, args.label)
    os.makedirs(out_dir, exist_ok=True)

    meta_dir = "data/metadata"
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "samples.csv")
    write_header = not os.path.exists(meta_path)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])

    cap = cv2.VideoCapture(0)
    detector = build_detector()
    timestamp = {"ms": 0}  # mutable counter shared with helper function

    recorded_this_session = 0
    print(f"Ready. Label={args.label} Participant={args.participant}")
    print("Press SPACE to record a sample, q to quit.")

    with open(meta_path, "a", newline="") as meta_file:
        writer = csv.writer(meta_file)
        if write_header:
            writer.writerow(["label", "participant", "filepath", "timestamp"])

        while cap.isOpened() and recorded_this_session < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp["ms"] += 33
            result = detector.detect_for_video(mp_image, timestamp["ms"])
            draw_landmarks_simple(frame, result)

            status = f"{args.label} | sample {existing + recorded_this_session + 1}/{existing + args.samples}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=record  q=quit", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Record Dataset", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                sequence = record_one_sample(cap, detector, timestamp)
                idx = existing + recorded_this_session
                fname = f"sample_{idx:03d}.npy"
                fpath = os.path.join(out_dir, fname)
                np.save(fpath, sequence)
                writer.writerow([args.label, args.participant, fpath, datetime.now().isoformat()])
                meta_file.flush()
                recorded_this_session += 1
                print(f"Saved {fpath}  ({recorded_this_session}/{args.samples})")

    cap.release()
    cv2.destroyAllWindows()
    print("Done for this session.")


def record_one_sample(cap, detector, timestamp):
    """Capture SEQ_LEN frames right now and return a (SEQ_LEN, 126) array."""
    countdown_start = time.time()
    while time.time() - countdown_start < 0.6:  # brief pause so you can get into position
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Get ready...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Record Dataset", frame)
        cv2.waitKey(1)

    frames = []
    while len(frames) < SEQ_LEN:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp["ms"] += 33
        result = detector.detect_for_video(mp_image, timestamp["ms"])
        landmarks = extract_landmarks(result)
        frames.append(landmarks)

        draw_landmarks_simple(frame, result)
        cv2.putText(frame, f"RECORDING {len(frames)}/{SEQ_LEN}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Record Dataset", frame)
        cv2.waitKey(1)

    return np.array(frames)  


if __name__ == "__main__":
    main()