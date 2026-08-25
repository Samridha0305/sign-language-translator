
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Standard 21-point hand connections (wrist=0 ... pinky_tip=20), defined
# manually so we don't depend on the deprecated mp.solutions submodule.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


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


def draw_landmarks(frame, hand_landmarks_list, handedness_list):
    h, w, _ = frame.shape
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # We mirror the frame for a natural selfie view, which also flips
        # MediaPipe's left/right classification -- so swap the label back.
        raw_label = handedness[0].category_name  # "Left" or "Right"
        label = "Right" if raw_label == "Left" else "Left"
        wx, wy = points[0]
        cv2.putText(frame, label, (wx - 20, wy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check that it's connected and not used by another app.")
        return

    detector = build_detector()
    frame_timestamp_ms = 0

    print("Webcam running. Press q to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33  # roughly 30fps step; must be increasing
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        num_hands = 0
        if result.hand_landmarks:
            num_hands = len(result.hand_landmarks)
            draw_landmarks(frame, result.hand_landmarks, result.handedness)

        cv2.putText(frame, f"Hands detected: {num_hands}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "q = quit", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Hand Landmark Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()