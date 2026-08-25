"""
validate_dataset.py

Two tools in one, for sanity-checking your recorded dataset before training:

1. REPORT MODE (default) - scans every .npy sample under data/raw/ and flags
   ones that look bad: too many frames where no hand was detected at all.

   USAGE:
       python validate_dataset.py --report

2. REPLAY MODE - plays back one specific sample as a stick-figure animation,
   so you can visually confirm it actually looks like the sign it's labeled as.

   USAGE:
       python validate_dataset.py --replay data/raw/p01/HELLO/sample_003.npy

CONTROLS (replay mode):
    q   -> quit the replay window
"""

import argparse
import os
import time

import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def frame_has_hand(vec63):
    """A hand slot is all zeros if that hand wasn't detected in this frame."""
    return not np.all(vec63 == 0)


def report(root="data/raw", zero_frame_threshold=0.5):
    """Walk every sample file and flag ones missing hands in too many frames."""
    flagged = []
    total = 0

    for participant in sorted(os.listdir(root)):
        p_dir = os.path.join(root, participant)
        if not os.path.isdir(p_dir):
            continue
        for label in sorted(os.listdir(p_dir)):
            l_dir = os.path.join(p_dir, label)
            if not os.path.isdir(l_dir):
                continue
            samples = sorted(f for f in os.listdir(l_dir) if f.endswith(".npy"))
            bad_in_label = 0

            for fname in samples:
                total += 1
                fpath = os.path.join(l_dir, fname)
                seq = np.load(fpath)  # shape (SEQ_LEN, 126)

                no_hand_frames = 0
                for frame in seq:
                    left, right = frame[:63], frame[63:]
                    if not frame_has_hand(left) and not frame_has_hand(right):
                        no_hand_frames += 1

                fraction_missing = no_hand_frames / len(seq)
                if fraction_missing > zero_frame_threshold:
                    flagged.append((fpath, fraction_missing))
                    bad_in_label += 1

            status = f"  {bad_in_label} flagged" if bad_in_label else "  OK"
            print(f"{participant}/{label}: {len(samples)} samples{status}")

    print(f"\nTotal samples: {total}")
    if flagged:
        print(f"\n{len(flagged)} sample(s) flagged (>{int(zero_frame_threshold*100)}% frames with no hand detected):")
        for fpath, frac in flagged:
            print(f"  {fpath}  ({frac*100:.0f}% missing)")
        print("\nOpen these with --replay to check them, and delete + re-record if they look wrong.")
    else:
        print("\nNo samples flagged. Doesn't guarantee correct signs, but hand tracking looks solid throughout.")


def replay(fpath, fps=10):
    """Show a sample as an animated stick figure so you can eyeball it."""
    seq = np.load(fpath)
    print(f"Replaying {fpath} - shape {seq.shape}")

    canvas_size = 500
    delay = 1.0 / fps

    for frame_idx, frame in enumerate(seq):
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        left, right = frame[:63], frame[63:]

        for hand_vec, color in [(left, (0, 165, 255)), (right, (255, 165, 0))]:
            if not frame_has_hand(hand_vec):
                continue
            points = hand_vec.reshape(21, 3)
            pts_2d = [(int(x * canvas_size), int(y * canvas_size)) for x, y, z in points]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(canvas, pts_2d[start_idx], pts_2d[end_idx], color, 2)
            for x, y in pts_2d:
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

        cv2.putText(canvas, f"frame {frame_idx+1}/{len(seq)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "orange=left  blue=right  q=quit", (10, canvas_size - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Replay Sample", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(delay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Scan all samples and flag bad ones")
    parser.add_argument("--replay", type=str, default=None, help="Path to a single .npy sample to replay")
    parser.add_argument("--root", type=str, default="data/raw", help="Root data folder for --report")
    args = parser.parse_args()

    if args.replay:
        replay(args.replay)
    else:
        report(root=args.root)


if __name__ == "__main__":
    main()