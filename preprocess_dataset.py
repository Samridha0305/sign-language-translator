"""
preprocess_dataset.py

Combines every recorded .npy sample under data/raw/ into training-ready
arrays: normalizes landmarks, encodes labels, and splits into
train/val/test.

USAGE:
    python preprocess_dataset.py

OUTPUT (in data/processed/):
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    labels.json   -> {"label_to_index": {...}, "index_to_label": {...}}

SPLIT LOGIC:
    If more than one participant exists under data/raw/, splits BY
    PARTICIPANT (held-out signer test set) -- the credible split from
    the roadmap.
    If only one participant exists (e.g. just p01 so far), falls back to
    a random 70/15/15 split, and prints a warning that this is a
    temporary measure until more participants are added.

NORMALIZATION (per hand, per frame):
    1. Translate: subtract the wrist position, so the hand is
       position-independent (doesn't matter where in the frame it is).
    2. Scale: divide by the wrist-to-middle-finger-MCP distance, so hand
       size / distance from camera doesn't matter.
    A hand that wasn't detected (all zeros) is left as all zeros.
"""

import json
import os

import numpy as np

RAW_ROOT = "data/raw"
OUT_DIR = "data/processed"
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
EPS = 1e-6


def normalize_hand(hand_vec63):
    """Wrist-center and scale-normalize one hand's 21x3 landmarks."""
    if not np.any(hand_vec63):
        return hand_vec63  # missing hand -> stays zeros

    points = hand_vec63.reshape(21, 3)
    wrist = points[WRIST_IDX]
    translated = points - wrist  # position-independent

    scale_ref = np.linalg.norm(translated[MIDDLE_MCP_IDX])
    scale_ref = max(scale_ref, EPS)  # avoid divide-by-zero
    normalized = translated / scale_ref

    return normalized.flatten()


def normalize_sequence(seq):
    """Apply normalize_hand to every frame's left and right hand."""
    normalized_seq = np.zeros_like(seq)
    for i, frame in enumerate(seq):
        left, right = frame[:63], frame[63:]
        normalized_seq[i, :63] = normalize_hand(left)
        normalized_seq[i, 63:] = normalize_hand(right)
    return normalized_seq


def load_all_samples(root=RAW_ROOT):
    """Returns lists: sequences, labels, participants."""
    sequences, labels, participants = [], [], []

    for participant in sorted(os.listdir(root)):
        p_dir = os.path.join(root, participant)
        if not os.path.isdir(p_dir):
            continue
        for label in sorted(os.listdir(p_dir)):
            l_dir = os.path.join(p_dir, label)
            if not os.path.isdir(l_dir):
                continue
            for fname in sorted(os.listdir(l_dir)):
                if not fname.endswith(".npy"):
                    continue
                seq = np.load(os.path.join(l_dir, fname))
                sequences.append(normalize_sequence(seq))
                labels.append(label)
                participants.append(participant)

    return sequences, labels, participants


def build_label_maps(labels):
    unique_labels = sorted(set(labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}
    return label_to_index, index_to_label


def random_split(n, train_frac=0.7, val_frac=0.15, seed=42):
    """Returns boolean-index arrays for train/val/test given n samples."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def participant_split(participants, seed=42):
    """Holds out the LAST participant (alphabetically) entirely for test.
    With 3+ participants, also holds out the second-to-last entirely for
    validation. With exactly 2 participants, there's no third person to
    give validation, so instead we carve a random 15% slice OUT OF THE
    TRAINING participant(s) for validation -- p02 (test) stays fully
    untouched either way, so your test accuracy is still a genuine
    unseen-person number."""
    unique_participants = sorted(set(participants))
    participants = np.array(participants)
    test_people = {unique_participants[-1]}
    test_idx = np.where(np.isin(participants, list(test_people)))[0]

    if len(unique_participants) >= 3:
        val_people = {unique_participants[-2]}
        val_idx = np.where(np.isin(participants, list(val_people)))[0]
        train_idx = np.where(
            ~np.isin(participants, list(test_people) + list(val_people))
        )[0]
    else:
        val_people = set()  # no dedicated val participant
        train_people_idx = np.where(~np.isin(participants, list(test_people)))[0]
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(train_people_idx)
        val_cut = int(len(shuffled) * 0.15)
        val_idx = shuffled[:val_cut]
        train_idx = shuffled[val_cut:]

    return train_idx, val_idx, test_idx, test_people, val_people


def main():
    print("Loading and normalizing samples...")
    sequences, labels, participants = load_all_samples()
    n = len(sequences)
    print(f"Loaded {n} samples across {len(set(labels))} labels, "
          f"{len(set(participants))} participant(s).")

    label_to_index, index_to_label = build_label_maps(labels)
    y = np.array([label_to_index[l] for l in labels])
    X = np.array(sequences)  # shape (n, SEQ_LEN, 126)

    unique_participants = sorted(set(participants))
    if len(unique_participants) > 1:
        train_idx, val_idx, test_idx, test_people, val_people = participant_split(participants)
        if val_people:
            print(f"Splitting BY PARTICIPANT. Test person(s): {test_people}, "
                  f"Val person(s): {val_people}")
        else:
            print(f"Splitting BY PARTICIPANT. Test person(s): {test_people} "
                  f"(fully held out). Only 2 participants total, so validation "
                  f"is a random 15% slice carved out of the remaining training "
                  f"participant(s) -- not a separate person, but the test set "
                  f"above is still untouched and unseen.")
    else:
        print("WARNING: only one participant found -- using a random "
              "70/15/15 split for now. This does NOT test generalization "
              "to a new signer. Add more participants and re-run this "
              "script once you can, and mention this limitation in your README.")
        train_idx, val_idx, test_idx = random_split(n)

    os.makedirs(OUT_DIR, exist_ok=True)

    def save_split(name, idx):
        np.save(os.path.join(OUT_DIR, f"X_{name}.npy"), X[idx])
        np.save(os.path.join(OUT_DIR, f"y_{name}.npy"), y[idx])
        print(f"  {name}: {len(idx)} samples")

    print("\nSaving splits:")
    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(
            {"label_to_index": label_to_index, "index_to_label": index_to_label},
            f,
            indent=2,
        )
    print(f"\nSaved label map to {os.path.join(OUT_DIR, 'labels.json')}")

    print("\nPer-label sample counts:")
    for label in sorted(label_to_index):
        count = labels.count(label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()