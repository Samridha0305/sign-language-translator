# Real-Time ISL Recognition & Speech Assistant

Recognizes a defined vocabulary of Indian Sign Language (ISL) signs from live
webcam video and speaks the recognized word aloud. Built as a portfolio
project to explore hand-landmark-based gesture recognition with a
sequence model (GRU).

> **Scope note:** this is a sign *recognition* system for a fixed vocabulary
> of ~10-25 signs, not a full ISL translator.

## Demo

<!-- Replace with an actual GIF/video once you have one, e.g.: -->
<!-- ![demo](reports/figures/demo.gif) -->

## How it works

```
Webcam → MediaPipe hand landmarks → normalize → 40-frame buffer
       → GRU classifier → confidence check → prediction smoothing
       → text overlay → speech (pyttsx3)
```

- **Landmarks, not raw pixels:** each frame is reduced to 126 numbers
  (2 hands × 21 landmarks × 3 coordinates), which is far easier to learn
  from than raw video.
- **Sequence model:** signs unfold over time, so a GRU classifies a
  40-frame window rather than a single frame.
- **Smoothing:** predictions are confirmed only after several consecutive
  high-confidence frames agree, to avoid flickering/repeated output.

## Vocabulary

<!-- Fill in your final sign list, e.g.: -->
| Category | Signs |
|---|---|
| Greetings/social | HELLO, THANK YOU |
| Common needs | WATER, HELP |
| Questions | WHERE |
| Emergency | EMERGENCY, PAIN |

## Dataset

- Self-recorded via webcam using MediaPipe Hand Landmarker.
- `<N>` signs × `<N>` participants × `<N>` samples/sign.
- **Train/test split by participant** (not randomly by sample) — the test
  set includes at least one participant excluded entirely from training,
  so the reported accuracy reflects generalization to a new signer rather
  than memorization of one person's hand.

## Results

<!-- Fill in once trained -->
| Model | Test accuracy | Unseen-participant accuracy |
|---|---|---|
| MLP (static baseline) | | |
| GRU | | |

## Project structure

See `sign-language-translator/` tree — `src/capture` for data recording,
`src/models` for training, `app/` for the live webcam application.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
# Record training data for one sign/participant
python src/capture/record_dataset.py --label HELLO --participant p01 --samples 25

# Train the model (once implemented)
python src/models/train.py

# Run the live app
python app/main.py
```

## Limitations

- Small, fixed vocabulary — not general ISL translation.
- Trained on a small number of participants; accuracy on new signers,
  skin tones, or camera setups is not guaranteed.
- Sensitive to lighting and background clutter.
- No handling of ISL grammar/sentence structure — outputs isolated signs.

## Future improvements

- Expand vocabulary and participant pool.
- Add facial/non-manual cues (MediaPipe Holistic).
- Rotation-invariant landmark normalization.
- On-device/mobile deployment.

## Tech stack

Python · OpenCV · MediaPipe · NumPy/Pandas · TensorFlow/Keras (GRU) · pyttsx3