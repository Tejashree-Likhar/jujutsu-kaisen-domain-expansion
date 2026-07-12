"""
Collect training data for one, several, or all domain hand-signs.

Usage:
    python src/data_collection.py                     # walks through ALL labels
    python src/data_collection.py --label gojo         # (re)record just one
    python src/data_collection.py --label neutral --samples 150

Controls while running:
    SPACE  -> start/stop recording the current label
    R      -> retake (clears samples recorded so far for current label)
    N      -> skip to next label
    Q      -> quit entirely (progress already saved is kept)

For each label we save:
    data/<label>.npy          -> (N, 126) float32 array of landmark vectors
    assets/signs/<label>.jpg  -> one representative thumbnail (first frame
                                  captured with a hand visible), used later
                                  by the instructions panel in main.py
"""

import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.domains import DOMAINS, ALL_LABELS, NEUTRAL_LABEL, SAMPLES_PER_CLASS
from src.hand_utils import HandTracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
THUMB_DIR = os.path.join(ROOT, "assets", "signs")


def label_display_name(label):
    if label == NEUTRAL_LABEL:
        return "NEUTRAL (no active sign / resting hands)"
    d = DOMAINS[label]
    return f"{d['domain_name']}  ({d['user_name']})"


def collect_for_label(cap, tracker, label, target_samples):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

    samples = []
    thumb_path = os.path.join(THUMB_DIR, f"{label}.jpg")
    saved_thumb = os.path.exists(thumb_path)
    recording = False

    print(f"\n=== Collecting: {label_display_name(label)} ===")
    print(f"Target samples: {target_samples}")
    print("SPACE = start/stop recording | R = retake | N = next | Q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        results = tracker.process(frame)
        vec, hands_present = tracker.feature_vector(results)
        display = tracker.draw(frame.copy(), results)

        if recording and hands_present > 0:
            samples.append(vec)
            if not saved_thumb:
                cv2.imwrite(thumb_path, frame)
                saved_thumb = True

        status_color = (0, 255, 0) if recording else (0, 0, 255)
        cv2.rectangle(display, (0, 0), (display.shape[1], 90), (20, 20, 20), -1)
        cv2.putText(display, label_display_name(label), (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Samples: {len(samples)}/{target_samples}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        state_txt = "RECORDING" if recording else "PAUSED (space to record)"
        cv2.putText(display, state_txt, (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        if hands_present == 0:
            cv2.putText(display, "No hand detected", (15, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Data Collection - Domain Expansion", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
        elif key == ord('r'):
            samples = []
            saved_thumb = False
            recording = False
            print("Retaking...")
        elif key == ord('n'):
            break
        elif key == ord('q'):
            _save(label, samples)
            return "quit"

        if len(samples) >= target_samples:
            print(f"Reached target for {label}.")
            break

    _save(label, samples)
    return "next"


def _save(label, samples):
    if not samples:
        print(f"No samples captured for {label}, nothing saved.")
        return
    arr = np.array(samples, dtype=np.float32)
    out_path = os.path.join(DATA_DIR, f"{label}.npy")
    # append to existing data if present, so you can top up in multiple runs
    if os.path.exists(out_path):
        existing = np.load(out_path)
        arr = np.concatenate([existing, arr], axis=0)
    np.save(out_path, arr)
    print(f"Saved {arr.shape[0]} total samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=None,
                         choices=ALL_LABELS,
                         help="Collect only this label. Default: walk through all.")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    labels = [args.label] if args.label else ALL_LABELS

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: could not open camera index", args.camera)
        return

    tracker = HandTracker()

    try:
        for label in labels:
            result = collect_for_label(cap, tracker, label, args.samples)
            if result == "quit":
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print("\nDone. Now run: python src/train_model.py")


if __name__ == "__main__":
    main()
