"""
Live Domain Expansion recognizer.

Hold a trained hand-sign steady for ~0.6s -> triggers that domain's
converge/spin -> explode -> active overlay sequence, with the
name/user/description panel top-left and the always-on sign reference
gallery bottom-right.

Usage:
    python src/main.py
Press Q to quit.
"""

import os
import sys
import time
from collections import deque, Counter

import cv2
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.domains import (
    DOMAINS, NEUTRAL_LABEL, ALL_LABELS,
    HOLD_FRAMES_TO_TRIGGER, CONFIDENCE_THRESHOLD,
    COOLDOWN_SECONDS,
)
from src.hand_utils import HandTracker
from src.effects import DomainEffect
from src.ui import draw_info_panel, draw_status_bar, InstructionsPanel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "domain_classifier.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            "No trained model found at models/domain_classifier.pkl.\n"
            "Run data_collection.py then train_model.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"]


def main():
    model, encoder = load_model()
    tracker = HandTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from camera.")
    h, w = frame.shape[:2]

    instructions_panel = InstructionsPanel(ROOT, ALL_LABELS, DOMAINS, NEUTRAL_LABEL)

    recent_predictions = deque(maxlen=HOLD_FRAMES_TO_TRIGGER)
    active_effect = None
    active_domain_key = None
    last_trigger_time = -999.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        results = tracker.process(frame)
        vec, hands_present = tracker.feature_vector(results)
        frame = tracker.draw(frame, results)

        pred_label = NEUTRAL_LABEL
        confidence = 0.0

        if hands_present > 0:
            proba = model.predict_proba([vec])[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            candidate = encoder.inverse_transform([best_idx])[0]
            if confidence >= CONFIDENCE_THRESHOLD:
                pred_label = candidate

        recent_predictions.append(pred_label)

        # Trigger logic: same non-neutral label held for enough consecutive
        # frames, and we're not already showing an effect, and cooldown ok.
        now = time.time()
        can_trigger = (
            active_effect is None
            and now - last_trigger_time > COOLDOWN_SECONDS
            and len(recent_predictions) == recent_predictions.maxlen
        )
        if can_trigger:
            counts = Counter(recent_predictions)
            top_label, top_count = counts.most_common(1)[0]
            if top_label != NEUTRAL_LABEL and top_count == recent_predictions.maxlen:
                active_domain_key = top_label
                active_effect = DomainEffect(w, h, DOMAINS[top_label])
                active_effect.start()
                last_trigger_time = now
                recent_predictions.clear()

        if active_effect is not None:
            frame = active_effect.update_and_draw(frame)
            if active_effect.is_active_phase() or active_effect.phase == "fade":
                frame = draw_info_panel(frame, DOMAINS[active_domain_key])
            if active_effect.finished():
                active_effect = None
                active_domain_key = None
        else:
            status_color = (0, 255, 0) if pred_label != NEUTRAL_LABEL else (200, 200, 200)
            label_txt = (
                DOMAINS[pred_label]["domain_name"] if pred_label != NEUTRAL_LABEL else "neutral"
            )
            draw_status_bar(
                frame,
                f"Detecting: {label_txt}  ({confidence*100:.0f}%)",
                status_color,
            )

        frame = instructions_panel.draw(frame)

        cv2.imshow("Domain Expansion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
