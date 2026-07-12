"""
Live Domain Expansion recognizer - finger-counting edition.

Show 1-6 fingers (split across one or both hands however you like, e.g.
6 = 5 on one hand + 1 on the other) and hold steady for ~0.6s to cast
that domain: a particle swirl converges and spins across the whole
screen in the domain's colors, then explodes into a colored overlay,
with the name/user/description shown top-left. Your live camera stays
visible in a small inset, bottom-left, the whole time.

Controls: ESC or Q to quit.

Usage:
    python src/main.py
"""

import os
import sys
import time
from collections import deque, Counter

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.domains import (
    DOMAINS, NUMBER_TO_DOMAIN, HOLD_FRAMES_TO_TRIGGER, COOLDOWN_SECONDS,
)
from src.hand_utils import HandTracker
from src.gesture import total_fingers
from src.effects import DomainEffect
from src.ui import draw_info_panel, draw_status_text, draw_instructions_panel

WINDOW_NAME = "Domain Expansion"


def get_screen_size(fallback=(1920, 1080)):
    """Query the real monitor resolution so the fullscreen canvas isn't
    stretched/blurry. Falls back to a common default if that fails (e.g.
    no display driver available)."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return fallback


def main():
    screen_w, screen_h = get_screen_size()

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Camera inset geometry: bottom-left corner, ~30% of screen width.
    inset_w = int(screen_w * 0.30)
    inset_h = int(inset_w * 3 / 4)  # 4:3, resized to fit regardless of real cam aspect
    margin = 28
    inset_x0 = margin
    inset_y0 = screen_h - inset_h - margin

    recent = deque(maxlen=HOLD_FRAMES_TO_TRIGGER)
    active_effect = None
    active_domain_key = None
    last_trigger_time = -999.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            results = tracker.process(frame)
            frame_annotated = tracker.draw(frame.copy(), results)

            count = total_fingers(results.hand_landmarks)
            current_number = count if 1 <= count <= 6 else 0
            recent.append(current_number)

            now = time.time()
            can_trigger = (
                active_effect is None
                and now - last_trigger_time > COOLDOWN_SECONDS
                and len(recent) == recent.maxlen
            )
            if can_trigger:
                top_num, top_count = Counter(recent).most_common(1)[0]
                if top_num != 0 and top_count == recent.maxlen:
                    domain_key = NUMBER_TO_DOMAIN[top_num]
                    active_domain_key = domain_key
                    active_effect = DomainEffect(screen_w, screen_h, DOMAINS[domain_key])
                    active_effect.start()
                    last_trigger_time = now
                    recent.clear()

            # ---- Compose the fullscreen canvas ----
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            show_info_panel = False
            if active_effect is not None:
                canvas = active_effect.update_and_draw(canvas)
                show_info_panel = active_effect.is_active_phase() or active_effect.phase == "fade"
                if active_effect.finished():
                    active_effect = None
                    active_domain_key = None

            # Camera inset drawn AFTER the effect so it always stays clear/live,
            # matching the reference clips (small live thumbnail, big transformed backdrop).
            cam_small = cv2.resize(frame_annotated, (inset_w, inset_h))
            canvas[inset_y0:inset_y0 + inset_h, inset_x0:inset_x0 + inset_w] = cam_small
            cv2.rectangle(
                canvas,
                (inset_x0 - 2, inset_y0 - 2),
                (inset_x0 + inset_w + 2, inset_y0 + inset_h + 2),
                (255, 255, 255), 2,
            )

            if show_info_panel and active_domain_key:
                canvas = draw_info_panel(canvas, DOMAINS[active_domain_key])
            elif active_effect is None:
                draw_status_text(canvas, inset_x0, inset_y0, count)

            draw_instructions_panel(canvas, DOMAINS, NUMBER_TO_DOMAIN)

            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            # Also bail out if the user closes the window itself.
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
