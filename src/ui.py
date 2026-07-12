"""
Drawing helpers for the two HUD panels:

  - Top-left     : active domain's name / user / description (only while
                    a domain expansion is in its "active" phase).
  - Bottom-right : always-on reference gallery of every trained sign,
                    using the thumbnail captured during data collection,
                    so you have a live cheat-sheet of your own poses.
"""

import os
import textwrap
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wrap_text(text, width_chars):
    return textwrap.wrap(text, width=width_chars)


def draw_info_panel(frame, domain_cfg):
    """Top-left panel shown while a domain is active/fading."""
    h, w = frame.shape[:2]
    panel_w = min(560, int(w * 0.5))
    lines = _wrap_text(domain_cfg["description"], 44)
    panel_h = 110 + 22 * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    accent = domain_cfg["color_primary"]
    cv2.rectangle(frame, (0, 0), (panel_w, 6), accent, -1)

    y = 34
    cv2.putText(frame, f"Domain Expansion: {domain_cfg['domain_name']}",
                (16, y), FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    y += 32
    cv2.putText(frame, f"User: {domain_cfg['user_name']}",
                (16, y), FONT, 0.62, (200, 220, 255), 2, cv2.LINE_AA)
    y += 30
    for line in lines:
        cv2.putText(frame, line, (16, y), FONT, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        y += 22

    return frame


def draw_status_bar(frame, text, color=(255, 255, 255)):
    """Small transient debug/status line, bottom-left, e.g. detection
    confidence while you hold a pose."""
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (16, h - 16), FONT, 0.55, color, 2, cv2.LINE_AA)
    return frame


class InstructionsPanel:
    """Always-visible bottom-right gallery: one thumbnail + label per
    trained sign, loaded once from assets/signs/<label>.jpg."""

    THUMB_SIZE = 64
    PADDING = 8

    def __init__(self, root_dir, labels, domains_cfg, neutral_label):
        self.thumbs = {}
        thumb_dir = os.path.join(root_dir, "assets", "signs")
        for label in labels:
            path = os.path.join(thumb_dir, f"{label}.jpg")
            img = cv2.imread(path) if os.path.exists(path) else None
            if img is not None:
                img = cv2.resize(img, (self.THUMB_SIZE, self.THUMB_SIZE))
            self.thumbs[label] = img

        self.entries = []
        for label in labels:
            if label == neutral_label:
                name = "Neutral (idle)"
            else:
                d = domains_cfg[label]
                name = f"{d['domain_name']}"
            self.entries.append((label, name))

    def draw(self, frame):
        h, w = frame.shape[:2]
        row_h = self.THUMB_SIZE + self.PADDING
        panel_h = row_h * len(self.entries) + self.PADDING
        panel_w = 300
        x0 = w - panel_w
        y0 = h - panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (w, h), (15, 15, 15), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "Sign Reference", (x0 + 10, y0 - 10),
                    FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        y = y0 + self.PADDING
        for label, name in self.entries:
            thumb = self.thumbs.get(label)
            tx = x0 + self.PADDING
            if thumb is not None:
                frame[y:y + self.THUMB_SIZE, tx:tx + self.THUMB_SIZE] = thumb
            else:
                cv2.rectangle(frame, (tx, y), (tx + self.THUMB_SIZE, y + self.THUMB_SIZE),
                              (60, 60, 60), -1)
                cv2.putText(frame, "?", (tx + 24, y + 40), FONT, 0.8, (150, 150, 150), 2)

            text_x = tx + self.THUMB_SIZE + 10
            text_y = y + self.THUMB_SIZE // 2 + 5
            cv2.putText(frame, name, (text_x, text_y), FONT, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += row_h

        return frame
