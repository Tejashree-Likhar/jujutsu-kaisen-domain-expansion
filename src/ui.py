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

    MAX_THUMB_SIZE = 64
    MIN_THUMB_SIZE = 30
    PADDING = 8
    PANEL_W = 300
    LABEL_H = 26  # space reserved above the panel for the "Sign Reference" title

    def __init__(self, root_dir, labels, domains_cfg, neutral_label):
        self.raw_thumbs = {}
        thumb_dir = os.path.join(root_dir, "assets", "signs")
        for label in labels:
            path = os.path.join(thumb_dir, f"{label}.jpg")
            img = cv2.imread(path) if os.path.exists(path) else None
            self.raw_thumbs[label] = img  # kept at native size, resized per-draw

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
        n = max(1, len(self.entries))

        # Fit thumbnails to whatever vertical space is actually available in
        # this frame (webcams commonly deliver 480p, not the 720p this was
        # first designed against), shrinking below MAX_THUMB_SIZE if needed.
        available_h = max(1, h - self.LABEL_H - 10)
        thumb_size = min(
            self.MAX_THUMB_SIZE,
            max(self.MIN_THUMB_SIZE, available_h // n - self.PADDING),
        )
        row_h = thumb_size + self.PADDING
        panel_h = row_h * n + self.PADDING
        panel_w = self.PANEL_W
        x0 = w - panel_w
        y0 = max(self.LABEL_H, h - panel_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (w, h), (15, 15, 15), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "Sign Reference", (x0 + 10, max(18, y0 - 10)),
                    FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        y = y0 + self.PADDING
        for label, name in self.entries:
            raw = self.raw_thumbs.get(label)
            tx = x0 + self.PADDING
            if raw is not None:
                thumb = cv2.resize(raw, (thumb_size, thumb_size))
                frame[y:y + thumb_size, tx:tx + thumb_size] = thumb
            else:
                cv2.rectangle(frame, (tx, y), (tx + thumb_size, y + thumb_size),
                              (60, 60, 60), -1)
                cv2.putText(frame, "?", (tx + thumb_size // 3, y + int(thumb_size * 0.65)),
                            FONT, 0.6, (150, 150, 150), 2)

            text_x = tx + thumb_size + 10
            text_y = y + thumb_size // 2 + 5
            cv2.putText(frame, name, (text_x, text_y), FONT, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += row_h

        return frame
