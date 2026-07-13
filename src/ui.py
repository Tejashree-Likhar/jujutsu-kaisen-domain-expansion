"""
Drawing helpers for the HUD:

  - Top-left      : active domain's name / user / description (only
                     while a domain expansion is in its active/fade
                     phase).
  - Bottom-right   : always-on, text-only legend of which number casts
                     which domain.
  - Near the camera inset : small status line showing fingers currently
                     detected, while no domain is active.
"""

import textwrap
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _wrap_text(text, width_chars):
    return textwrap.wrap(text, width=width_chars)


def draw_info_panel(frame, domain_cfg):
    """Top-left panel shown while a domain is active/fading."""
    h, w = frame.shape[:2]
    panel_w = min(620, int(w * 0.42))
    lines = _wrap_text(domain_cfg["description"], 46)
    panel_h = 110 + 22 * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    accent = domain_cfg["color_primary"]
    cv2.rectangle(frame, (0, 0), (panel_w, 6), accent, -1)

    y = 34
    cv2.putText(frame, f"Domain Expansion: {domain_cfg['domain_name']}",
                (16, y), FONT, 0.78, domain_cfg["title_color"], 2, cv2.LINE_AA)
    y += 32
    cv2.putText(frame, f"User: {domain_cfg['user_name']}",
                (16, y), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 30
    for line in lines:
        cv2.putText(frame, line, (16, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    return frame


def draw_status_text(frame, anchor_x, anchor_y, finger_count):
    """Small status line placed just above the camera inset, showing what
    the app currently reads off your hand(s)."""
    if finger_count and 1 <= finger_count <= 6:
        text = f"Fingers detected: {finger_count} - hold steady..."
        color = (0, 255, 0)
    elif finger_count and finger_count > 6:
        text = f"Fingers detected: {finger_count} (only 1-6 cast a domain)"
        color = (0, 165, 255)
    else:
        text = "Show a number (1-6) with your hand(s) to cast a domain"
        color = (200, 200, 200)

    y = max(24, anchor_y - 14)
    cv2.putText(frame, text, (anchor_x, y), FONT, 0.6, color, 2, cv2.LINE_AA)
    return frame


def draw_instructions_panel(frame, domains_cfg, number_to_domain):
    """Always-visible, text-only legend, bottom-right: which number casts
    which domain."""
    h, w = frame.shape[:2]
    entries = []
    for number in sorted(number_to_domain.keys()):
        d = domains_cfg[number_to_domain[number]]
        entries.append(f"{number}  -  {d['user_name']}: {d['domain_name']}")

    line_h = 26
    panel_w = 430
    panel_h = 40 + line_h * len(entries)
    x0 = w - panel_w
    y0 = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w, h), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "Sign Reference (fingers -> domain)", (x0 + 14, y0 + 26),
                FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    y = y0 + 26 + line_h
    for line in entries:
        cv2.putText(frame, line, (x0 + 14, y), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y += line_h

    return frame
