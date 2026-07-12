"""
Rule-based finger counting from MediaPipe hand landmarks. No training,
no model - just geometry.

Numbers 1-6 map to the 6 domains (see config/domains.py NUMBER_TO_DOMAIN).
Since one hand can only show 0-5, a count of 6 is meant to be shown as
5 fingers on one hand + 1 on the other (or any other split that sums to
6) - fingers are simply summed across every hand MediaPipe detects.
"""

import math

# Landmark indices (MediaPipe hand model)
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
PINKY_MCP = 17
FINGER_TIPS = (8, 12, 16, 20)   # index, middle, ring, pinky
FINGER_PIPS = (6, 10, 14, 18)


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def count_single_hand(landmarks):
    """landmarks: list of 21 MediaPipe landmarks (with .x/.y in [0,1]).
    Returns an int 0-5."""
    count = 0

    # Thumb: orientation-independent heuristic. A curled thumb tip sits
    # close to the palm (near the pinky's base); an extended thumb sits
    # much farther away than its own base joint does.
    tip, mcp, pinky_mcp = landmarks[THUMB_TIP], landmarks[THUMB_MCP], landmarks[PINKY_MCP]
    if _dist(tip, pinky_mcp) > _dist(mcp, pinky_mcp) * 1.15:
        count += 1

    # Other four fingers: extended if the tip sits clearly above (smaller
    # y = higher on screen) its own pip joint - works for an upright,
    # camera-facing hand, which is the natural way to "show a number".
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02:
            count += 1

    return count


def total_fingers(hand_landmarks_list):
    """hand_landmarks_list: results.hand_landmarks from HandTracker.process()
    (a list with one entry per detected hand). Returns the summed finger
    count across all detected hands (0-10 in theory, but we only care
    about 1-6)."""
    if not hand_landmarks_list:
        return 0
    return sum(count_single_hand(hand) for hand in hand_landmarks_list)
