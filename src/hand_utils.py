"""
Wraps MediaPipe's HandLandmarker (new Tasks API - mediapipe >= 0.10.30
removed the old `mp.solutions.hands` interface) and turns raw landmarks
into a fixed-length, position/scale-invariant feature vector suitable
for a classifier.

On first use this downloads the official hand_landmarker.task model
bundle (a few MB) from Google's model storage into models/, then reuses
the cached copy on every later run.

Feature vector layout (126 floats):
  [0:63]   -> "left"  hand, 21 landmarks * (x, y, z), zeros if absent
  [63:126] -> "right" hand, 21 landmarks * (x, y, z), zeros if absent

Normalization per hand:
  1. Translate so the wrist (landmark 0) sits at the origin.
  2. Scale so the distance from wrist -> middle-finger MCP (landmark 9)
     equals 1.0. This removes dependence on how close your hand is to
     the camera and where it sits in the frame.
"""

import os
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp

NUM_LANDMARKS = 21
VECTOR_LEN = NUM_LANDMARKS * 3 * 2  # two hands

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_ROOT, "models", "hand_landmarker.task")

_BaseOptions = mp.tasks.BaseOptions
_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode
_HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def _ensure_model():
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    print("Downloading hand-landmark model (one-time, ~6MB)...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"Saved model -> {_MODEL_PATH}")
    except Exception as e:
        raise SystemExit(
            "Could not auto-download the MediaPipe hand model.\n"
            f"Reason: {e}\n"
            f"Please download it manually from:\n  {_MODEL_URL}\n"
            f"and save it to:\n  {_MODEL_PATH}"
        )


class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.6, tracking_conf=0.6):
        _ensure_model()
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def _next_timestamp_ms(self):
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process(self, frame_bgr):
        """frame_bgr: OpenCV BGR image. Returns a HandLandmarkerResult."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = self._next_timestamp_ms()
        return self._landmarker.detect_for_video(mp_image, ts_ms)

    def draw(self, frame_bgr, results):
        if not results.hand_landmarks:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        for hand_lms in results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for conn in _HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
            for x, y in pts:
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 255), -1)
        return frame_bgr

    @staticmethod
    def _normalize_single_hand(landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        wrist = pts[0].copy()
        pts -= wrist
        scale = np.linalg.norm(pts[9]) or 1e-6
        pts /= scale
        return pts.flatten()

    def feature_vector(self, results):
        """Returns a (126,) float32 vector, zero-padded for missing hands.
        Also returns hands_present (0, 1, or 2) for gating logic."""
        vec = np.zeros(VECTOR_LEN, dtype=np.float32)
        hands_present = 0

        if not results.hand_landmarks or not results.handedness:
            return vec, hands_present

        for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            normalized = self._normalize_single_hand(hand_lms)
            hands_present += 1
            if label == "Left":
                vec[0:63] = normalized
            else:
                vec[63:126] = normalized

        return vec, hands_present

    def close(self):
        self._landmarker.close()
