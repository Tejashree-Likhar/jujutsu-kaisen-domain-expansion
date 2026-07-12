"""
Particle animation for a Domain Expansion "cast":

  Phase 1 - CONVERGE : particles spawn at random positions around the
            screen edges and spiral inward toward the center, all
            rotating the same direction (galaxy-like).
  Phase 2 - EXPLODE   : once converged, particles burst outward fast and
            the screen flashes the domain color.
  Phase 3 - ACTIVE     : a colored vignette / tint representing the
            domain lingers over the feed while the info panel shows
            name/user/description.
  Phase 4 - FADE       : tint fades back out to the plain camera feed.

Usage:
    fx = DomainEffect(width, height, domain_cfg)
    fx.start()
    ...
    frame = fx.update_and_draw(frame)   # call every loop iteration
    if fx.finished(): ...
"""

import time
import math
import random
import numpy as np
import cv2

CONVERGE_DURATION = 1.4
EXPLODE_DURATION = 0.5
ACTIVE_DURATION = 6.0
FADE_DURATION = 1.0

N_PARTICLES = 140


class _Particle:
    __slots__ = ("angle", "radius", "start_radius", "speed", "color", "size")

    def __init__(self, cx, cy, max_radius, colors):
        self.angle = random.uniform(0, 2 * math.pi)
        self.start_radius = random.uniform(max_radius * 0.5, max_radius)
        self.radius = self.start_radius
        self.speed = random.uniform(0.8, 1.6)
        self.color = random.choice(colors)
        self.size = random.randint(2, 4)


class DomainEffect:
    def __init__(self, width, height, domain_cfg):
        self.w, self.h = width, height
        self.cfg = domain_cfg
        self.colors = [domain_cfg["color_primary"], domain_cfg["color_secondary"]]
        self.cx, self.cy = width // 2, height // 2
        self.max_radius = math.hypot(width, height) * 0.55
        self.particles = [
            _Particle(self.cx, self.cy, self.max_radius, self.colors)
            for _ in range(N_PARTICLES)
        ]
        self.phase = "idle"
        self.phase_start = 0.0
        self.rotation_dir = 1  # spin direction, consistent through converge

    def start(self):
        self.phase = "converge"
        self.phase_start = time.time()
        for p in self.particles:
            p.radius = p.start_radius

    def finished(self):
        return self.phase == "idle"

    def is_active_phase(self):
        return self.phase == "active"

    def _phase_elapsed(self):
        return time.time() - self.phase_start

    def _advance_phase_if_needed(self):
        t = self._phase_elapsed()
        if self.phase == "converge" and t >= CONVERGE_DURATION:
            self.phase = "explode"
            self.phase_start = time.time()
        elif self.phase == "explode" and t >= EXPLODE_DURATION:
            self.phase = "active"
            self.phase_start = time.time()
        elif self.phase == "active" and t >= ACTIVE_DURATION:
            self.phase = "fade"
            self.phase_start = time.time()
        elif self.phase == "fade" and t >= FADE_DURATION:
            self.phase = "idle"

    def update_and_draw(self, frame):
        if self.phase == "idle":
            return frame

        self._advance_phase_if_needed()
        t = self._phase_elapsed()
        overlay = np.zeros_like(frame)

        if self.phase == "converge":
            progress = min(t / CONVERGE_DURATION, 1.0)
            spin = progress * 6.0 * self.rotation_dir  # radians of extra spin
            for p in self.particles:
                r = p.start_radius * (1 - progress) ** 1.5
                ang = p.angle + spin * p.speed
                x = int(self.cx + r * math.cos(ang))
                y = int(self.cy + r * math.sin(ang) * 0.6)  # slight ellipse for style
                cv2.circle(overlay, (x, y), p.size, p.color, -1)
            # glowing core building up
            core_r = int(10 + 40 * progress)
            cv2.circle(overlay, (self.cx, self.cy), core_r, self.colors[0], -1)
            frame = cv2.addWeighted(frame, 1.0, overlay, min(0.9, progress + 0.2), 0)

        elif self.phase == "explode":
            progress = min(t / EXPLODE_DURATION, 1.0)
            for p in self.particles:
                r = p.size + progress * self.max_radius * 1.3
                x = int(self.cx + r * math.cos(p.angle))
                y = int(self.cy + r * math.sin(p.angle))
                cv2.circle(overlay, (x, y), max(1, int(p.size * (1 - progress) + 2)),
                           p.color, -1)
            frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
            # white/color flash fading out across the explosion
            flash_alpha = (1 - progress) * 0.6
            flash = np.full_like(frame, 255)
            frame = cv2.addWeighted(frame, 1 - flash_alpha, flash, flash_alpha, 0)

        elif self.phase == "active":
            tint = np.full_like(frame, self.cfg["bg_tint"], dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.72, tint, 0.28, 0)
            self._vignette(frame)

        elif self.phase == "fade":
            progress = min(t / FADE_DURATION, 1.0)
            alpha = 0.28 * (1 - progress)
            tint = np.full_like(frame, self.cfg["bg_tint"], dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1 - alpha, tint, alpha, 0)

        return frame

    def _vignette(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.7), int(h * 0.7)),
                    0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (99, 99), 0)
        mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        dark = (frame.astype(np.float32) * 0.55).astype(np.uint8)
        frame[:] = (frame.astype(np.float32) * mask3 +
                    dark.astype(np.float32) * (1 - mask3)).astype(np.uint8)
