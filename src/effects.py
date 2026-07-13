"""
Domain Expansion cast animation, rebuilt around a real particle field
(see particles3d.py) instead of a flat color tint:

  Phase 1 - CONVERGE : the ambient "neutral" particle field spirals inward
             (spinning) and collapses into a small glowing core, in the
             domain's own colors.
  Phase 2 - EXPLODE   : that core bursts outward into the domain's actual
             shape (Sukuna's shrine pillars/dome, Gojo's void ring, etc).
  Phase 3 - ACTIVE     : the formed domain holds, with its own slow
             ambient rotation, while the info panel shows name/user/desc.
  Phase 4 - FADE       : the shape dims out to black.

Rendering is fully vectorized numpy: thousands of small particles are
projected to 2D and additively scattered onto a float buffer, then
lightly blurred for glow - fast enough for real-time video without a
GPU/WebGL, unlike drawing each particle with cv2.circle in a loop.
"""

import sys
import os
import time
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.domains import PARTICLE_COUNT
from src.particles3d import get_shape_fixed, rotate_z, rotate_y, project

CONVERGE_DURATION = 1.3
EXPLODE_DURATION = 0.7
ACTIVE_DURATION = 6.0
FADE_DURATION = 1.0

# Per-domain idle rotation while the shape is held (axis, radians/sec).
# Locked (0 speed) domains read as heavier / more architectural.
_ACTIVE_SPIN = {
    "gojo": ("y", 0.25),
    "sukuna": ("y", 0.0),
    "megumi": ("y", 0.15),
    "mahito": ("y", 0.12),
    "jogo": ("y", 0.0),
    "yuta": ("y", 0.30),
}


def _ease_in(t):
    return t * t


def _ease_out(t):
    return 1 - (1 - t) ** 3


def _lerp(a, b, t):
    return a + (b - a) * t


class DomainEffect:
    def __init__(self, width, height, domain_cfg):
        self.w, self.h = width, height
        self.cfg = domain_cfg
        self.key = domain_cfg["key"]
        self.n = PARTICLE_COUNT

        self.neutral_pos, self.neutral_col, self.neutral_size = get_shape_fixed("neutral", self.n)
        self.domain_pos, self.domain_col, self.domain_size = get_shape_fixed(self.key, self.n)

        rng = np.random.default_rng(7)
        theta = rng.random(self.n) * 2 * np.pi
        phi = np.arccos(2 * rng.random(self.n) - 1)
        r = rng.random(self.n) * 3.0
        self.core_pos = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ], axis=1)
        core_color = np.clip(np.array(domain_cfg["color_primary"], dtype=np.float32) * 1.8, 0, 400)
        self.core_col = np.tile(core_color, (self.n, 1))
        self.core_size = np.full(self.n, 2.6, dtype=np.float32)

        self.phase = "idle"
        self.phase_start = 0.0
        self._frozen_pos = None  # snapshot used during fade
        self.spin_axis, self.spin_speed = _ACTIVE_SPIN.get(self.key, ("y", 0.2))

    def start(self):
        self.phase = "converge"
        self.phase_start = time.time()

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
        t_raw = min(1.0, self._phase_elapsed() / self._phase_duration())

        if self.phase == "converge":
            t = _ease_in(t_raw)
            spin_angle = t_raw * 6.0  # a few fast turns while collapsing
            spun = rotate_z(self.neutral_pos, spin_angle)
            pos = _lerp(spun, self.core_pos, t)
            col = _lerp(self.neutral_col, self.core_col, t)
            size = _lerp(self.neutral_size, self.core_size, t)

        elif self.phase == "explode":
            t = _ease_out(t_raw)
            pos = _lerp(self.core_pos, self.domain_pos, t)
            col = _lerp(self.core_col, self.domain_col, t)
            size = _lerp(self.core_size, self.domain_size, t)
            decaying_spin = (1 - t_raw) * 4.0
            pos = rotate_z(pos, decaying_spin)
            self._frozen_pos = pos

        elif self.phase == "active":
            angle = self._phase_elapsed() * self.spin_speed
            rot = rotate_y if self.spin_axis == "y" else rotate_z
            pos = rot(self.domain_pos, angle)
            col = self.domain_col
            size = self.domain_size
            self._frozen_pos = pos

        else:  # fade
            fade_amount = 1 - t_raw
            pos = self._frozen_pos if self._frozen_pos is not None else self.domain_pos
            col = self.domain_col * fade_amount
            size = self.domain_size

        return self._render(frame, pos, col, size)

    def _phase_duration(self):
        return {
            "converge": CONVERGE_DURATION,
            "explode": EXPLODE_DURATION,
            "active": ACTIVE_DURATION,
            "fade": FADE_DURATION,
        }.get(self.phase, 1.0)

    def _render(self, frame, pos3d, colors, sizes):
        xy, scale = project(pos3d, self.w, self.h)
        x, y = xy[:, 0], xy[:, 1]
        valid = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
        )
        if not np.any(valid):
            return frame

        xi = x[valid].astype(np.int32)
        yi = y[valid].astype(np.int32)
        c = colors[valid]
        depth_weight = np.clip(scale[valid] / 8.0, 0.35, 2.2)
        s = sizes[valid] * depth_weight
        weighted = c * s[:, None]

        accum = np.zeros((self.h, self.w, 3), dtype=np.float32)
        # Additive "plus" splat instead of a Gaussian blur: a real blur
        # normalizes/averages, which quietly halves a lone lit pixel's
        # peak brightness. Adding to a couple of neighbors instead keeps
        # each particle a small, bright, crisp dot rather than a dim smear.
        np.add.at(accum, (yi, xi), weighted)
        for dy, dx, wgt in ((-1, 0, 0.35), (1, 0, 0.35), (0, -1, 0.35), (0, 1, 0.35)):
            yy = np.clip(yi + dy, 0, self.h - 1)
            xx = np.clip(xi + dx, 0, self.w - 1)
            np.add.at(accum, (yy, xx), weighted * wgt)

        # Soft ambient bloom/glow, computed at low resolution (cheap) and
        # blended back in lightly so clusters get an atmospheric halo
        # without washing out the crisp particle cores above.
        small = cv2.resize(accum, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        bloom_small = cv2.GaussianBlur(small, (9, 9), 0)
        bloom = cv2.resize(bloom_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        accum = accum + bloom * 0.6
        accum = np.clip(accum, 0, 255).astype(np.uint8)

        return cv2.add(frame, accum)
