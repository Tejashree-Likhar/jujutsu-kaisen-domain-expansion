"""
Vectorized "3D" particle shapes for each domain, plus a lightweight
perspective projector. This ports the *approach* used by the reference
three.js project (github.com/awnish9002/jjk): every domain is a big
point cloud where each particle has a target (x, y, z, b, g, r, size),
and the whole field smoothly moves from one shape to another. Instead of
WebGL + shaders, we do it with plain numpy: thousands of points, a
simple perspective projection to 2D, and an additive glow render (scatter
onto a float buffer + blur) which is fast enough for real-time video in
pure Python/OpenCV.

Coordinate convention (matches the reference): particles live in a
volume roughly x,y,z in [-100, 100], camera sits back on +z looking at
the origin. Colors are BGR floats that are intentionally allowed to
exceed 255 for "hot" particles (ring cores, embers, etc.) - the additive
render clips them to white-hot on overlap, which is what gives the glow
look without a real bloom shader.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Random helpers
# ---------------------------------------------------------------------------

def _sphere_shell(n, r_min, r_max, rng):
    """n random points on/within a spherical shell of radii [r_min, r_max]."""
    r = r_min + (r_max - r_min) * rng.random(n)
    theta = rng.random(n) * 2 * np.pi
    phi = np.arccos(2 * rng.random(n) - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _ring(n, radius, thickness, rng):
    theta = rng.random(n) * 2 * np.pi
    r = radius + (rng.random(n) - 0.5) * thickness
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = (rng.random(n) - 0.5) * thickness
    return np.stack([x, y, z], axis=1)


def _ground_plane(n, half_size, y_level, rng, y_jitter=1.5):
    x = (rng.random(n) - 0.5) * 2 * half_size
    z = (rng.random(n) - 0.5) * 2 * half_size
    y = y_level + (rng.random(n) - 0.5) * y_jitter
    return np.stack([x, y, z], axis=1)


def _spiral_arms(n, arms, max_radius, rise, rng, tightness=12.0):
    t = rng.random(n)
    arm_idx = rng.integers(0, arms, n)
    angle = t * tightness + arm_idx * (2 * np.pi / arms)
    radius = 2 + t * max_radius
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = (rng.random(n) - 0.5) * rise * t
    return np.stack([x, y, z], axis=1)


def _lerp_color(c1, c2, t):
    return c1 + (np.array(c2, dtype=np.float32) - np.array(c1, dtype=np.float32)) * t[:, None]


# ---------------------------------------------------------------------------
# Domain shapes. Each returns (positions (n,3), colors (n,3) BGR float,
# sizes (n,) float) for exactly `n` particles. A fixed `seed` keeps each
# domain's shape stable frame to frame (only interpolation progress moves).
# ---------------------------------------------------------------------------

def shape_neutral(n, seed=0):
    """Idle state: a faint, sparse halo - matches the reference's dim
    'neutral' cloud so the screen isn't just empty black between casts."""
    rng = np.random.default_rng(seed)
    pos = _sphere_shell(n, 15, 40, rng)
    colors = np.tile(np.array([60, 40, 20], dtype=np.float32), (n, 1))  # dim cool white-blue (BGR)
    sizes = np.full(n, 0.5, dtype=np.float32)
    # only ~8% actually visible, rest parked at the origin (invisible/dark)
    visible = rng.random(n) < 0.08
    pos[~visible] = 0
    colors[~visible] = 0
    return pos, colors, sizes


def shape_gojo(n, seed=1):
    """Unlimited Void: bright event-horizon ring + a deep cosmic shell of
    blue debris flecked with golden galactic dust."""
    rng = np.random.default_rng(seed)
    n_ring = int(n * 0.12)
    n_dust_gold = int(n * 0.25)
    n_rest = n - n_ring - n_dust_gold

    ring_pos = _ring(n_ring, radius=26, thickness=3, rng=rng)
    ring_col = np.tile(np.array([180, 230, 255], dtype=np.float32), (n_ring, 1))  # bright gold-white
    ring_size = np.full(n_ring, 2.2, dtype=np.float32)

    gold_pos = _sphere_shell(n_dust_gold, 20, 95, rng)
    gold_col = np.tile(np.array([40, 190, 255], dtype=np.float32), (n_dust_gold, 1))  # golden dust (BGR)
    gold_size = np.full(n_dust_gold, 0.8, dtype=np.float32)

    blue_pos = _sphere_shell(n_rest, 20, 100, rng)
    blue_col = np.tile(np.array([230, 120, 30], dtype=np.float32), (n_rest, 1))  # cosmic blue (BGR)
    blue_size = np.full(n_rest, 0.7, dtype=np.float32)

    pos = np.concatenate([ring_pos, gold_pos, blue_pos])
    col = np.concatenate([ring_col, gold_col, blue_col])
    size = np.concatenate([ring_size, gold_size, blue_size])
    return pos, col, size


def shape_sukuna(n, seed=2):
    """Malevolent Shrine: a mountain of bones underfoot, four dark pillars,
    a curved bone-red roof overhead, and drifting cursed-red haze."""
    rng = np.random.default_rng(seed)
    n_ground = int(n * 0.30)
    n_pillars = int(n * 0.14)
    n_roof = int(n * 0.20)
    n_haze = n - n_ground - n_pillars - n_roof

    ground_pos = _ground_plane(n_ground, half_size=55, y_level=-20, rng=rng, y_jitter=3)
    ground_col = np.tile(np.array([70, 90, 140], dtype=np.float32), (n_ground, 1))  # pale bone/red-tan
    ground_size = np.full(n_ground, 0.8, dtype=np.float32)

    # 4 pillar footprints (like a torii gate's support beams)
    corner_choice = rng.integers(0, 4, n_pillars)
    px = np.where(corner_choice % 2 == 0, 14.0, -14.0)
    pz = np.where(corner_choice < 2, 9.0, -9.0)
    pillar_pos = np.stack([
        px + (rng.random(n_pillars) - 0.5) * 2,
        -20 + rng.random(n_pillars) * 38,
        pz + (rng.random(n_pillars) - 0.5) * 2,
    ], axis=1)
    pillar_col = np.tile(np.array([45, 30, 120], dtype=np.float32), (n_pillars, 1))  # dark red pillar, rim-lit
    pillar_size = np.full(n_pillars, 1.3, dtype=np.float32)

    t = rng.random(n_roof) * 2 * np.pi
    rad = rng.random(n_roof) * 32
    curve = (rad / 32) ** 2 * 12
    roof_pos = np.stack([
        rad * np.cos(t),
        18 - curve + rng.random(n_roof) * 2,
        rad * np.sin(t) * 0.6,
    ], axis=1)
    roof_col = np.tile(np.array([35, 30, 210], dtype=np.float32), (n_roof, 1))  # glowing dark red
    roof_size = np.full(n_roof, 1.1, dtype=np.float32)

    haze_pos = _sphere_shell(n_haze, 10, 70, rng)
    haze_col = np.tile(np.array([15, 10, 90], dtype=np.float32), (n_haze, 1))
    haze_size = np.full(n_haze, 0.45, dtype=np.float32)

    pos = np.concatenate([ground_pos, pillar_pos, roof_pos, haze_pos])
    col = np.concatenate([ground_col, pillar_col, roof_col, haze_col])
    size = np.concatenate([ground_size, pillar_size, roof_size, haze_size])
    return pos, col, size


def shape_megumi(n, seed=3):
    """Chimera Shadow Garden: a vast dark, fluid abyss with sparse stark
    white highlights and reaching shadow tendrils."""
    rng = np.random.default_rng(seed)
    n_void = int(n * 0.55)
    n_highlight = int(n * 0.15)
    n_tendrils = n - n_void - n_highlight

    void_pos = _sphere_shell(n_void, 10, 100, rng)
    void_col = np.tile(np.array([35, 15, 5], dtype=np.float32), (n_void, 1))  # near-black deep blue
    void_size = np.full(n_void, 0.55, dtype=np.float32)

    hi_pos = _sphere_shell(n_highlight, 15, 55, rng)
    hi_col = np.tile(np.array([255, 250, 245], dtype=np.float32), (n_highlight, 1))  # stark white
    hi_size = np.full(n_highlight, 1.4, dtype=np.float32)

    tendril_pos = _spiral_arms(n_tendrils, arms=5, max_radius=85, rise=70, rng=rng, tightness=9.0)
    tendril_col = np.tile(np.array([90, 35, 10], dtype=np.float32), (n_tendrils, 1))  # dark blue tendrils
    tendril_size = np.full(n_tendrils, 0.6, dtype=np.float32)

    pos = np.concatenate([void_pos, hi_pos, tendril_pos])
    col = np.concatenate([void_col, hi_col, tendril_col])
    size = np.concatenate([void_size, hi_size, tendril_size])
    return pos, col, size


def shape_mahito(n, seed=4):
    """Self-Embodiment of Perfection: a writhing, knotted mass of many
    interlocking spiral 'arms' in black and violet, dense at the core."""
    rng = np.random.default_rng(seed)
    n_core = int(n * 0.12)
    n_arms = n - n_core

    core_pos = _sphere_shell(n_core, 0, 9, rng)
    core_col = np.tile(np.array([120, 10, 90], dtype=np.float32), (n_core, 1))  # hot violet core
    core_size = np.full(n_core, 2.0, dtype=np.float32)

    arm_pos = _spiral_arms(n_arms, arms=7, max_radius=55, rise=50, rng=rng, tightness=16.0)
    arm_col = np.tile(np.array([100, 15, 90], dtype=np.float32), (n_arms, 1))  # deep purple-black
    arm_size = np.full(n_arms, 0.85, dtype=np.float32)

    pos = np.concatenate([core_pos, arm_pos])
    col = np.concatenate([core_col, arm_col])
    size = np.concatenate([core_size, arm_size])
    return pos, col, size


def shape_jogo(n, seed=5):
    """Coffin of the Iron Mountain: a rising volcanic cone of jagged rock,
    a molten ground, and embers drifting upward through the heat."""
    rng = np.random.default_rng(seed)
    n_ground = int(n * 0.28)
    n_cone = int(n * 0.35)
    n_embers = n - n_ground - n_cone

    ground_pos = _ground_plane(n_ground, half_size=50, y_level=-22, rng=rng, y_jitter=2)
    ground_col = np.tile(np.array([20, 60, 230], dtype=np.float32), (n_ground, 1))  # molten orange-red
    ground_size = np.full(n_ground, 0.8, dtype=np.float32)

    h = rng.random(n_cone)                       # 0 (base) -> 1 (peak)
    radius = 45 * (1 - h) + 2 * rng.random(n_cone)
    theta = rng.random(n_cone) * 2 * np.pi
    cone_pos = np.stack([
        radius * np.cos(theta),
        -22 + h * 60,
        radius * np.sin(theta),
    ], axis=1)
    # rock gray at the base, hotter red-orange near the jagged peak
    cone_col = _lerp_color(np.array([55, 50, 55]), (10, 60, 255), h)
    cone_size = np.full(n_cone, 0.9, dtype=np.float32)

    ember_pos = _sphere_shell(n_embers, 5, 75, rng)
    ember_col = np.tile(np.array([10, 120, 255], dtype=np.float32), (n_embers, 1))  # bright ember orange
    ember_size = np.full(n_embers, 0.5, dtype=np.float32)

    pos = np.concatenate([ground_pos, cone_pos, ember_pos])
    col = np.concatenate([ground_col, cone_col, ember_col])
    size = np.concatenate([ground_size, cone_size, ember_size])
    return pos, col, size


def shape_yuta(n, seed=6):
    """Authentic Mutual Love: a ring of hovering katana blades around an
    intertwined knot, in vibrant pink and white."""
    rng = np.random.default_rng(seed)
    n_blades = int(n * 0.35)
    n_knot = int(n * 0.30)
    n_dust = n - n_blades - n_knot

    n_swords = 24
    per_sword = max(1, n_blades // n_swords)
    blade_chunks = []
    for i in range(n_swords):
        ang = i * (2 * np.pi / n_swords)
        length = rng.uniform(18, 30)
        t = rng.random(per_sword)
        r = 30 + t * length
        x = r * np.cos(ang)
        z = r * np.sin(ang)
        y = np.sin(ang * 3) * 10 + (rng.random(per_sword) - 0.5) * 2
        blade_chunks.append(np.stack([x, y, z], axis=1))
    blade_pos = np.concatenate(blade_chunks) if blade_chunks else np.zeros((0, 3))
    n_blades_actual = blade_pos.shape[0]
    blade_col = np.tile(np.array([235, 230, 255], dtype=np.float32), (n_blades_actual, 1))  # silver-white
    blade_size = np.full(n_blades_actual, 0.9, dtype=np.float32)

    # torus knot for the wedding knot motif
    t = rng.random(n_knot) * 2 * np.pi
    R, r_minor = 22, 7
    tube_angle = rng.random(n_knot) * 2 * np.pi
    knot_pos = np.stack([
        (R + r_minor * np.cos(tube_angle)) * np.cos(t * 2),
        r_minor * np.sin(tube_angle),
        (R + r_minor * np.cos(tube_angle)) * np.sin(t * 2),
    ], axis=1)
    knot_col = np.tile(np.array([170, 20, 255], dtype=np.float32), (n_knot, 1))  # vibrant pink
    knot_size = np.full(n_knot, 1.0, dtype=np.float32)

    dust_pos = _sphere_shell(n_dust, 15, 90, rng)
    dust_col = np.tile(np.array([200, 150, 255], dtype=np.float32), (n_dust, 1))  # soft pink sparkle
    dust_size = np.full(n_dust, 0.5, dtype=np.float32)

    pos = np.concatenate([blade_pos, knot_pos, dust_pos])
    col = np.concatenate([blade_col, knot_col, dust_col])
    size = np.concatenate([blade_size, knot_size, dust_size])
    return pos, col, size


SHAPE_FUNCS = {
    "neutral": shape_neutral,
    "gojo": shape_gojo,
    "sukuna": shape_sukuna,
    "megumi": shape_megumi,
    "mahito": shape_mahito,
    "jogo": shape_jogo,
    "yuta": shape_yuta,
}


def get_shape(key, n):
    return SHAPE_FUNCS[key](n)


def _fit_exact(pos, col, size, n):
    """Pad (by jittered resampling) or truncate so arrays are exactly length
    n - needed since a couple of the generators above produce an array
    length that's only approximately n (integer-division rounding), but
    interpolating between two phases requires identical array lengths."""
    cur = pos.shape[0]
    if cur == n:
        return pos, col, size
    if cur > n:
        return pos[:n], col[:n], size[:n]
    rng = np.random.default_rng(12345)
    pad_n = n - cur
    idx = rng.integers(0, cur, pad_n)
    pad_pos = pos[idx] + rng.normal(scale=0.5, size=(pad_n, 3))
    return (
        np.concatenate([pos, pad_pos]),
        np.concatenate([col, col[idx]]),
        np.concatenate([size, size[idx]]),
    )


def get_shape_fixed(key, n):
    """Like get_shape, but guarantees exactly n particles - use this
    everywhere two shapes need to be interpolated between."""
    pos, col, size = SHAPE_FUNCS[key](n)
    return _fit_exact(pos, col, size, n)


# ---------------------------------------------------------------------------
# Rotation + perspective projection
# ---------------------------------------------------------------------------

def rotate_y(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x2 = x * c + z * s
    z2 = -x * s + z * c
    return np.stack([x2, y, z2], axis=1)


def rotate_z(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x2 = x * c - y * s
    y2 = x * s + y * c
    return np.stack([x2, y2, z], axis=1)


def project(points, screen_w, screen_h, cam_dist=140):
    """Simple perspective projection. Returns (screen_xy (n,2) float,
    depth_scale (n,) float) - depth_scale > 1 means closer/bigger."""
    focal = screen_h * 0.9
    z = points[:, 2]
    denom = np.clip(cam_dist - z, 20, None)  # avoid blow-up / negative depth
    scale = focal / denom
    x2d = screen_w / 2 + points[:, 0] * scale
    y2d = screen_h / 2 - points[:, 1] * scale
    return np.stack([x2d, y2d], axis=1), scale
