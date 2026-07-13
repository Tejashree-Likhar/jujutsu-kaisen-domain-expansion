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


def _infinity(n, scale, thickness, rng):
    """Points along a lemniscate-of-Bernoulli (infinity sign), with a
    tube of `thickness` around the curve so it reads as a solid loop
    rather than a thin wire."""
    t = rng.random(n) * 2 * np.pi
    denom = 1 + np.sin(t) ** 2
    x = scale * np.cos(t) / denom
    y = scale * np.sin(t) * np.cos(t) / denom
    x += rng.normal(scale=thickness * 0.4, size=n)
    y += rng.normal(scale=thickness * 0.4, size=n)
    z = rng.normal(scale=thickness * 0.4, size=n)
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


def _capsule(n, p0, p1, radius, rng):
    """n points scattered along the segment p0->p1 with gaussian jitter -
    a cheap stand-in for a "limb" in a stylized point-cloud figure."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    t = rng.random(n)
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    base = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    jitter = rng.normal(scale=radius, size=(n, 3)).astype(np.float32)
    return base + jitter


def _human_figure(n, rng, origin=(0.0, 0.0, 0.0), scale=1.0, reach_up=True):
    """A very stylized standing humanoid point-cloud: two legs, torso, head,
    two arms (reaching up and outward if reach_up, otherwise hanging)."""
    ox, oy, oz = origin

    def pt(x, y, z):
        return (ox + x * scale, oy + y * scale, oz + z * scale)

    n_leg = int(n * 0.12)
    n_torso = int(n * 0.20)
    n_head = int(n * 0.12)
    n_arm = (n - 2 * n_leg - n_torso - n_head) // 2

    left_leg = _capsule(n_leg, pt(-5, -8, 0), pt(-5, -28, 0), 1.6 * scale, rng)
    right_leg = _capsule(n_leg, pt(5, -8, 0), pt(5, -28, 0), 1.6 * scale, rng)
    torso = _capsule(n_torso, pt(0, -8, 0), pt(0, 10, 0), 3.2 * scale, rng)
    head = _sphere_shell(n_head, 0, 4.2 * scale, rng) + np.array(pt(0, 16, 0), dtype=np.float32)

    if reach_up:
        left_arm = _capsule(n_arm, pt(-6, 9, 0), pt(-20, 26, 5), 1.4 * scale, rng)
        right_arm = _capsule(n_arm, pt(6, 9, 0), pt(20, 26, 5), 1.4 * scale, rng)
    else:
        left_arm = _capsule(n_arm, pt(-6, 9, 0), pt(-9, -6, 0), 1.4 * scale, rng)
        right_arm = _capsule(n_arm, pt(6, 9, 0), pt(9, -6, 0), 1.4 * scale, rng)

    return np.concatenate([left_leg, right_leg, torso, head, left_arm, right_arm])


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
    """Unlimited Void: bright event-horizon infinity loop + a deep cosmic
    shell of blue debris flecked with golden galactic dust."""
    rng = np.random.default_rng(seed)
    n_ring = int(n * 0.12)
    n_dust_gold = int(n * 0.25)
    n_rest = n - n_ring - n_dust_gold

    ring_pos = _infinity(n_ring, scale=30, thickness=4, rng=rng)
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
    """Chimera Shadow Garden (green edition): a dense skyline of tall
    buildings in green-lit tones, with no figure among them."""
    rng = np.random.default_rng(seed)
    n_buildings_total = int(n * 0.78)
    n_ambient = n - n_buildings_total

    n_buildings = 13
    per_building = n_buildings_total // n_buildings
    centers = np.linspace(-75, 75, n_buildings)
    b_pos_chunks, b_col_chunks = [], []
    for i, cx in enumerate(centers):
        cx = cx + rng.uniform(-4, 4)
        width = rng.uniform(8, 15)
        height = rng.uniform(22, 82)
        depth_center = rng.uniform(-5, 50)  # some buildings recede further back
        x = cx + (rng.random(per_building) - 0.5) * width
        y = -30 + rng.random(per_building) * height
        z = depth_center + (rng.random(per_building) - 0.5) * 6
        b_pos_chunks.append(np.stack([x, y, z], axis=1))

        base_green = np.array([15, 60, 25], dtype=np.float32)  # dark building silhouette
        col = np.tile(base_green, (per_building, 1))
        window_mask = rng.random(per_building) < 0.07
        col[window_mask] = np.array([120, 255, 160], dtype=np.float32)  # lit windows
        b_col_chunks.append(col)

    building_pos = np.concatenate(b_pos_chunks)
    building_col = np.concatenate(b_col_chunks)
    building_size = np.full(building_pos.shape[0], 0.65, dtype=np.float32)

    ambient_pos = _sphere_shell(n_ambient, 30, 100, rng)
    ambient_col = np.tile(np.array([40, 140, 60], dtype=np.float32), (n_ambient, 1))  # green haze
    ambient_size = np.full(n_ambient, 0.45, dtype=np.float32)

    pos = np.concatenate([building_pos, ambient_pos])
    col = np.concatenate([building_col, ambient_col])
    size = np.concatenate([building_size, ambient_size])
    return pos, col, size


def shape_mahito(n, seed=4):
    """Self-Embodiment of Perfection: a spherical cage woven from
    latitude/longitude netting, glowing node-spheres at every crossing,
    a figure standing at its center, hovering over a cracked floor.
    Tilted so we're looking down into the dome from above, like the dome
    resting in a crater."""
    rng = np.random.default_rng(seed)
    n_grid = int(n * 0.44)
    n_nodes = int(n * 0.14)
    n_ground = int(n * 0.09)
    n_human = int(n * 0.15)
    n_ambient = n - n_grid - n_nodes - n_ground - n_human

    R = 42
    cy = 4  # dome center sits a bit above the ground

    n_lat = 9   # horizontal rings
    n_lon = 16  # pole-to-pole arcs
    lat_cos = np.linspace(-0.9, 0.95, n_lat)  # cos(phi) for each ring

    # --- netting: latitude rings + longitude arcs ---
    per_lat = max(1, int(n_grid * 0.55) // n_lat)
    per_lon = max(1, int(n_grid * 0.45) // n_lon)
    chunks = []
    for cphi in lat_cos:
        phi = np.arccos(cphi)
        ring_r = R * np.sin(phi)
        y = R * np.cos(phi) + cy
        theta = rng.random(per_lat) * 2 * np.pi
        x = ring_r * np.cos(theta) + rng.normal(scale=0.4, size=per_lat)
        z = ring_r * np.sin(theta) + rng.normal(scale=0.4, size=per_lat)
        yy = np.full(per_lat, y) + rng.normal(scale=0.4, size=per_lat)
        chunks.append(np.stack([x, yy, z], axis=1))
    for i in range(n_lon):
        theta = i * (2 * np.pi / n_lon)
        phi = rng.random(per_lon) * np.pi
        x = R * np.sin(phi) * np.cos(theta) + rng.normal(scale=0.4, size=per_lon)
        z = R * np.sin(phi) * np.sin(theta) + rng.normal(scale=0.4, size=per_lon)
        y = R * np.cos(phi) + cy + rng.normal(scale=0.4, size=per_lon)
        chunks.append(np.stack([x, y, z], axis=1))

    grid_pos = np.concatenate(chunks)
    grid_col = np.tile(np.array([150, 40, 130], dtype=np.float32), (grid_pos.shape[0], 1))
    grid_size = np.full(grid_pos.shape[0], 0.5, dtype=np.float32)

    # --- bright node spheres at every lat/lon crossing ---
    node_centers = []
    for cphi in lat_cos:
        phi = np.arccos(cphi)
        ring_r = R * np.sin(phi)
        y = R * np.cos(phi) + cy
        for i in range(n_lon):
            theta = i * (2 * np.pi / n_lon)
            node_centers.append([ring_r * np.cos(theta), y, ring_r * np.sin(theta)])
    node_centers = np.array(node_centers, dtype=np.float32)
    reps = max(1, n_nodes // len(node_centers))
    node_pos = np.repeat(node_centers, reps, axis=0)
    node_pos = node_pos + rng.normal(scale=0.7, size=node_pos.shape)
    node_col = np.tile(np.array([225, 140, 255], dtype=np.float32), (node_pos.shape[0], 1))
    node_size = np.full(node_pos.shape[0], 1.7, dtype=np.float32)

    # --- cracked ground beneath the dome ---
    ground_y = cy - R - 3
    n_cracks = 14
    per_crack = max(1, n_ground // n_cracks)
    ground_chunks = []
    for _ in range(n_cracks):
        ang0 = rng.random() * 2 * np.pi
        length = rng.uniform(20, 58)
        t = rng.random(per_crack)
        r = t * length
        jitter_ang = ang0 + np.cumsum(rng.normal(scale=0.04, size=per_crack))
        x = r * np.cos(jitter_ang)
        z = r * np.sin(jitter_ang)
        y = np.full(per_crack, ground_y) + rng.normal(scale=0.3, size=per_crack)
        ground_chunks.append(np.stack([x, y, z], axis=1))
    ground_pos = np.concatenate(ground_chunks)
    ground_col = np.tile(np.array([60, 15, 55], dtype=np.float32), (ground_pos.shape[0], 1))
    ground_size = np.full(ground_pos.shape[0], 0.5, dtype=np.float32)

    # --- the figure standing at the center of the dome, on the ground ---
    human_scale = 0.9
    foot_y = ground_y + 1
    human_origin_y = foot_y + 28 * human_scale
    human_pos = _human_figure(n_human, rng, origin=(0, human_origin_y, 0), scale=human_scale, reach_up=False)
    # Warm, bright pale skin - deliberately far lighter than the dark
    # purple net/background so the figure actually pops instead of
    # blending into the mesh around it.
    human_col = np.tile(np.array([140, 165, 215], dtype=np.float32), (human_pos.shape[0], 1))
    human_size = np.full(human_pos.shape[0], 1.4, dtype=np.float32)

    ambient_pos = _sphere_shell(n_ambient, 30, 110, rng)
    ambient_col = np.tile(np.array([55, 8, 45], dtype=np.float32), (n_ambient, 1))
    ambient_size = np.full(n_ambient, 0.4, dtype=np.float32)

    pos = np.concatenate([grid_pos, node_pos, ground_pos, human_pos, ambient_pos])
    col = np.concatenate([grid_col, node_col, ground_col, human_col, ambient_col])
    size = np.concatenate([grid_size, node_size, ground_size, human_size, ambient_size])

    # Tilt so the near/top of the dome swings toward the camera and grows
    # large in frame while the far side recedes - the "looking down into
    # it from above" fisheye read from the reference image.
    pos = rotate_x(pos, np.radians(36))
    pos = pos * 1.25
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
    """Authentic Mutual Love: a ring facing the viewer at a shallow angle,
    encircled by katana standing upright, piercing through its
    circumference like a crown of blades."""
    rng = np.random.default_rng(seed)
    n_blades = int(n * 0.40)
    n_knot = int(n * 0.28)
    n_dust = n - n_blades - n_knot

    # Ring / knot motif: built facing the camera, then tilted only
    # modestly so it reads as a clean halo/oval (a steeper tilt makes the
    # near/far edges scale so unevenly under perspective that the ring
    # reads as a tall vertical sliver instead of a ring).
    TILT = np.radians(25)
    R, r_minor = 24, 6
    main_angle = rng.random(n_knot) * 2 * np.pi
    tube_angle = rng.random(n_knot) * 2 * np.pi
    knot_pos = np.stack([
        (R + r_minor * np.cos(tube_angle)) * np.cos(main_angle),
        (R + r_minor * np.cos(tube_angle)) * np.sin(main_angle),
        r_minor * np.sin(tube_angle),
    ], axis=1)
    knot_pos = rotate_x(knot_pos, TILT)
    knot_col = np.tile(np.array([170, 20, 255], dtype=np.float32), (n_knot, 1))  # vibrant pink
    knot_size = np.full(n_knot, 1.1, dtype=np.float32)

    # Katana standing upright (extending in world-Y, like stakes), each
    # anchored to a point on the ring's circumference - anchors are
    # computed by tilting the same pre-tilt angle used for the ring, so
    # the blades pierce through exactly where the ring sits. Because the
    # blades stay vertical rather than following the ring's tilt, the
    # whole thing spins cleanly around the vertical axis with no
    # crossing/"+" artifacts.
    n_swords = 26
    per_sword = max(1, n_blades // n_swords)
    blade_chunks, blade_col_chunks = [], []
    for i in range(n_swords):
        angle = i * (2 * np.pi / n_swords) + rng.uniform(-0.05, 0.05)
        anchor_r = R + rng.uniform(-2, 2)
        anchor_local = np.array([[anchor_r * np.cos(angle), anchor_r * np.sin(angle), 0.0]], dtype=np.float32)
        ax, ay, az = rotate_x(anchor_local, TILT)[0]
        length = rng.uniform(18, 28)

        t_local = rng.random(per_sword)  # 0 = tip below the ring, 1 = tip above
        y = ay + (t_local - 0.3) * length
        x = ax + rng.normal(scale=0.6, size=per_sword)
        z = az + rng.normal(scale=0.6, size=per_sword)
        blade_chunks.append(np.stack([x, y, z], axis=1))

        col = np.tile(np.array([235, 230, 255], dtype=np.float32), (per_sword, 1))  # silver-white blade
        guard_mask = (t_local > 0.26) & (t_local < 0.34)
        col[guard_mask] = np.array([210, 90, 150], dtype=np.float32)  # small pink hilt/guard accent
        blade_col_chunks.append(col)

    blade_pos = np.concatenate(blade_chunks)
    blade_col = np.concatenate(blade_col_chunks)
    blade_size = np.full(blade_pos.shape[0], 0.85, dtype=np.float32)

    dust_pos = _sphere_shell(n_dust, 15, 90, rng)
    dust_col = np.tile(np.array([200, 150, 255], dtype=np.float32), (n_dust, 1))  # soft pink sparkle
    dust_size = np.full(n_dust, 0.5, dtype=np.float32)

    pos = np.concatenate([knot_pos, blade_pos, dust_pos])
    col = np.concatenate([knot_col, blade_col, dust_col])
    size = np.concatenate([knot_size, blade_size, dust_size])
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

def rotate_x(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    y2 = y * c - z * s
    z2 = y * s + z * c
    return np.stack([x, y2, z2], axis=1)


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
