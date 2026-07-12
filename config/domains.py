"""
Central configuration for all Domain Expansions.

Colors are given as (B, G, R) tuples since we render with OpenCV.
Each domain can have 1 or 2 particle colors (primary / secondary) used in
the swirl -> explode intro animation.

`key` is the internal class name used everywhere (data files, model labels,
folder names). Do not rename these once you've started collecting data,
or your dataset labels and thumbnails will fall out of sync.
"""

DOMAINS = {
    "gojo": {
        "number": 1,
        "user_name": "Satoru Gojo",
        "domain_name": "Unlimited Void",
        "description": (
            "An infinite, pitch-black expanse resembling the center of the "
            "universe, filled with drifting cosmic debris and swirling "
            "galactic dust. Victims are bombarded with infinite "
            "information, paralyzing the mind."
        ),
        "color_primary": (255, 120, 0),    # vivid blue (BGR)
        "color_secondary": (0, 215, 255),  # gold
        "bg_tint": (255, 120, 0),
    },
    "sukuna": {
        "number": 2,
        "user_name": "Ryomen Sukuna",
        "domain_name": "Malevolent Shrine",
        "description": (
            "A decrepit, twisted shrine of giant mouths, horns and "
            "buffalo skulls, resting atop a mountain of bones. Everything "
            "within is dismantled by Sukuna's dismantle and cleave."
        ),
        "color_primary": (0, 0, 255),      # red
        "color_secondary": (0, 60, 160),   # dark red / rust
        "bg_tint": (0, 0, 255),
    },
    "megumi": {
        "number": 3,
        "user_name": "Megumi Fushiguro",
        "domain_name": "Chimera Shadow Garden",
        "description": (
            "The surroundings dissolve into a massive, shadowy, fluid "
            "abyss capable of birthing endless Shikigami and swallowing "
            "opponents into bottomless darkness."
        ),
        "color_primary": (90, 20, 0),      # deep blue-black
        "color_secondary": (255, 255, 255),  # stark white highlight
        "bg_tint": (80, 10, 0),
    },
    "mahito": {
        "number": 4,
        "user_name": "Mahito",
        "domain_name": "Self-Embodiment of Perfection",
        "description": (
            "A dark, claustrophobic void of countless interconnected "
            "giant arms and hands, wrapping around the target in an "
            "inescapable net of soul-warping flesh."
        ),
        "color_primary": (60, 0, 40),      # deep black-purple
        "color_secondary": (200, 0, 160),  # purple
        "bg_tint": (80, 0, 90),
    },
    "jogo": {
        "number": 5,
        "user_name": "Jogo",
        "domain_name": "Coffin of the Iron Mountain",
        "description": (
            "The inside of a violently active volcano - jagged rock "
            "fissures and rivers of molten lava. The heat alone is "
            "enough to incinerate an unprepared sorcerer."
        ),
        "color_primary": (0, 30, 200),     # deep red
        "color_secondary": (0, 120, 255),  # orange
        "bg_tint": (0, 40, 200),
    },
    "yuta": {
        "number": 6,
        "user_name": "Yuta Okkotsu",
        "domain_name": "Authentic Mutual Love",
        "description": (
            "A surreal shrine-space filled with countless hovering "
            "katana and intertwined wedding knots, reflecting the depth "
            "of Yuta's bonds and his immense cursed energy."
        ),
        "color_primary": (180, 20, 255),   # vibrant pink
        "color_secondary": (255, 255, 255),  # white
        "bg_tint": (180, 20, 255),
    },
}

# Number of extended fingers (summed across both hands, so e.g. 6 can be
# shown as 5 on one hand + 1 on the other) -> which domain it casts.
NUMBER_TO_DOMAIN = {cfg["number"]: key for key, cfg in DOMAINS.items()}

# Tunable behavior
HOLD_FRAMES_TO_TRIGGER = 18      # consecutive frames showing the same number (~0.6s @30fps)
COOLDOWN_SECONDS = 3.0           # min gap between two domain expansions
DOMAIN_DISPLAY_SECONDS = 6.0     # how long the domain overlay stays up (see effects.ACTIVE_DURATION)
