# Domain Expansion — Finger-Count Recognizer

Cast Jujutsu Kaisen Domain Expansions live from your webcam by holding
up a number with your hand(s).

**Domains & their numbers:**
1. Satoru Gojo — Infinite Void (blue)
2. Ryomen Sukuna — Malevolent Shrine (red)
3. Megumi Fushiguro — Chimera Shadow Garden (deep blue-black + white)
4. Mahito — Self-Embodiment of Perfection (black + purple)
5. Jogo — Coffin of the Iron Mountain (red + orange)
6. Yuta Okkotsu — Authentic Mutual Love (pink + white)

No training, no data collection — finger counting is done with plain
geometry on MediaPipe's hand landmarks, so it works the moment you run it.

## How it works

Hold up a number of fingers (across one or both hands — e.g. show 6 as
5 fingers on one hand + 1 on the other) and hold it steady for about
0.6 seconds. That triggers the matching domain, rendered as a real
particle field (~10,000 small points):
- the ambient particle field spirals inward, spinning, and collapses
  into a small glowing core in the domain's colors
- the core bursts outward into that domain's actual shape - Sukuna's
  shrine gets a bone-mountain floor, four dark pillars and a curved
  glowing roof; Gojo's void gets a bright event-horizon ring inside a
  deep shell of blue cosmic debris and gold dust; Yuta's is a shrine
  built the same way but in his pink/white palette, ringed by 22
  katana planted upright around its perimeter; Jogo's gets a rising
  volcanic cone with molten ground and drifting embers; and so on for
  all six domains (see `src/particles3d.py` for each domain's exact
  point-cloud shape)
- **top-left:** domain name, user, description
- **bottom-right:** an always-visible legend of which number casts which domain
- your live camera stays visible the whole time in a small inset,
  bottom-left, so the effect never blocks your view of yourself

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Needs a working webcam. No GPU required.

The first time you run it, it auto-downloads the official MediaPipe
hand-landmark model (~6MB) into `models/hand_landmarker.task` and reuses
that cached copy afterward — so you need internet the very first run only.

## Run it

```bash
python src/main.py
```

Opens in fullscreen. Press **ESC** or **Q** to quit.

## Tuning

Open `config/domains.py` to adjust:
- `HOLD_FRAMES_TO_TRIGGER` — how many consecutive frames (~30fps) you
  must hold a number before it fires. Lower = snappier but more
  accidental triggers.
- `COOLDOWN_SECONDS` — minimum gap between two domain expansions.
- Domain names/descriptions/colors — edit directly in the `DOMAINS` dict.
- Which number maps to which domain — set via each domain's `"number"`
  field (derives `NUMBER_TO_DOMAIN` automatically).

Open `src/effects.py` to adjust animation timing (`CONVERGE_DURATION`,
`EXPLODE_DURATION`, `ACTIVE_DURATION`, `FADE_DURATION`) or each domain's
idle rotation while its shape is held (`_ACTIVE_SPIN`).

Open `config/domains.py`'s `PARTICLE_COUNT` to trade density for
performance - 10,000 runs comfortably in real time on a normal laptop
CPU alongside the camera + hand tracking; drop it (e.g. to 5,000-6,000)
if the window feels laggy on your machine, or raise it if you have
headroom to spare for an even denser field.

Open `src/particles3d.py` to redesign any domain's actual shape - each
domain is one function (`shape_gojo`, `shape_sukuna`, etc.) returning
per-particle 3D position, BGR color, and size arrays; mix in more/fewer
particles for a given part of the shape (ground, pillars, ring, embers,
...) by changing the fractional splits at the top of each function.

Open `src/main.py` to adjust the camera inset size/position
(`inset_w`, `margin`, `inset_x0`/`inset_y0` near the top of `main()`).

## Improving finger detection

The counting logic in `src/gesture.py` uses simple, fast heuristics:
each of the four fingers is "extended" if its tip sits clearly above its
own middle knuckle; the thumb is "extended" if its tip sits much farther
from the base of the pinky than the thumb's own base joint does. This
works well for a hand held up facing the camera, which is the natural
way to show a number. If you find a particular number misreads
consistently for you, open `src/gesture.py` — `count_single_hand()` is
short and easy to nudge (e.g. loosen or tighten the `1.15` thumb ratio,
or the `0.02` finger-tip/pip margin).

## Project structure

```
config/domains.py       domain names/colors/descriptions/number mapping, tunable constants
src/hand_utils.py        MediaPipe hand-landmark wrapper (tracking + drawing)
src/gesture.py           rule-based finger counting (no ML, no training)
src/particles3d.py       per-domain 3D particle shapes + perspective projection
src/effects.py           converge -> explode -> active -> fade animation, particle rendering
src/ui.py                info panel + status text + instructions legend
src/main.py              fullscreen app: camera inset, canvas compositing, quit handling
models/                  auto-downloaded hand_landmarker.task cache (.task file is gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
