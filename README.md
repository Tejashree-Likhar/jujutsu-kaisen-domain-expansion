# Domain Expansion — Hand Sign Recognizer

Cast Jujutsu Kaisen Domain Expansions with your own hand signs, recognized
live from your webcam.

**Domains included:** Gojo (Unlimited Void), Sukuna (Malevolent Shrine),
Megumi (Chimera Shadow Garden), Mahito (Self-Embodiment of Perfection),
Jogo (Coffin of the Iron Mountain), Yuta (Authentic Mutual Love).

## How it works

1. **Collect data** — show the camera your own hand pose for each domain
   (whatever gesture you want to represent it — these are *your* signs).
2. **Train** — a classifier learns to tell your 6 poses (+ neutral/idle)
   apart, from MediaPipe hand-landmarks.
3. **Run** — hold a trained pose for ~0.6s to trigger that domain: a
   particle swirl converges and spins in the domain's colors, then
   explodes into a colored overlay, showing:
   - **top-left:** domain name, user name, description
   - **bottom-right:** an always-visible reference gallery of every
     trained sign (thumbnail auto-captured during data collection)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Needs a working webcam and a machine that can run MediaPipe (standard
laptop CPU is fine, no GPU required).

The first time you run `data_collection.py` or `main.py`, it will
auto-download the official hand-landmark model file (~6MB) from Google
into `models/hand_landmarker.task` and reuse it after that — so you need
an internet connection the very first time, but not after.

## 1. Collect training data

```bash
python src/data_collection.py
```

This walks you through all 7 classes in order (6 domains + `neutral`).
For each:
- Get into position, press **SPACE** to start recording.
- Hold your pose and vary it slightly (rotate your hand a little, move
  closer/further from camera) for ~250 samples — this variation is what
  makes the model robust instead of memorizing one exact frame.
- Press **SPACE** again to pause, **R** to retake, **N** for the next
  class, **Q** to quit early (progress is saved as you go).

Important: also record `neutral` — just talk, scratch your head, rest
your hands normally. Without it the model will always force your hands
into one of the 6 domains even when you're not casting anything.

To redo just one class later (e.g. you weren't happy with `jogo`):
```bash
python src/data_collection.py --label jogo
```
This *adds* to existing data for that label rather than replacing it,
unless you press **R** (retake) first.

Data is saved to `data/<label>.npy`. A representative thumbnail of each
pose is auto-saved to `assets/signs/<label>.jpg` — this becomes the
picture used in the bottom-right instructions panel later.

## 2. Train the model

```bash
python src/train_model.py
```

Trains a Random Forest on the landmark vectors and prints accuracy on a
held-out test split. Saves the trained model to
`models/domain_classifier.pkl`. If accuracy looks poor (a lot of
confusion between two domains), that usually means those two poses are
too visually similar — go back and make them more distinct, or collect
more/varied samples with `data_collection.py --label <name>`.

## 3. Run it live

```bash
python src/main.py
```

Hold a trained pose steady. The bottom-left status line shows what the
model currently thinks you're doing and its confidence, e.g.
`Detecting: Unlimited Void (91%)`. Once held long enough it triggers the
full converge → explode → active-overlay sequence. Press **Q** to quit.

## Tuning

Open `config/domains.py` to adjust:
- `HOLD_FRAMES_TO_TRIGGER` — how long you must hold a pose (in frames,
  ~30fps) before it fires. Lower = snappier but more false triggers.
- `CONFIDENCE_THRESHOLD` — minimum model probability per frame to count.
- `COOLDOWN_SECONDS` — minimum gap between two domain expansions.
- `DOMAIN_DISPLAY_SECONDS` — how long the active overlay/info panel stays
  up before fading (in `effects.py` as `ACTIVE_DURATION`).
- Domain names/descriptions/colors — edit directly in the `DOMAINS` dict.

## Adding real background art

Right now the "active" phase renders as a colored tint + vignette over
your live camera feed rather than swapping in a background image (to
avoid bundling copyrighted anime frames into the project). If you want
to drop in your own reference art per domain, put an image at
`assets/domains/<label>.jpg` and extend `DomainEffect.update_and_draw`'s
`"active"` branch in `src/effects.py` to alpha-blend it in instead of
(or behind) the tint — the hook is right there, it's just not wired up
by default.

## Project structure

```
config/domains.py       domain names/colors/descriptions, tunable constants
src/hand_utils.py        MediaPipe wrapper + landmark normalization
src/data_collection.py   record training samples per domain
src/train_model.py       train + save the classifier
src/effects.py           particle swirl/explode/active animation
src/ui.py                info panel + instructions panel rendering
src/main.py              ties it all together, run this to play
data/                    your recorded landmark samples (.npy per class)
models/                  trained classifier (domain_classifier.pkl)
assets/signs/            auto-captured thumbnail per trained sign
assets/domains/          optional: your own background art per domain
```
