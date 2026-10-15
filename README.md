# Litter Log

**Free, self-hosted cat ID for Litter-Robot owners with more than one cat.** It tells you which cat used which box using the robot's own scale, and optionally your cameras. It runs on hardware you already own, down to a 16 GB M1 MacBook or an old laptop with no graphics card.

Not affiliated with Whisker. Litter-Robot and Whisker+ are Whisker's trademarks.

## Benchmarks

Seconds to analyze one 39-second 1080p Ring clip (80 frames at 2 fps), with the models already loaded. Anything under 39 s keeps up in real time, and nothing has to be instant: each clip is checked once, about a minute after the visit.

| Hardware | Full | Medium | Small | Tiny |
|---|---|---|---|---|
| MacBook Pro M1 Pro, 16 GB (2021) | 69 s | 10.4 s | 2.9 s | 1.2 s |
| MacBook Pro M1 Pro, 16 GB (2021), CPU only | not run | not run | 10.7 s | 3.8 s |
| MacBook Air M5, 24 GB | 45 s | 10.5 s | 2.5 s | 0.8 s |
| Gaming laptop, RTX 3060 6 GB + i7-11800H (2021) | 9.7 s | 2.7 s | 1.1 s | 1.0 s |
| Gaming laptop, RTX 3060 6 GB + i7-11800H (2021), CPU only | not run | not run | 41 s | 4.3 s |
| Business laptop, i7-8665U, no GPU (2019) | not run | not run | 40 s | 8.2 s |
| Peak RAM, whole process | 4.5 to 6.3 GB | 5.9 to 8.3 GB | 2.4 to 4.0 GB | 2.2 to 3.9 GB |

The tiers (`CATVISION_TIER`; `auto` = Full on an NVIDIA GPU, Small everywhere else):

- **Full:** YOLO26-X at 1920 px + SigLIP 2 so400m (384 px)
- **Medium:** YOLO26-M at 1280 px + SigLIP 2 so400m
- **Small:** YOLO26-S at 1280 px + SigLIP 2 base (224 px)
- **Tiny:** YOLO26-N at 960 px + SigLIP 2 base

On Apple GPUs the embedder runs on the CPU (see Camera model). Laptop CPUs slow down as they heat up: the gaming laptop's CPU took 22 s on its first Small run and 41 s on the second, and the table shows the slower. Measured 2026-10-08 with torch 2.14, transformers 5.19 and Ultralytics 8.4; `bench/` has the script.

**Accuracy, honestly:** so far we've compared tiers on 5 real clips: our dog walking past, 3 empty ones, and one cat walking to the box at night. Every tier found the dog and the cat, and none invented an animal; Tiny missed 1 of 9 frames with the dog in it. Naming the cat from a coat description failed on every tier at night: all four called our small dark tabby the large brown one (Tiny also said the orange cat twice). The camera's own naming counts only after it passes a self-test in that light: it must name its own past clips of each cat right 9 times in 10 (3+ clips per cat), separately for colour and for night vision. Until then, weight, box and time decide, and a cat on camera is "a cat". One simple rule works from day one: in colour, an orange coat is the orange cat and any other coat is a tabby (measured: our orange cat's crops read 16 to 18 on a red-minus-blue scale, every tabby crop 13 or less), with its own self-test that switches it off if it ever stops holding. Night vision has its own version: orange fur shows up near-white in infrared, so on a camera that sees cats from across a room a light coat is the orange cat (ours read 122 to 147 on a 0 to 255 grey scale, every tabby 26 to 71). Set it per camera (`ir_coat`), because up close the infrared lamp lights a tabby up to 136. Both rules look only at the cat walking up to the box or out of it: a cat seen in between may be another one walking past. Each camera also knows which cats use its box, so upstairs a tabby is named as the tabby who goes there. In our first tests it hadn't earned a vote in either: in daylight it called our orange cat walking away the brown tabby, and close-up phone photos didn't fix that. Night infrared is the hardest case: in grey, our two brown tabbies' photos look as alike to each other as to themselves, so night photos only count once the night references pass a self-test (they must name each other's clips right 9 times in 10). Until then the camera says "a cat" at night instead of guessing. The weight-based cat ID needs no camera model at all.


## Why

Whisker's app guesses which cat used the box from weight alone. That breaks down when two cats weigh about the same. Whisker+ adds camera face ID, but only on the Litter-Robot 5 Pro. Litter Log works with the boxes you already have:

- **Per-box weight models.** Two boxes' scales disagree. On ours, the same cats read 0.3 to 0.5 lb lighter on the Litter-Robot 5 than on the Litter-Robot 4, and modeling each box separately is what pulled two "identical" 10 lb cats apart.
- **Time in the box** counts as supporting evidence (some cats always take longer).
- **Household facts:** move-in dates, and who can reach which box overnight.
- **One-tap labels** from anyone in the family. Taps train the model and get re-checked against everything else, so an accidental tap gets flagged.
- **Best guesses with plain reasons**, never dressed up as fact. A true toss-up is asked as a question instead.
- **A per-cat view:** where each cat goes, how often, when, its weight on each box over time, and its longest stretch without a visit.
- **Photo-library references (optional):** `bench/photolib.py` finds your cats in your own photos (macOS Photos, on your machines) and turns the ones you name into camera references. Only their fingerprints are kept, never the photos.
- **Box controls** for anyone with the family code: clean now (two taps), night light, button lock, wait time, and "I emptied the drawer" on a Litter-Robot 5.
- **Feeder controls (optional):** an HGsmart / HoneyGuaridan feeder's buttons on the same page: feed now (two taps), the meal schedule in local time (kept right across daylight saving, which the feeder's own UTC schedule isn't), button lock, meal call sound, "I filled it up" and "new desiccant". Meals inside the house's quiet hours ask twice. Tested on the S25D.
- **Optional cameras:** Ring clips of each visit. [YOLO26](https://huggingface.co/Ultralytics/YOLO26) finds the cat, and [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch16-384) helps name it.

## What you need

- A Whisker account with one Litter-Robot 4 and/or one Litter-Robot 5 (EVO).
- A computer that stays on: Linux or macOS, Python 3.11+.
- Optional: Ring cameras pointed at the boxes (Node 20+ for the Ring sidecar).
- Optional: a GPU for the camera model, either an NVIDIA card or an Apple-silicon Mac. A CPU works for the small models (see the table).

## Quick start

```bash
git clone https://github.com/<you>/litter-log && cd litter-log
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
WHISKER_EMAIL=you@example.com .venv/bin/python mint-whisker-token.py   # saves tokens only, never your password
.venv/bin/python server.py                                             # first run prints the family code
```

Open http://localhost:8902 and type the family code. It's saved in `secrets/family-code`, and you can change it to a word. To use the app from phones on your Wi-Fi, run with `CATBOX_BIND=0.0.0.0`. From anywhere, put it behind Tailscale Funnel or another HTTPS proxy.

Household config is optional. Copy the examples and edit them:

| File | What it holds |
|---|---|
| `cats.json` | each cat's look ("an orange tabby cat") for the camera model, and its color |
| `house.json` | box names (Upstairs/Downstairs), move-in dates, the night routine |
| `cams.json` | which Ring camera watches which box |
| `pets.json` | pets Whisker doesn't know, like a dog the camera may see |

## Cameras (optional)

```bash
cd ring && npm ci && node auth.mjs you@example.com   # one-time Ring sign-in; saves a refresh token
node ring.mjs                                        # records clips on motion and on each visit
```

Aim each camera at the box's opening. A Litter-Robot 5's cloud reports a cat about a minute after it steps in, so a visit recording starts with the cat already inside; a camera that only sees the room around the box catches the cat just when it leaves through that room (for us, about half of visits). Ring's own motion sensor didn't help either: on our battery doorbell, with People Only Mode off and sensitivity raised, it fired for the dog six times in two hours and never once for a cat.

## Feeder (optional)

Put the HGsmart account's tokens in `secrets/hgsmart.json` (mode 600): `{"email": ..., "refresh_token": ..., "device_id": ...}`. A refresh token comes from one sign-in with the vendored client in `hgsmart_client/` (MIT, from [HA_hgsmart-pet-feeder](https://github.com/Serjlee/HA_hgsmart-pet-feeder)); the password is never stored. Each token use returns a new one, which the app saves, and older ones keep working, so the HGsmart app on your phone stays signed in. Quiet hours go in `house.json` (`feeder_quiet`).

## Camera model (optional)

On the machine with the GPU (or the same machine):

```bash
python3 -m venv .venv && .venv/bin/pip install -r vision/requirements.txt
head -c 24 /dev/urandom | base64 > key      # copy the same value to the app's secrets/vision-key
.venv/bin/python vision/vision.py           # port 8904; set CATBOX_VISION=http://<that machine>:8904 on the app
```

The worker picks CUDA, then Apple's GPU (MPS), then CPU. On Apple silicon the embedder runs on the CPU, because sharing MPS with the detector deadlocked it on macOS 26.4 with torch 2.14. Models load on first use and unload after 15 idle minutes.

`systemd/` has user units for all three services, and `deploy.sh` ships the lot to a Linux box over SSH (settings in `deploy.env`; it points the app at the GPU box's camera model for you).

## Privacy

Everything runs on your machines. Only Whisker, Ring and (optionally) HGsmart session tokens are stored (`secrets/`, mode 600), never passwords. Clips with no pet and no visit are deleted after 30 minutes, and all clips after 14 days. The bedroom nap check stores nothing at all when a person is in view. The access log never records query strings.

## Limits

- Whisker and Ring have no public APIs. This uses the community libraries [pylitterbot](https://github.com/natekspencer/pylitterbot) and [ring-client-api](https://github.com/dgreif/ring), and either side can break them at any time.
- One Litter-Robot 4 and one Litter-Robot 5 per account, for now.
- Tested in one home with three cats, two boxes and two Ring cameras.

## License

AGPL-3.0, because the detector (Ultralytics YOLO) is AGPL-3.0.
