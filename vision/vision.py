#!/usr/bin/env python3
"""catvision (GPU worker): find the cat in a camera clip, crop it, and embed the crop.
Runs on an NVIDIA GPU (CUDA), an Apple-silicon Mac (MPS) or, slowly, on CPU.

POST /analyze  body = mp4 bytes; header x-catbox-key; header x-params = {"prompts": [...], "fps": 2}
POST /embed    body = jpg bytes (a reference photo); same headers
  -> {"frames", "persons", "cats": [{"t", "box", "score", "emb", "text", "crop"}], "ms"}
     box is normalized [x0, y0, x1, y1]; emb is base64 float16 (unit length); text holds SigLIP logits,
     one per prompt; crop is a base64 jpg of the cat.

Models from Hugging Face: Ultralytics/YOLO26 (yolo26x, Jan 2026) finds the cat at full 1080p (a camera shot is
wide and the cat is small, so detect then crop), and google/siglip2-so400m-patch16-384 embeds the crop and
scores it zero-shot ("orange and white cat" vs "dark tabby"). DINOv3 and SAM 3.1 would be candidates too but
are gated behind a Hugging Face login + Meta approval. Models load on first use and unload after
IDLE_UNLOAD seconds, so the GPU stays free for other jobs between clips.
"""
from __future__ import annotations

import asyncio
import base64
import gc
import hmac
import io
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

# model tiers (README benchmarks): full is for an NVIDIA GPU; small finds the same cats in a fraction of the time on
# a Mac or a CPU. CATVISION_TIER=full|medium|small|tiny, or auto (full on CUDA, small otherwise).
TIERS = {"full": ("yolo26x.pt", 1920, "google/siglip2-so400m-patch16-384"),
         "medium": ("yolo26m.pt", 1280, "google/siglip2-so400m-patch16-384"),
         "small": ("yolo26s.pt", 1280, "google/siglip2-base-patch16-224"),
         "tiny": ("yolo26n.pt", 960, "google/siglip2-base-patch16-224")}
TIER = os.environ.get("CATVISION_TIER", "auto")
if TIER not in TIERS:
    TIER = "full" if torch.cuda.is_available() else "small"
DET_REPO = "Ultralytics/YOLO26"
DET_FILE, DET_IMGSZ, EMB_ID = TIERS[TIER]   # full 1080p width at "full": a cat across the room is only ~100 px tall
DET_ID = f"{DET_REPO}/{DET_FILE}"
HOME = Path(os.environ.get("CATVISION_HOME", Path.home() / "catvision"))
KEY = (HOME / "key").read_text().strip()
PORT = int(os.environ.get("CATVISION_PORT", "8904"))
IDLE_UNLOAD = 900
DECODE_W = 1920          # decode at the camera's full resolution (Ring 1080p); crops come from it
MAX_FRAMES = 200         # server.py spreads these over a long visit recording (fps under 2)
CAT_MIN_SCORE = 0.35     # YOLO alone is trusted from here
CAT_WEAK_SCORE = 0.12    # 0.12-0.35: a half-hidden or dark cat; kept only if SigLIP also says "cat"
VERIFY_PROMPTS = ["a photo of a cat", "a photo of a pillow", "a photo of a blanket", "a photo of furniture",
                  "a photo of a person", "a photo of a dark shadow", "a photo of a bag", "a photo of an empty floor"]
VERIFY_MIN = 0.6
CROPS_PER_TRACK = 3
MAX_TRACKS = 3
TRACK_DIST = 0.12       # normalized centre distance that still counts as the same cat between frames
log = logging.getLogger("catvision")


class Models:
    def __init__(self):
        self.det = None
        self.last = 0.0

    def load(self):
        if self.det is None:
            t0 = time.time()
            self.dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            # Apple GPU: SigLIP sharing MPS with YOLO deadlocks detection (torch 2.14, macOS 26.4, M1 Pro, 2026-10-08),
            # and it only embeds a few crops per clip, so it runs on the CPU there unless CATVISION_EMB_DEVICE says otherwise
            self.emb_dev = os.environ.get("CATVISION_EMB_DEVICE") or ("cpu" if self.dev == "mps" else self.dev)
            half = torch.float16 if self.emb_dev == "cuda" else torch.float32
            self.det = YOLO(hf_hub_download(DET_REPO, DET_FILE))
            self.emb_proc = AutoProcessor.from_pretrained(EMB_ID)
            self.emb = AutoModel.from_pretrained(EMB_ID, dtype=half).to(self.emb_dev).eval()
            self.half = half
            labels = {v.lower(): int(k) for k, v in self.det.names.items()}
            self.cat_id, self.person_id, self.dog_id = labels["cat"], labels["person"], labels["dog"]
            self.species = {self.cat_id: "cat", self.dog_id: "dog"}
            log.info("models loaded (%s tier: %s @%d + %s; detector %s, embedder %s) in %.1fs",
                     TIER, DET_FILE, DET_IMGSZ, EMB_ID.split("/")[-1], self.dev, self.emb_dev, time.time() - t0)
        self.last = time.time()

    def unload(self):
        if self.det is not None and time.time() - self.last > IDLE_UNLOAD:
            self.det = self.emb = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            log.info("models unloaded (idle)")

    def detect(self, images: list[Image.Image]):
        """Per image: {"labels", "scores", "boxes"} in pixels: cats, dogs and people."""
        out = []
        for i in range(0, len(images), 2):
            for r in self.det.predict(images[i:i + 2], imgsz=DET_IMGSZ, conf=CAT_WEAK_SCORE, classes=[self.cat_id, self.dog_id, self.person_id],
                                      device=0 if self.dev == "cuda" else self.dev, agnostic_nms=True, verbose=False,
                                      **({"half": True} if self.dev == "cuda" else {})):
                b = r.boxes
                out.append({"labels": b.cls.int().cpu(), "scores": b.conf.float().cpu(), "boxes": b.xyxy.float().cpu()})
        return out

    @torch.inference_mode()
    def embed(self, crops: list[Image.Image], prompts: list[str]):
        inp = self.emb_proc(images=crops, return_tensors="pt").to(self.emb_dev)
        img = self.emb.get_image_features(pixel_values=inp["pixel_values"].to(self.half),
                                          **{k: v for k, v in inp.items() if k != "pixel_values"})
        img = torch.nn.functional.normalize(_features(img).float(), dim=-1)
        logits = None
        if prompts:
            t = self.emb_proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(self.emb_dev)
            txt = torch.nn.functional.normalize(_features(self.emb.get_text_features(**t)).float(), dim=-1)
            logits = img @ txt.T * self.emb.logit_scale.exp().float() + self.emb.logit_bias.float()
        return img.cpu().numpy(), (logits.cpu().numpy() if logits is not None else None)


def _features(x):
    """transformers 5 returns an output object from get_*_features; the embedding is its pooler_output."""
    return x if torch.is_tensor(x) else x.pooler_output


M = Models()
LOCK = asyncio.Lock()


def decode(mp4: bytes, fps: float) -> list[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4)
        f.flush()
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", f.name], capture_output=True, text=True, check=True)
        w0, h0 = (int(x) for x in probe.stdout.strip().split(","))
        w = min(DECODE_W, w0)
        h = int(round(h0 * w / w0 / 2) * 2)
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f.name, "-vf", f"fps={fps},scale={w}:{h}",
                              "-frames:v", str(MAX_FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)
    return [Image.fromarray(a) for a in arr]


def crop_of(im: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * 0.15, (y1 - y0) * 0.15
    c = im.crop((max(0, x0 - pw), max(0, y0 - ph), min(im.width, x1 + pw), min(im.height, y1 + ph)))
    c.thumbnail((640, 640))
    return c


def run(images: list[Image.Image], fps: float, prompts: list[str]) -> dict:
    """Every cat in every frame, grouped into tracks (one track = one cat), up to CROPS_PER_TRACK crops each."""
    t0 = time.time()
    M.load()
    dets = M.detect(images)
    persons, cand = 0, []
    for i, (im, d) in enumerate(zip(images, dets)):
        labels, scores, boxes = d["labels"].tolist(), d["scores"].tolist(), d["boxes"].tolist()
        if any(l == M.person_id and s >= 0.5 for l, s in zip(labels, scores)):
            persons += 1
        for l, s, b in zip(labels, scores, boxes):
            if l in M.species and s >= CAT_WEAK_SCORE:
                cand.append({"i": i, "t": round(i / fps, 1), "score": round(s, 3), "box": b, "species": M.species[l],
                             "area": (b[2] - b[0]) * (b[3] - b[1]) / (im.width * im.height),
                             "cx": (b[0] + b[2]) / 2 / im.width, "cy": (b[1] + b[3]) / 2 / im.height,
                             "verified": s >= CAT_MIN_SCORE})
    # borderline boxes: ask SigLIP "cat (or dog), or pillow / blanket / shadow / ...?" and keep only real animals
    weak = sorted((c for c in cand if not c["verified"]), key=lambda c: -c["score"])[:24]
    for species in ("cat", "dog"):
        mine = [c for c in weak if c["species"] == species]
        if mine:
            prompts = [f"a photo of a {species}"] + VERIFY_PROMPTS[1:]
            _, vlog = M.embed([crop_of(images[c["i"]], c["box"]) for c in mine], prompts)
            for c, row in zip(mine, vlog):
                e = np.exp(row - row.max())
                c["cat_p"] = round(float(e[0] / e.sum()), 3)
                c["verified"] = c["cat_p"] >= VERIFY_MIN
    cand = [c for c in cand if c["verified"]]
    # tracks: link each detection to the nearest track seen in an earlier frame (cats move slowly at 2 fps)
    tracks: list[dict] = []
    for c in sorted(cand, key=lambda c: (c["i"], -c["score"])):
        near = [tr for tr in tracks if tr["last_i"] < c["i"] and math.hypot(tr["cx"] - c["cx"], tr["cy"] - c["cy"]) < TRACK_DIST]
        tr = min(near, key=lambda tr: math.hypot(tr["cx"] - c["cx"], tr["cy"] - c["cy"])) if near else None
        if tr is None:
            tr = {"id": len(tracks), "items": [], "species": c["species"]}
            tracks.append(tr)
        tr.update(cx=c["cx"], cy=c["cy"], last_i=c["i"])
        tr["items"].append(c)
    tracks = [tr for tr in tracks if len(tr["items"]) >= 2 or len(images) <= 2]   # one-frame blips are noise
    tracks.sort(key=lambda tr: -len(tr["items"]))
    picked = []
    for tr in tracks[:MAX_TRACKS]:
        mine = []
        for c in sorted(tr["items"], key=lambda c: -(c["score"] * c["area"] ** 0.5)):
            if len(mine) >= CROPS_PER_TRACK:
                break
            if all(abs(c["t"] - m["t"]) >= 2 for m in mine):
                mine.append({**c, "track": tr["id"], "track_frames": len(tr["items"])})
        picked += mine
    picked.sort(key=lambda c: (c["track"], c["t"]))
    cats = []
    if picked:
        crops = [crop_of(images[c["i"]], c["box"]) for c in picked]
        embs, logits = M.embed(crops, prompts)
        # species per track: SigLIP "cat or dog" averaged over the track's crops (YOLO flips class between frames)
        _, sp = M.embed(crops, ["a photo of a cat", "a photo of a dog"])
        votes: dict[int, list] = {}
        for c, row in zip(picked, sp):
            e = np.exp(row - row.max())
            votes.setdefault(c["track"], []).append(float(e[1] / e.sum()))
        for c in picked:
            dog_p = sum(votes[c["track"]]) / len(votes[c["track"]])
            c["species"], c["dog_p"] = ("dog" if dog_p >= 0.5 else "cat"), round(dog_p, 3)
        for k, (c, crop) in enumerate(zip(picked, crops)):
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=88)
            im = images[c["i"]]
            x0, y0, x1, y1 = c["box"]
            px = np.asarray(crop, dtype=np.int16)
            cats.append({"t": c["t"], "score": c["score"], "cat_p": c.get("cat_p"), "track": c["track"], "species": c["species"], "dog_p": c["dog_p"],
                         "ir": bool(np.abs(px[..., 0] - px[..., 2]).mean() < 2),   # night vision: Ring's IR picture is pure grey
                         "track_frames": c["track_frames"],
                         "box": [round(x0 / im.width, 4), round(y0 / im.height, 4), round(x1 / im.width, 4), round(y1 / im.height, 4)],
                         "emb": base64.b64encode(embs[k].astype(np.float16).tobytes()).decode(),
                         "text": [round(float(x), 3) for x in logits[k]] if logits is not None else [],
                         "crop": base64.b64encode(buf.getvalue()).decode()})
    return {"frames": len(images), "cat_frames": len({c["i"] for c in cand}), "tracks": len(tracks), "persons": persons,
            "cats": cats, "ms": int((time.time() - t0) * 1000), "models": [DET_ID, EMB_ID]}


def params(request) -> dict:
    try:
        return json.loads(request.headers.get("x-params") or "{}")
    except ValueError:
        return {}


@web.middleware
async def auth(request, handler):
    if request.path != "/healthz" and not hmac.compare_digest(request.headers.get("x-catbox-key", "").encode(), KEY.encode()):
        raise web.HTTPForbidden()
    return await handler(request)


async def analyze(request):
    p = params(request)
    fps = float(p.get("fps", 2))
    body = await request.read()
    loop = asyncio.get_running_loop()
    images = await loop.run_in_executor(None, decode, body, fps)
    async with LOCK:
        res = await loop.run_in_executor(None, run, images, fps, p.get("prompts") or [])
    log.info("analyze: %d frames, %d with a cat, %d crops, %d ms", res["frames"], res["cat_frames"], len(res["cats"]), res["ms"])
    return web.json_response(res)


async def embed_photo(request):
    p = params(request)
    im = Image.open(io.BytesIO(await request.read())).convert("RGB")
    loop = asyncio.get_running_loop()
    async with LOCK:
        res = await loop.run_in_executor(None, run, [im], 1.0, p.get("prompts") or [])
        if not res["cats"]:  # close-up profile photo the detector didn't box: embed the whole picture
            embs, logits = await loop.run_in_executor(None, M.embed, [im], p.get("prompts") or [])
            res["cats"] = [{"t": 0, "score": 0.0, "box": [0, 0, 1, 1], "crop": "",
                            "emb": base64.b64encode(embs[0].astype(np.float16).tobytes()).decode(),
                            "text": [round(float(x), 3) for x in logits[0]] if logits is not None else []}]
    return web.json_response(res)


async def healthz(request):
    return web.json_response({"ok": True, "loaded": M.det is not None, "cuda": torch.cuda.is_available()})


async def idle_loop(app):
    while True:
        await asyncio.sleep(60)
        async with LOCK:
            M.unload()


async def on_startup(app):
    app["idle"] = asyncio.create_task(idle_loop(app))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = web.Application(middlewares=[auth], client_max_size=64 * 1024 * 1024)
    app.router.add_post("/analyze", analyze)
    app.router.add_post("/embed", embed_photo)
    app.router.add_get("/healthz", healthz)
    app.on_startup.append(on_startup)
    host = os.environ.get("CATVISION_BIND") or subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.strip() or "127.0.0.1"
    web.run_app(app, host=host, port=PORT, access_log=None)
