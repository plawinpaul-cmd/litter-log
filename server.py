#!/usr/bin/env python3
"""Litter Log: which cat used which litter box, with camera photos.

Whisker cloud (one Litter-Robot 4 and/or one Litter-Robot 5 "EVO") -> SQLite -> a phone web app.
The optional Ring sidecar (ring/ring.mjs) records clips on motion or when a robot sees a cat, and
posts them to /internal/clip; visits and clips are joined by time.

Cat ID per visit: a human tap wins. Otherwise per-box weight and time-in-box models learned from
taps and known visits, times the camera model (vision/vision.py) when a clip exists.

Env: CATBOX_HOME (default: this dir), CATBOX_PORT (8902), CATBOX_RING (http://127.0.0.1:8903),
CATBOX_VISION (http://127.0.0.1:8904), CATBOX_TZ (America/New_York)
Secrets in CATBOX_HOME/secrets (mode 600): whisker-token.json, web-key, internal-key, family-code
Household config (all optional, see *.example.json): cams.json, pets.json, house.json, cats.json
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import math
import os
import secrets
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import aiohttp
from aiohttp import web
from pylitterbot import Account
from pylitterbot.enums import LitterBoxStatus, NightLightMode
from pylitterbot.event import EVENT_UPDATE
from pylitterbot.transport import PollingTransport

from feeder import Feeder, feeder_loop

HOME = Path(os.environ.get("CATBOX_HOME", Path(__file__).resolve().parent))
DATA = HOME / "data"
MEDIA = DATA / "media"
SECRETS = HOME / "secrets"
STATIC = HOME / "static"
PORT = int(os.environ.get("CATBOX_PORT", "8902"))
RING = os.environ.get("CATBOX_RING", "http://127.0.0.1:8903")
VISION = os.environ.get("CATBOX_VISION", "http://127.0.0.1:8904")   # the GPU worker, vision/vision.py


def config(name: str) -> dict:
    f = HOME / name
    return json.loads(f.read_text()) if f.exists() else {}


# which litter box each camera watches: its clips only count for that robot's visits
CAMS = config("cams.json")        # Ring camera name -> {robot, label}; ring.mjs reads it too
CAM_ROBOT = {name: c["robot"] for name, c in CAMS.items()}
PETS = config("pets.json")        # animals Whisker doesn't know (a dog)
DOG = next(({"id": n.lower(), "name": n, "species": "dog", "color": "#D9D4CC"} for n, x in PETS.items()
            if x.get("species") == "dog"), None)
NY = ZoneInfo(os.environ.get("CATBOX_TZ", "America/New_York"))   # the household's local time
HOUSE = config("house.json")      # who can reach which box when (move-in dates, night routine)


def house_prior(db_cats: list[dict], robot: str, ts: int) -> tuple[dict | None, list[str]]:
    """Who could have used this box then: residency (a cat that hadn't moved in yet is 0) times the night routine."""
    ids = {c["name"].strip().lower(): c["id"] for c in db_cats}
    pr, how = {cid: 1.0 for cid in ids.values()}, []
    for r in HOUSE.get("residents", []):
        if r["cat"] in ids and ts < datetime.fromisoformat(r["since"]).replace(tzinfo=NY).timestamp():
            pr[ids[r["cat"]]] = 0.0
            if "before the move" not in how:
                how.append("before the move")
    rt = routine(db_cats, robot, ts)
    if rt:
        for cid, m in rt.items():
            pr[cid] *= m
        how.append("night routine")
    return (pr, how) if how else (None, [])


def routine(db_cats: list[dict], robot: str, ts: int) -> dict | None:
    """Night routine prior for one visit: {cat_id: multiplier}, or None outside the window."""
    r = HOUSE.get("night_routine")
    if not r:
        return None
    t = datetime.fromtimestamp(ts, NY)
    if t.strftime("%a") not in r["days"] or not (r["from"] <= t.strftime("%H:%M") < r["to"]):
        return None
    ids = {c["name"].strip().lower(): c["id"] for c in db_cats}
    if robot == "evo" and r.get("evo_only") in ids:
        return {cid: (1.0 if cid == ids[r["evo_only"]] else r["strength"]) for cid in ids.values()}
    if robot == "lr4" and r.get("lr4_never") in ids:
        return {cid: (r["strength"] if cid == ids[r["lr4_never"]] else 1.0) for cid in ids.values()}
    return None
# per cat (lower-case name): "look" = zero-shot description for the camera model, "color" = its UI colour
CATS_CFG = config("cats.json")
CAT_LOOKS = {k: v["look"] for k, v in CATS_CFG.items() if v.get("look")}
TOKEN_FILE = SECRETS / "whisker-token.json"
ROBOT_KEY = {"LitterRobot4": "lr4", "LitterRobot5": "evo"}
ROBOT_LABEL = {"lr4": "LR4", "evo": "EVO"}
CAT_COLORS = {k: v["color"] for k, v in CATS_CFG.items() if v.get("color")}
PALETTE = ["#e8743b", "#7b8794", "#c8873a", "#5b9a8b", "#b46fa8", "#d1a43a"]
EVO_POLL_SECONDS = 15   # the EVO has no push; its cat-in/out wakes the EVO cam, so poll it faster than the default 30
EVO_POLL_INSIDE = 4     # while a cat is on the scale: catch the exit within seconds, since the camera sees the room, not the box
# A cat in the box. pylitterbot ranks the cycle state first, and both robots report a visit there as
# "cat detect", which it names CAT_SENSOR_INTERRUPTED; checking CAT_DETECTED alone missed every visit.
CAT_INSIDE = (LitterBoxStatus.CAT_DETECTED, LitterBoxStatus.CAT_SENSOR_INTERRUPTED)
BUSY = (LitterBoxStatus.CLEAN_CYCLE, LitterBoxStatus.EMPTY_CYCLE, LitterBoxStatus.POWER_UP, LitterBoxStatus.POWER_DOWN)
# states that need a person before the box cleans again, in words for whoever reads the page
# the bonnet sensor can read "off" with the cover sitting on but not seated, so say what the box sees
ATTENTION = {LitterBoxStatus.BONNET_REMOVED: "Top cover not detected: it won't clean until the cover is lifted off and reseated", LitterBoxStatus.OFF: "Switched off",
             **{s: s.text for s in (LitterBoxStatus.PINCH_DETECT, LitterBoxStatus.STARTUP_PINCH_DETECT, LitterBoxStatus.CAT_SENSOR_FAULT,
                                    LitterBoxStatus.STARTUP_CAT_SENSOR_FAULT, LitterBoxStatus.OVER_TORQUE_FAULT, LitterBoxStatus.DUMP_POSITION_FAULT,
                                    LitterBoxStatus.HOME_POSITION_FAULT, LitterBoxStatus.DUMP_HOME_POSITION_FAULT)}}


def robot_status(r) -> LitterBoxStatus:
    """r.status, except where the box's own status light names a problem its state field hides: the EVO keeps
    reporting StRobotIdle with its bonnet off and pylitterbot reads that field first, so a box that couldn't clean
    showed as Ready for hours (2026-10-15). A remote reset, a remote power cycle and a clean command don't clear it:
    the box rechecks the sensor and still refuses, so only reseating the cover does."""
    st = r.status
    s = getattr(r, "_state", None) or {}
    if st == LitterBoxStatus.READY and (s.get("isBonnetRemoved") is True or (s.get("statusIndicator") or {}).get("type") == "BONNET_REMOVED"):
        return LitterBoxStatus.BONNET_REMOVED
    return st
CLIP_KEEP_DAYS = 14      # mp4s; the frames stay
FRAME_KEEP_DAYS = 120
log = logging.getLogger("catbox")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cats(id TEXT PRIMARY KEY, name TEXT, weight REAL, updated INTEGER);
CREATE TABLE IF NOT EXISTS whisker_weights(pet_id TEXT, ts INTEGER, weight REAL, PRIMARY KEY(pet_id, ts));
CREATE TABLE IF NOT EXISTS visits(
  id TEXT PRIMARY KEY, robot TEXT, start_ts INTEGER, end_ts INTEGER, weight REAL, duration INTEGER,
  waste_weight REAL, whisker_cat TEXT, label_cat TEXT, label_by TEXT, label_at INTEGER, raw TEXT);
CREATE INDEX IF NOT EXISTS visits_end ON visits(end_ts);
CREATE TABLE IF NOT EXISTS clips(
  id INTEGER PRIMARY KEY, cam TEXT, start_ts INTEGER, end_ts INTEGER, reason TEXT, robot TEXT,
  path TEXT, frames TEXT, look TEXT);
CREATE INDEX IF NOT EXISTS clips_start ON clips(start_ts);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS nap_checks(id INTEGER PRIMARY KEY, cam TEXT, ts INTEGER, persons INTEGER, animals TEXT);
CREATE INDEX IF NOT EXISTS nap_checks_ts ON nap_checks(ts);
"""


# ---------- storage ----------

def db_open() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / "catbox.db")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    if "vision" not in {r["name"] for r in db.execute("PRAGMA table_info(clips)")}:
        db.execute("ALTER TABLE clips ADD COLUMN vision TEXT")
    return db


def kv_get(db, k, default=None):
    row = db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(row["v"]) if row else default


def kv_set(db, k, v):
    db.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))
    db.commit()


def write_private(f: Path, text: str):
    """Create or replace a file readable only by this user, with no moment where it is world-readable."""
    tmp = f.with_name(f.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, f)


def secret(name: str) -> str:
    f = SECRETS / name
    if not f.exists() or len(f.read_text().strip()) < 6:   # an empty key file would match an empty cookie
        SECRETS.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_private(f, secrets.token_urlsafe(32) + "\n")
    return f.read_text().strip()


def epoch(s: str | None) -> int | None:
    """Whisker timestamps: LR4 '2026-10-08 17:58:43.000000000', EVO '...T11:27:28Z', pets '...T00:26:11'. All UTC."""
    if not s:
        return None
    s = s.strip().replace(" ", "T").rstrip("Z")
    s = s.split(".")[0]
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def upsert_visit(db, v: dict):
    db.execute(
        """INSERT INTO visits(id,robot,start_ts,end_ts,weight,duration,waste_weight,whisker_cat,raw)
           VALUES(:id,:robot,:start_ts,:end_ts,:weight,:duration,:waste_weight,:whisker_cat,:raw)
           ON CONFLICT(id) DO UPDATE SET
             start_ts=MIN(visits.start_ts, excluded.start_ts), end_ts=excluded.end_ts,
             weight=excluded.weight, duration=COALESCE(excluded.duration, visits.duration),
             waste_weight=COALESCE(excluded.waste_weight, visits.waste_weight),
             whisker_cat=COALESCE(excluded.whisker_cat, visits.whisker_cat), raw=excluded.raw""",
        {"duration": None, "waste_weight": None, "whisker_cat": None, **v, "raw": json.dumps(v.get("raw"))},
    )


# ---------- Whisker sync ----------

LR4_ACTIVITY = """query Q($serial:String!,$limit:Int,$consumer:String){getLitterRobot4Activity(
  serial:$serial,limit:$limit,consumer:$consumer){measure timestamp value actionValue}}"""


def parse_lr4(rows: list[dict]) -> list[dict]:
    """A visit = cat detected ... catWeight (posted as the cat leaves; the wait timer starts then)."""
    out, detect = [], None
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        t = epoch(r["timestamp"])
        if r["value"] in ("robotStatusCatDetect", "robotCycleStateCatDetect"):
            detect = detect or t
        elif r["value"] == "catWeight":
            start = detect if detect and 0 <= t - detect <= 900 else t - 60
            out.append({"id": f"lr4:{t}", "robot": "lr4", "start_ts": start, "end_ts": t,
                        "weight": float(r["actionValue"]), "duration": t - start, "raw": r})
            detect = None
    return out


def parse_evo(acts: list[dict]) -> list[dict]:
    """EVO PET_VISIT: timestamp is the exit (cycle completes wait+~2 min later); petWeight is in 1/100 lb."""
    out = []
    for x in acts:
        if x.get("type") != "PET_VISIT" or not x.get("petWeight"):
            continue
        end = epoch(x.get("timestamp"))
        dur = int(x.get("duration") or 0)
        w = float(x["petWeight"])
        pets = x.get("petIds") or []
        out.append({"id": f"evo:{x['eventId']}", "robot": "evo", "start_ts": end - dur, "end_ts": end,
                    "weight": round(w / 100 if w > 100 else w, 2), "duration": dur,
                    "waste_weight": x.get("wasteWeight"), "whisker_cat": pets[0] if len(pets) == 1 else None, "raw": x})
    return out


class Whisker:
    def __init__(self, app: web.Application):
        self.app = app
        self.account: Account | None = None
        self.robots: list = []
        self.status: dict[str, dict] = {}
        self.last_cat_state: dict[str, bool] = {}
        self.last_status: dict[str, LitterBoxStatus] = {}
        self.last_sig: dict[str, tuple] = {}
        self.tasks: set[asyncio.Task] = set()
        self.locks: dict[str, asyncio.Lock] = {}
        self.last_pets = 0

    def save_token(self, tokens):
        if tokens:
            write_private(TOKEN_FILE, json.dumps(tokens) + "\n")

    async def connect(self):
        token = json.loads(TOKEN_FILE.read_text())
        self.account = Account(token=token, token_update_callback=self.save_token)
        await self.account.connect(load_robots=True, load_pets=True)
        self.robots = list(self.account.robots)
        for r in self.robots:
            if ROBOT_KEY.get(type(r).__name__) == "evo":
                r._transport = PollingTransport(interval=EVO_POLL_SECONDS)
            await r.subscribe()  # LR4: websocket push; EVO: polling
            r.on(EVENT_UPDATE, lambda r=r: self.on_robot_update(r))
            self.on_robot_update(r)
        log.info("whisker: %s", [r.name for r in self.robots])

    def robot(self, key: str):
        return next((r for r in self.robots if ROBOT_KEY.get(type(r).__name__) == key), None)

    def on_robot_update(self, r):
        key = ROBOT_KEY.get(type(r).__name__, r.serial)
        st = robot_status(r)
        raw = {**getattr(r, "_data", {}), **(getattr(r, "_state", None) or {})}
        # the EVO's cloud state skips "cat detect" for most of a visit (2026-10-09: a 166 s visit showed up 94 s in,
        # straight as the wait timer), but its scale flag says a cat is on it: WeightSet, then WeightClear on the way out
        inside = st in CAT_INSIDE or raw.get("catDetect") == "WeightSet"
        if isinstance(getattr(r, "_transport", None), PollingTransport):
            r._transport._interval = EVO_POLL_INSIDE if inside else EVO_POLL_SECONDS
        nl = prop(r, "night_light_mode")
        self.status[key] = {"key": key, "label": ROBOT_LABEL.get(key, key), "name": r.name,
                            "status": st.text if isinstance(st, LitterBoxStatus) else str(st), "attention": ATTENTION.get(st),
                            "cat_inside": inside, "busy": st in BUSY, "timing": st == LitterBoxStatus.CAT_SENSOR_TIMING,
                            "online": bool(r.is_online), "drawer": r.waste_drawer_level,
                            "litter": prop(r, "litter_level"), "seen": int(time.time()),
                            "night_light": nl.value if isinstance(nl, NightLightMode) else None,
                            "panel_lock": prop(r, "panel_lock_enabled"), "sleeping": prop(r, "is_sleeping"),
                            "wait": prop(r, "clean_cycle_wait_time_minutes"), "waits": list(r.VALID_WAIT_TIMES),
                            "drawer_reset": hasattr(r, "reset_waste_drawer")}
        was_st = self.last_status.get(key)
        self.last_status[key] = st
        sig = (st, raw.get("catDetect"))
        if self.last_sig.get(key) not in (None, sig):   # what each robot really reports, for the next surprise
            log.info("robot %s: %s -> %s %s", key, getattr(was_st, "name", was_st), getattr(st, "name", st),
                     {k: raw.get(k) for k in ("robotStatus", "robotCycleState", "state", "cycleState", "displayCode", "catDetect") if raw.get(k)})
        self.last_sig[key] = sig
        was = self.last_cat_state.get(key)
        self.last_cat_state[key] = inside
        if was is None or was == inside:
            # the wait timer starting with the scale already clear: the cat left before this side saw it, so at least
            # record the room it walked into
            if was is False and st == LitterBoxStatus.CAT_SENSOR_TIMING and was_st not in (LitterBoxStatus.CAT_SENSOR_TIMING, None):
                log.info("robot %s: wait timer started without a cat seen inside; recording the exit", key)
                self.spawn(ring_capture("out", key))
            return
        self.spawn(ring_capture("in" if inside else "out", key))

    def spawn(self, coro):
        t = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(t)                      # the loop keeps only a weak reference to a task
        t.add_done_callback(self.tasks.discard)

    async def sync(self):
        db = self.app["db"]
        if time.time() - self.last_pets > 1800:
            await self.sync_pets(db)
            self.last_pets = time.time()
        for r in self.robots:
            key = ROBOT_KEY.get(type(r).__name__)
            if key == "lr4":
                data = await r._post(json={"query": LR4_ACTIVITY,
                                           "variables": {"serial": r.serial, "limit": 200, "consumer": "app"}})
                visits = parse_lr4(data["data"]["getLitterRobot4Activity"] or [])
                for v in visits:
                    v["whisker_cat"] = whisker_pick_lr4(db, v["end_ts"], v["weight"])
            elif key == "evo":
                visits = parse_evo(await r.get_activities(limit=200))
            else:
                continue
            for v in visits:
                upsert_visit(db, v)
            db.commit()
        kv_set(db, "whisker_sync", int(time.time()))

    async def sync_pets(self, db):
        await self.account.load_pets()
        for p in self.account.pets:
            name = p.name.strip()
            db.execute("INSERT INTO cats(id,name,weight,updated) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                       "name=excluded.name, weight=excluded.weight, updated=excluded.updated",
                       (p.id, name, p.weight, int(time.time())))
            for w in await p.fetch_weight_history(limit=200):
                db.execute("INSERT OR IGNORE INTO whisker_weights(pet_id,ts,weight) VALUES(?,?,?)",
                           (p.id, int(w.timestamp.timestamp()), w.weight))
            photo = DATA / "cats" / f"{p.id}.jpg"
            if p.image_url and (not photo.exists() or time.time() - photo.stat().st_mtime > 7 * 86400):
                photo.parent.mkdir(parents=True, exist_ok=True)
                async with aiohttp.ClientSession() as s, s.get(p.image_url) as resp:
                    if resp.status == 200:
                        photo.write_bytes(await resp.read())
        db.commit()


def prop(r, name: str):
    """A robot property that may be missing from this model or this payload."""
    try:
        return getattr(r, name)
    except Exception:
        return None


def whisker_pick_lr4(db, ts: int, weight: float) -> str | None:
    """Whisker assigns LR4 visits only inside its app; recover its pick from the pet weight histories."""
    row = db.execute("SELECT pet_id FROM whisker_weights WHERE ts BETWEEN ? AND ? AND ABS(weight-?)<0.015 "
                     "ORDER BY ABS(ts-?) LIMIT 1", (ts - 180, ts + 180, weight, ts)).fetchone()
    return row["pet_id"] if row else None


async def ring_capture(reason: str, robot: str):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as s:
            await s.post(f"{RING}/capture", json={"reason": reason, "robot": robot},
                         headers={"x-catbox-key": secret("internal-key")})
    except Exception as e:  # sidecar down or camera offline: the visit still logs, just without photos
        log.warning("ring capture %s/%s failed: %s", robot, reason, e)


async def whisker_loop(app):
    w: Whisker = app["whisker"]
    backoff = 60
    while True:
        if w.account is None and not TOKEN_FILE.exists():
            log.warning("no Whisker sign-in yet: run mint-whisker-token.py, then this picks it up within a minute")
            app["health"]["whisker"] = {"ok": False, "at": int(time.time()), "error": "not signed in to Whisker yet"}
            await asyncio.sleep(60)
            continue
        try:
            if w.account is None:
                await w.connect()
            await w.sync()
            app["health"]["whisker"] = {"ok": True, "at": int(time.time())}
            backoff = 60
        except Exception as e:
            log.exception("whisker sync failed")
            app["health"]["whisker"] = {"ok": False, "at": int(time.time()), "error": str(e)[:200]}
            if w.account is not None:
                try:
                    await w.account.disconnect()
                except Exception:
                    pass
                w.account = None
            backoff = min(backoff * 2, 900)
        await asyncio.sleep(backoff)


# ---------- photo colour features ----------

def analyze_frames(folder: Path, frames: list[dict]) -> tuple[dict | None, dict]:
    """Foreground = pixels that differ from the clip's median frame (the empty room, for a fixed camera).

    Returns (look, areas): look = colour of the frame with the biggest foreground blob, areas = per-frame
    foreground fraction, which is the "is the cat in this frame" score. (ffmpeg's scene score only sees cuts;
    a cat walking at a steady pace scores 0 there.)"""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None, {}
    names, imgs = [], []
    for f in (frames[2:] if len(frames) > 8 else frames):  # first ~2 s: live view exposure still settling
        try:
            im = Image.open(folder / f["f"]).convert("RGB")
            im.thumbnail((480, 480))   # a visit recording runs up to 7 min: score every frame, small
            imgs.append(np.asarray(im, dtype=np.uint8))
            names.append(f["f"])
        except Exception:
            continue
    if len(imgs) < 4:
        return None, {}
    bg = np.median(np.stack(imgs[:: max(1, len(imgs) // 90)]), axis=0)
    stack = imgs
    best, areas = None, {}
    for name, im in zip(names, stack):
        im = im.astype(np.int16)
        mask = np.abs(im - bg).sum(axis=2) > 60
        area = float(mask.mean())
        areas[name] = round(area, 4)
        if area < 0.004 or (best and area <= best["area"]):
            continue
        px = im[mask]
        r, g, b = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
        spread = float(np.abs(im[..., 0] - im[..., 2]).mean())
        best = {"area": round(area, 4), "lum": round(float(0.299 * r + 0.587 * g + 0.114 * b), 1),
                "warm": round(float(r - b), 1), "mode": "ir" if spread < 6 else "color", "frame": name}
    return best, areas


# ---------- camera model (GPU worker) ----------

def cat_prompts(db) -> tuple[list[str], list[str]]:
    rows = db.execute("SELECT id, name FROM cats ORDER BY id").fetchall()
    pairs = [(r["id"], CAT_LOOKS[r["name"].strip().lower()]) for r in rows if r["name"].strip().lower() in CAT_LOOKS]
    return [a for a, _ in pairs], [b for _, b in pairs]


VISION_MAX_FRAMES = 200   # vision.py MAX_FRAMES


async def vision_post(app, path: str, body: bytes, prompts: list[str], timeout: int = 240, fps: float = 2) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
        async with s.post(f"{VISION}{path}", data=body, headers={
                "x-catbox-key": app["vision_key"], "x-params": json.dumps({"prompts": prompts, "fps": fps})}) as r:
            r.raise_for_status()
            return await r.json()


async def vision_clip(app, clip_id: int):
    db = app["db"]
    row = db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    mp4 = MEDIA / (row["path"] + ".mp4")
    if not mp4.exists():
        db.execute("UPDATE clips SET vision=? WHERE id=?", (json.dumps({"error": "clip gone"}), clip_id))
        db.commit()
        return
    ids, prompts = cat_prompts(db)
    async with app["vision_sem"]:
        secs = max(1, row["end_ts"] - row["start_ts"])
        res = await vision_post(app, "/analyze", mp4.read_bytes(), prompts,
                                fps=round(min(2, (VISION_MAX_FRAMES - 2) / secs), 2))
    folder = MEDIA / row["path"]
    folder.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(res.get("cats", [])):
        name = f"crop-{k + 1}.jpg"
        (folder / name).write_bytes(base64.b64decode(c.pop("crop")))
        c["crop"] = name
    res["prompt_cats"] = ids
    if row["reason"] == "nap":
        keep_nap_check(db, row, res, folder)
        return
    db.execute("UPDATE clips SET vision=? WHERE id=?", (json.dumps(res), clip_id))
    db.commit()
    log.info("vision clip %s: %s cat frames of %s, %s crops, %s ms", clip_id, res.get("cat_frames"),
             res.get("frames"), len(res.get("cats", [])), res.get("ms"))


def keep_nap_check(db, row, res: dict, folder: Path):
    """A nap check is a 6 s look at a room. Keep the pet crops only; if a person is in view keep nothing
    but the fact that the check happened. The clip and its frames are deleted either way."""
    animals = []
    if not res.get("persons"):
        day = MEDIA / "naps" / row["path"].split("/")[0]
        day.mkdir(parents=True, exist_ok=True)
        base = row["path"].split("/")[-1]
        for k in res.get("cats", []):
            dest = day / f"{base}-{k['crop']}"
            (folder / k["crop"]).replace(dest)
            animals.append({**k, "crop": str(dest.relative_to(MEDIA))})
    db.execute("INSERT INTO nap_checks(cam, ts, persons, animals) VALUES(?,?,?,?)",
               (row["cam"], row["start_ts"], 1 if res.get("persons") else 0, json.dumps(animals)))
    drop_clip(db, row)
    db.commit()
    log.info("nap check %s: %s animals, person in view: %s", row["cam"], len(animals), bool(res.get("persons")))


IR_CACHE: dict = {}


def crop_ir(k: dict, path: Path) -> bool:
    """Night vision: Ring's infrared picture is pure grey (colour spread 0), while a dim lamp-lit room still
    reads 5+. vision.py sets "ir" on new crops; older crops are measured from their file once."""
    if "ir" in k:
        return bool(k["ir"])
    key = str(path)
    if key not in IR_CACHE:
        try:
            import numpy as np
            from PIL import Image
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
            IR_CACHE[key] = float(np.abs(a[..., 0] - a[..., 2]).mean()) < 2
        except Exception:
            IR_CACHE[key] = False
    return IR_CACHE[key]


WARM_CACHE: dict = {}
ORANGE_AT = 14.5   # coat warmth (red minus blue, middle of the crop) between Syd (16-18) and every tabby (13 or less), 2026-10-09


def crop_warm(path: Path) -> float | None:
    """How orange a colour crop is: mean red minus blue over the middle 60% (where the cat is). None in night vision."""
    key = str(path)
    if key not in WARM_CACHE:
        try:
            import numpy as np
            from PIL import Image
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            h, w = a.shape[:2]
            m = a[int(h * .2):int(h * .8), int(w * .2):int(w * .8)].reshape(-1, 3)
            d = m[:, 0] - m[:, 2]
            WARM_CACHE[key] = None if float(np.abs(d).mean()) < 2 else float(d.mean())
        except Exception:
            WARM_CACHE[key] = None
    return WARM_CACHE[key]


LIGHT_AT = 95      # night vision, room-distance cameras (ir_coat_at): median grey of the middle of the crop.
                   # Orange fur shows up near-white in infrared: upstairs Syd read 122-147, every tabby crop 26-71 (2026-10-11).
                   # Close up, the infrared lamp lights tabbies to 110-136 (downstairs), so it is per camera.
LIGHT_CACHE: dict = {}


def crop_light(path: Path) -> float | None:
    """How light a night-vision crop is: median grey over the middle 60% (where the cat is)."""
    key = str(path)
    if key not in LIGHT_CACHE:
        try:
            import numpy as np
            from PIL import Image
            a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
            h, w = a.shape[:2]
            LIGHT_CACHE[key] = float(np.median(a[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]))
        except Exception:
            LIGHT_CACHE[key] = None
    return LIGHT_CACHE[key]


def box_windows(v) -> list[tuple[int, int]]:
    """When a camera sees the visiting cat: walking up (45 s before the box logs it in, to 10 s after) and walking
    out (10 s before the box logs the exit, to 45 s after). A cat in view in between is another one walking past
    (one cat crossed the room 100 s into another's visit). The downstairs camera catches the walk-up
    (every known downstairs crop through 10/11 was 2 to 42 s before the visit); the upstairs one, which records
    only from the late cat-in report, catches the exit."""
    return [(v["start_ts"] - 45, v["start_ts"] + 10), (v["end_ts"] - 10, v["end_ts"] + 45)]


def box_crops(crops: list[dict], v) -> list[dict]:
    wins = box_windows(v)
    return [c for c in crops if any(lo <= c["ts"] <= hi for lo, hi in wins)]


def coat_score(c: dict) -> float | None:
    """Above 0 = the orange cat's coat, below 0 = a tabby's: colour by warmth, night vision by lightness."""
    if c.get("warm") is not None:
        return c["warm"] - ORANGE_AT
    if c.get("light") is not None:
        return (c["light"] - LIGHT_AT) / 10
    return None


def orange_cats(db) -> set:
    ids, looks = cat_prompts(db)
    return {i for i, l in zip(ids, looks) if "orange" in l or "ginger" in l}


def coat_probs(crops: list[dict], cat_ids: list[str], orange: set, check: dict | None = None, v=None) -> dict | None:
    """Field note (2026-10-09): the camera can tell the orange cat from a tabby upstairs. It can, by coat: an orange coat is the
    orange cat, any other coat is one of the tabbies. In colour by warmth, in night vision by lightness (cameras
    with "ir_coat"); each mode is off if its own self-test fails. For a visit, only the cat walking up or out counts."""
    if not orange:
        return None
    off = {m for m in ("colour", "night") if ((check or {}).get(m) or {}).get("ok") is False}
    ws = [x for c in (box_crops(crops, v) if v else crops)
          if (x := coat_score(c)) is not None and ("colour" if c.get("warm") is not None else "night") not in off]
    if not ws:
        return None
    p = 1 / (1 + math.exp(-(sum(ws) / len(ws))))
    p = min(max(p, 0.02), 0.98)
    return {c: (p if c in orange else 1 - p) for c in cat_ids}


def coat_check(rows: list[dict], crops: dict, auto: dict, orange: set) -> dict:
    """Self-test for the coat rule, per mode: on crops of the cat walking up to or out of visits whose cat is known (tapped,
    or certain without the camera), does "orange coat" pick out the orange cat? Once both sides have 2+ clips,
    under 90% switches that mode off."""
    out = {}
    for mode in ("colour", "night"):
        clips, right, n = {True: set(), False: set()}, 0, 0
        for v in rows:
            cat = v["label_cat"] or (auto.get(v["id"]) or (None,))[0]
            if not cat or cat in ("none", "unknown"):
                continue
            for c in box_crops(crops.get(v["id"], []), v):
                x = coat_score(c)
                if x is None or (c.get("warm") is not None) != (mode == "colour"):
                    continue
                truth = cat in orange
                clips[truth].add(c["trk"][0])
                n += 1
                right += (x >= 0) == truth
        m = {"orange_clips": len(clips[True]), "other_clips": len(clips[False]), "crops": n,
             "acc": round(right / n, 3) if n else None, "ok": None}
        if len(clips[True]) >= 2 and len(clips[False]) >= 2:
            m["ok"] = m["acc"] >= CHECK_MIN_ACC
        out[mode] = m
    return out


def merge_probs(*ps) -> dict | None:
    ps = [p for p in ps if p]
    if not ps:
        return None
    out = {c: 1.0 for c in ps[0]}
    for p in ps:
        for c in out:
            out[c] *= max(p.get(c, 1e-6), 1e-6)
    z = sum(out.values()) or 1.0
    return {c: v / z for c, v in out.items()}


def who_is(vm, k: dict, cat_ids: list[str], ids: list[str], ir: bool = False) -> str | None:
    """Name one crop: a dog is the household dog (pets.json); a cat goes through the camera model (None when unsure)."""
    if k.get("species") == "dog":
        return DOG["id"] if DOG else None
    cr = {"vec": emb_vec(k["emb"]), "text": dict(zip(ids, k.get("text") or [])), "ir": ir, "warm": k.get("_warm"),
          "light": k.get("_light") if ir else None}
    vp = merge_probs(vision_probs(vm, [cr], cat_ids), coat_probs([cr], cat_ids, vm.get("orange", set()), vm.get("coat")))
    if not vp:
        return None
    # who uses this camera's box: upstairs a tabby is almost always Buggs (Rocky makes 7% of upstairs visits)
    mix = (vm.get("mix") or {}).get(k.get("_robot")) or {}
    if mix:
        vp = merge_probs(vp, {c: mix.get(c, 0.01) for c in vp})
    best = max(vp.items(), key=lambda x: x[1], default=(None, 0))
    return best[0] if best[1] >= 0.6 else None


def nap_ok(k: dict) -> bool:
    """A nap needs a clear, whole animal: a weak detection (a shadow, a blanket fold) or one cut off by the edge of
    the picture is not enough to name. Field note (2026-10-09): two looks at a shadow on the bed showed as Rocky napping."""
    x0, y0, x1, y1 = k.get("box") or (0, 0, 1, 1)
    return k.get("score", 0) >= 0.45 and min(x0, y0) > 0.01 and max(x1, y1) < 0.985


def nap_who(vm, k: dict, cat_ids: list[str], ids: list[str]) -> str | None:
    return who_is(vm, k, cat_ids, ids, crop_ir(k, MEDIA / k["crop"])) if nap_ok(k) else None


def build_naps(db, vm, cat_ids: list[str], interval_min: int, now: int) -> dict:
    """Same animal at about the same spot on consecutive checks = a nap. Checks with a person in view count as
    neither present nor absent; a gap of more than 3 intervals ends every nap."""
    ids, _ = cat_prompts(db)
    checks = [dict(r) for r in db.execute("SELECT * FROM nap_checks WHERE ts>? ORDER BY ts", (now - 36 * 3600,))]
    step = interval_min * 60
    runs: dict[str, list] = {}
    open_run: dict[str, dict] = {}
    last_ts = None
    for c in checks:
        if last_ts is not None and c["ts"] - last_ts > 3 * step:
            for who in list(open_run):
                runs.setdefault(who, []).append(open_run.pop(who))
        last_ts = c["ts"]
        if c["persons"]:
            continue
        seen = {}
        for k in json.loads(c["animals"] or "[]"):
            who = nap_who(vm, k, cat_ids, ids)
            if who and (who not in seen or k["score"] > seen[who]["score"]):
                seen[who] = k
        for who, k in seen.items():
            cx, cy = (k["box"][0] + k["box"][2]) / 2, (k["box"][1] + k["box"][3]) / 2
            r = open_run.get(who)
            if r and math.hypot(cx - r["cx"], cy - r["cy"]) > 0.2:  # moved across the room: a new nap
                runs.setdefault(who, []).append(open_run.pop(who))
                r = None
            if r is None:
                r = open_run[who] = {"start": c["ts"], "end": c["ts"], "n": 0, "best": k}
            r.update(end=c["ts"], cx=cx, cy=cy, n=r["n"] + 1)
            if k["score"] > r["best"]["score"]:
                r["best"] = k
        for who in [w for w in open_run if w not in seen]:
            runs.setdefault(who, []).append(open_run.pop(who))
    midnight = int(datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    out = {}
    for who in set(runs) | set(open_run):
        naps = [dict(r, live=False) for r in runs.get(who, [])]
        if who in open_run:
            naps.append(dict(open_run[who], live=now - open_run[who]["end"] <= 1.6 * step))
        naps = [n for n in naps if n["n"] >= 2]   # one sighting is a visit, not a nap
        for n in naps:
            n["min"] = round((n["end"] - n["start"]) / 60 + interval_min / 2)
        if naps:
            out[who] = {"naps": [{"start": n["start"], "end": n["end"], "min": n["min"], "live": n["live"],
                                  "crop": f"/media/{n['best']['crop']}"} for n in naps[-12:]],
                        "today_min": sum(n["min"] for n in naps if n["end"] >= midnight),
                        "napping_now": any(n["live"] for n in naps)}
    return {"interval_min": interval_min, "checks": len(checks), "last_check": checks[-1]["ts"] if checks else None,
            "who": out, "dog": DOG}


def build_sightings(db, vm, cat_ids: list[str], clips: list[dict], visit_rows: list[dict],
                    robots: dict, now: int) -> list[dict]:
    """Pets on camera that weren't a litter-box visit (e.g. the dog wandering past)."""
    ids, _ = cat_prompts(db)
    # Whisker logs a visit only after the cat leaves (and the box may still be cycling): until then a cat on that
    # box's camera is the visit in progress. Shown under a guessed name here, it moved to the visit minutes later.
    going = {k for k, st in robots.items() if st.get("cat_inside") or st.get("timing") or st.get("busy")}
    out = []
    for c in clips:
        v = json.loads(c["vision"]) if c.get("vision") else None
        if not v or not v.get("cats") or c["reason"] == "nap":
            continue
        if any(visit_clips([c], vr) for vr in visit_rows):
            continue
        if CAM_ROBOT.get(c["cam"]) in going and c["end_ts"] > now - 1800:
            continue
        tracks = {}
        for k in v["cats"]:
            tracks.setdefault(k.get("track", 0), []).append(k)
        for ks in tracks.values():
            k = max(ks, key=lambda x: x["score"])
            out.append({"ts": c["start_ts"] + k["t"], "cam": CAMS.get(c["cam"], {}).get("label", c["cam"]),
                        "who": who_is(vm, {**k, "_warm": crop_warm(MEDIA / c["path"] / k["crop"]), "_robot": CAM_ROBOT.get(c["cam"]),
                                           "_light": crop_light(MEDIA / c["path"] / k["crop"]) if ir_coat_at(c["cam"], c["start_ts"]) else None},
                                      cat_ids, ids, crop_ir(k, MEDIA / c["path"] / k["crop"])),
                        "species": k.get("species", "cat"), "score": k["score"],
                        "crop": f"/media/{c['path']}/{k['crop']}",
                        "clip": video_url(c["path"])})
    return condense_sightings(out)[:16]


def condense_sightings(out: list[dict], gap: int = 900) -> list[dict]:
    """One entry per stretch of the same pet on the same camera (2026-10-09): Finn walking about showed as
    three entries minutes apart. A stretch ends after `gap` seconds without that pet; the clearest crop stands
    for it, and start/end/n say how long and how many sightings."""
    runs: list[dict] = []
    for x in sorted(out, key=lambda x: x["ts"]):
        key = (x["cam"], x["who"] or x["species"])
        r = next((r for r in reversed(runs) if r["key"] == key and x["ts"] - r["end"] <= gap), None)
        if r is None:
            runs.append({**x, "key": key, "start": x["ts"], "end": x["ts"], "n": 1})
            continue
        r["end"], r["n"] = x["ts"], r["n"] + 1
        if x["score"] > r["score"]:
            r.update(crop=x["crop"], clip=x["clip"] or r["clip"], score=x["score"], ts=x["ts"])
    return [{k: v for k, v in r.items() if k not in ("key", "score")} for r in sorted(runs, key=lambda r: -r["end"])]


async def vision_protos(app):
    """Reference embeddings from the Whisker profile photos (the starting point before anyone taps)."""
    db = app["db"]
    protos = kv_get(db, "protos", {}) or {}
    ids, prompts = cat_prompts(db)
    changed = False
    for cid in ids:
        f = DATA / "cats" / f"{cid}.jpg"
        if not f.exists() or protos.get(cid, {}).get("mtime") == int(f.stat().st_mtime):
            continue
        res = await vision_post(app, "/embed", f.read_bytes(), prompts, timeout=180)
        if res.get("cats"):
            protos[cid] = {"mtime": int(f.stat().st_mtime), "emb": res["cats"][0]["emb"]}
            changed = True
    if changed:
        kv_set(db, "protos", protos)


async def vision_loop(app):
    """Every clip gets one camera-model pass; retried while the GPU worker is busy or rebooting, for up to 2 days."""
    tries: dict[int, int] = {}
    while True:
        try:
            await vision_protos(app)
            db = app["db"]
            todo = [r["id"] for r in db.execute(
                "SELECT id FROM clips WHERE vision IS NULL AND start_ts>? ORDER BY start_ts", (time.time() - 2 * 86400,))]
            for cid in todo:
                try:
                    await vision_clip(app, cid)
                    app["health"]["vision"] = {"ok": True, "at": int(time.time())}
                except Exception as e:
                    tries[cid] = tries.get(cid, 0) + 1
                    app["health"]["vision"] = {"ok": False, "at": int(time.time()), "error": str(e)[:160]}
                    log.warning("vision clip %s failed (%s): %s", cid, tries[cid], e)
                    if tries[cid] >= 20:
                        db.execute("UPDATE clips SET vision=? WHERE id=?", (json.dumps({"error": str(e)[:160]}), cid))
                        db.commit()
                    break  # the GPU worker is down or busy: wait for the next round
        except Exception as e:
            app["health"]["vision"] = {"ok": False, "at": int(time.time()), "error": str(e)[:160]}
            log.warning("vision loop: %s", e)
        try:
            await asyncio.wait_for(app["vision_kick"].wait(), 30)  # a new clip wakes the queue at once
        except asyncio.TimeoutError:
            pass
        app["vision_kick"].clear()


def emb_vec(b64: str):
    import numpy as np
    return np.frombuffer(base64.b64decode(b64), dtype=np.float16).astype("float32")


def visit_crops(cl: list[dict], v) -> list[dict]:
    lo, hi = visit_window(v)
    out = []
    for c in cl:
        v = json.loads(c["vision"]) if c.get("vision") else None
        if not v or not v.get("cats"):
            continue
        ids = v.get("prompt_cats") or []
        for k in v["cats"]:
            if k.get("species", "cat") != "cat":   # a dog never uses a litter box
                continue
            if k.get("score", 0) < 0.35 and (k.get("cat_p") or 0) < 0.6:  # weak YOLO box that SigLIP didn't confirm
                continue
            if not lo <= c["start_ts"] + k["t"] <= hi:
                continue
            out.append({"ts": c["start_ts"] + k["t"], "score": k["score"], "url": f"/media/{c['path']}/{k['crop']}",
                        "vec": emb_vec(k["emb"]), "text": dict(zip(ids, k.get("text") or [])),
                        "ir": (ir := crop_ir(k, MEDIA / c["path"] / k["crop"])), "trk": (c["id"], k.get("track", 0)),
                        "warm": crop_warm(MEDIA / c["path"] / k["crop"]),
                        "light": crop_light(MEDIA / c["path"] / k["crop"]) if ir and ir_coat_at(c["cam"], c["start_ts"]) else None})
    return out


def softmax(d: dict) -> dict:
    if not d:
        return {}
    m = max(d.values())
    e = {k: math.exp(v - m) for k, v in d.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


# Cat photos from the family's photo library (bench/photolib.py), named once by the owner. Only their
# fingerprints and his labels stay (data/photos/groups.json + kv photo_labels); the photos themselves were deleted
# and are never served, because the page is shared with family (the owner's photos stay private).
PHOTOS = DATA / "photos"
PHOTO_CACHE: dict = {}


def photo_groups() -> list[dict]:
    f = PHOTOS / "groups.json"
    if not f.exists():
        return []
    m = f.stat().st_mtime
    if PHOTO_CACHE.get("m") != m:
        PHOTO_CACHE.update(m=m, g=json.loads(f.read_text()))
    return PHOTO_CACHE["g"]


def photo_refs(db) -> dict:
    """{cat: [(vec, ir, src)]} from photo groups the owner named: each crop in colour (src "photo:<group>", so the night
    self-test never scores a photo, only real camera crops)."""
    labels = kv_get(db, "photo_labels", {}) or {}
    out = {}
    for g in photo_groups():
        lab = labels.get(g["id"]) or {}
        cat, per = lab.get("cat"), lab.get("per") or {}
        if not cat or cat in ("none", "other", "skip", "together"):
            continue
        # one name on a group whose photo shows two cats means the prominent one: keep only the
        # biggest crop of each photo
        area = lambda c: (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) if c.get("box") else 0
        biggest = {}
        for c in g["crops"]:
            if c.get("uuid") and area(c) > area(biggest.get(c["uuid"], {})):
                biggest[c["uuid"]] = c
        for c in g["crops"]:
            if cat != "multi" and c.get("uuid") and biggest.get(c["uuid"]) is not c:
                continue
            if cat == "multi":                      # several cats in one group: each photo was named on its own
                cat_c = per.get(c["file"])
                if not cat_c or cat_c in ("none", "skip", "other"):
                    continue
            else:
                cat_c = cat
            out.setdefault(cat_c, []).append((emb_vec(c["emb"]), False, f"photo:{g['id']}"))
            # The grey copies (emb_gray) are NOT night references: measured 2026-10-09, they named the real
            # night-vision camera crops right 4 times in 13, worse than a coin flip. A phone photo in grey is not
            # infrared, so they would only drag the night self-test down. Kept in groups.json for a later test.
    return out


CHECK_MIN_CLIPS = 3     # camera references from at least this many clips per cat before the camera's vote is tested
CHECK_MIN_ACC = 0.9     # ...and they must name each other's clips right this often (leave one clip out)


CERTAIN_AT = 0.99    # without the camera, this sure teaches the camera. Measured 2026-10-11 on the 43 tapped visits,
                     # each scored with its own tap left out: all 12 at 99% or more agreed with the tap (the one miss
                     # was at 94%). It was 99.9% before readings got a fat tail, which no Syd visit could reach.


def certain_visits(cats: list[dict], rows: list[dict], crops: dict, models: dict) -> dict:
    """Visits whose cat is CERTAIN_AT sure without the camera (weight, box, house rules) and with one animal on
    camera: {visit: (cat, windows)}, where windows (box_windows) keep only the cat walking up or out (mid-visit,
    another cat can be waiting its turn). These teach the camera; they never count as taps."""
    auto = {}
    for v in rows:
        cr = crops[v["id"]]
        if v["label_cat"] or not cr or v["weight"] is None:
            continue
        per_clip = {}
        for c in cr:
            per_clip.setdefault(c["trk"][0], set()).add(c["trk"][1])
        if any(len(t) > 1 for t in per_clip.values()):   # two animals on camera: which one used the box?
            continue
        hp = house_prior(cats, v["robot"], v["end_ts"])
        p0, _ = posterior(models, v["weight"], None, None, hp if hp[0] else None, v["robot"], v["duration"])
        if p0 and (top := max(p0.items(), key=lambda kv: kv[1]))[1] >= CERTAIN_AT:
            auto[v["id"]] = (top[0], box_windows(v))
    return auto


def camera_reference_crops(db, now: int, ir: bool, days: int = 30) -> list[dict]:
    """Every camera crop of a known cat in one mode (night vision or colour), with its cat and clip, from taps and
    visits certain without the camera: for comparing embedders offline (bench/nightcompare.py). Same sources and
    windows as build_vision_model."""
    cats = [dict(r) for r in db.execute("SELECT * FROM cats ORDER BY weight DESC")]
    names = {c["id"]: c["name"].strip().lower() for c in cats}
    rows = [dict(r) for r in db.execute("SELECT * FROM visits WHERE end_ts>?", (now - days * 86400,))]
    clips = [dict(r) for r in db.execute("SELECT * FROM clips WHERE end_ts>?", (now - days * 86400 - 600,))]
    looks, crops = {}, {}
    for v in rows:
        cl = visit_clips(clips, v)
        looks[v["id"]] = max((json.loads(c["look"]) for c in cl if c["look"]), key=lambda l: l["area"], default=None)
        crops[v["id"]] = visit_crops(cl, v)
    known = [dict(r) for r in db.execute("SELECT * FROM visits WHERE end_ts>? AND weight IS NOT NULL", (now - 60 * 86400,))]
    auto = certain_visits(cats, rows, crops, build_models(db, cats, looks, known))
    out = []
    for v in rows:
        cat, wins = v["label_cat"], None
        if not cat and v["id"] in auto:
            cat, wins = auto[v["id"]]
        if cat not in names:
            continue
        for cr in crops[v["id"]]:
            if bool(cr.get("ir")) == ir and (wins is None or any(lo <= cr["ts"] <= hi for lo, hi in wins)):
                out.append({"cat": names[cat], "clip": cr["trk"][0], "url": cr["url"], "visit": v["id"]})
    return out


def build_vision_model(db, cats: list[dict], crops_by_visit: dict, auto: dict | None = None) -> dict:
    """Reference crops per cat, each tagged colour or night vision (a night crop is only compared with night crops).
    Sources: the Whisker profile photos, hand-named crops (kv "refs"), crops from tapped visits, and from visits whose
    cat is certain without the camera (auto: {visit: (cat, windows)}, only the crops of the cat walking up or out),
    which never count as taps."""
    protos = {c["id"]: [] for c in cats}
    for cid, pr in (kv_get(db, "protos", {}) or {}).items():
        if cid in protos:
            protos[cid].append((emb_vec(pr["emb"]), False, None))
    labeled = {r["id"]: r["label_cat"] for r in db.execute("SELECT id, label_cat FROM visits WHERE label_cat IS NOT NULL")}
    n_labeled = {False: {c: 0 for c in protos}, True: {c: 0 for c in protos}}
    # confirmed crops that belong to no visit (kv "refs", e.g. a cat at the window, named by hand); all colour so far
    for cid, embs in (kv_get(db, "refs", {}) or {}).items():
        if cid in protos:
            for e in embs:
                ir = bool(e.get("ir")) if isinstance(e, dict) else False
                protos[cid].append((emb_vec(e["emb"] if isinstance(e, dict) else e), ir, None))
                n_labeled[ir][cid] += 1
    for cid, refs in photo_refs(db).items():           # named by the family: they count as taps
        if cid in protos:
            protos[cid] += refs
            for _, ir, _ in refs:
                n_labeled[ir][cid] += 1
    for vid, crops in crops_by_visit.items():
        cat, wins, tapped = labeled.get(vid), None, True
        if cat is None and auto and vid in auto:
            (cat, wins), tapped = auto[vid], False
        if cat in protos:
            for cr in crops:
                if wins and not any(lo <= cr["ts"] <= hi for lo, hi in wins):
                    continue
                protos[cat].append((cr["vec"], cr.get("ir", False), cr.get("trk", (None,))[0]))
                if tapped:
                    n_labeled[cr.get("ir", False)][cat] += 1
    checks = {ir: camera_check(protos, ir) for ir in (False, True)}
    return {"protos": protos, "n_labeled": n_labeled, "checks": checks, "night": checks[True], "day": checks[False]}


def camera_check(protos: dict, ir: bool) -> dict:
    """Has the camera earned a vote (in colour, or in night vision)? Name every camera crop of a known cat from the
    other references (its own clip left out; photo-library crops may be references but are never scored) and count
    how often that's right. Until this passes, the camera's naming in that mode is no evidence: measured 2026-10-09,
    Buggs's night crops matched Rocky's as well as her own, and in daylight the camera called Syd walking away
    "Rocky" outright (the coat description picked "large brown tabby" for an orange cat seen from behind)."""
    import numpy as np
    refs = [(cid, v, src) for cid, ps in protos.items() for v, pir, src in ps if pir == ir and src is not None]
    cam = [i for i, r in enumerate(refs) if not str(r[2]).startswith("photo:")]
    clips = {cid: {refs[i][2] for i in cam if refs[i][0] == cid} for cid in protos}
    out = {"clips": {c: len(x) for c, x in clips.items()}, "acc": None, "ok": False,
           "photo_refs": sum(1 for r in refs if str(r[2]).startswith("photo:"))}
    if not cam or min(len(x) for x in clips.values()) < CHECK_MIN_CLIPS or len({len(v) for _, v, _ in refs}) > 1:
        return out
    m = np.stack([v for _, v, _ in refs])
    srcs = [src for _, _, src in refs]
    right = 0
    for i in cam:
        row = m @ m[i]
        row[[j for j, s2 in enumerate(srcs) if s2 == srcs[i]]] = -9
        right += refs[int(row.argmax())][0] == refs[i][0]
    out["acc"] = round(right / len(cam), 3)
    out["ok"] = out["acc"] >= CHECK_MIN_ACC
    return out


def vision_probs(vm: dict, crops: list[dict], cat_ids: list[str]) -> dict | None:
    """Per crop: zero-shot coat description x similarity to that cat's reference crops; then the
    geometric mean over the visit's crops. Once every cat has 2+ tapped crops, references decide alone.
    Night vision has no colour, so a night crop skips the coat description and is compared only with night
    references. A crop counts only once the camera has passed its self-test in that mode (camera_check): better
    unsure than Buggs called Rocky, or Syd called Rocky."""
    if not crops:
        return None
    logp, used = {c: 0.0 for c in cat_ids}, 0
    for cr in crops:
        ir = bool(cr.get("ir"))
        if not vm.get("checks", {}).get(ir, {}).get("ok"):   # the camera hasn't earned a vote in this mode yet
            continue
        trained = all(vm["n_labeled"][ir].get(c, 0) >= 2 for c in cat_ids)
        # references made by a different embedder (the camera model's tier changed) have another size: skip them
        sims = {c: max(ok) for c in cat_ids
                if (ok := [float(cr["vec"] @ p) for p, pir, _ in vm["protos"].get(c) or []
                           if pir == ir and len(p) == len(cr["vec"])])}
        proto = softmax({c: s * 30 for c, s in sims.items()}) if len(sims) == len(cat_ids) else None
        text = softmax({c: cr["text"][c] for c in cat_ids if c in cr["text"]}) if cr["text"] and not ir else None
        if trained and proto:
            p = proto
        elif proto and text:
            p = {c: text.get(c, 1e-6) * proto[c] ** 0.5 for c in cat_ids}
        else:
            p = text or proto or {}
        if not p:
            continue
        used += 1
        z = sum(p.values()) or 1.0
        for c in cat_ids:
            logp[c] += math.log(max(p.get(c, 0) / z, 1e-6))
    return softmax({c: v / used for c, v in logp.items()}) if used else None


# ---------- cat ID ----------

TAP_TRUST = 0.9      # people tap what they're about 90% sure of; a tap is strong evidence, not gospel
ASSIGN_AT = 0.985    # at or above this a visit is assigned to the cat even on overlapping weights (the page says 99%)
ROBOTS = ("evo", "lr4")


def _fit(pts: list[tuple], floor: float) -> dict | None:
    """Weighted mean/sd of (value, weight) pairs, trimmed once at 3 sd so a mis-tap can't drag it."""
    def stats(ps):
        n = sum(w for _, w in ps)
        if n <= 0:
            return None
        m = sum(x * w for x, w in ps) / n
        sd = math.sqrt(sum(w * (x - m) ** 2 for x, w in ps) / n)
        return m, max(sd, floor), n
    s = stats(pts)
    if not s:
        return None
    # trim around the median: one far-off reading inflates the sd enough to survive a 3 sd trim around the mean
    # (Rocky upstairs 13.1-13.4 lb six times, then a tapped 11.9 made his spread 0.45)
    xs = sorted(x for x, _ in pts)
    med = xs[len(xs) // 2]
    mad = sorted(abs(x - med) for x in xs)[len(xs) // 2] * 1.4826
    kept = [(x, w) for x, w in pts if abs(x - med) <= 3 * max(mad, floor)] if len(pts) >= 5 else pts
    m, sd, n = stats(kept) or s
    return {"mean": m, "sd": sd if n >= 5 else max(sd, floor * 1.5), "n": round(n, 1)}


def build_models(db, cats: list[dict], clips_by_visit: dict, rows: list[dict] | None = None, skip: str | None = None) -> dict:
    """Per-box models (two boxes' scales disagree: here the EVO read the same cats 0.3-0.5 lb lighter than
    the LR4), learned only from what is known: taps (at 90%), the move-in date and night routine, and the one
    13 lb cat. `skip` leaves one visit out, to re-check that tap against everything else."""
    now = time.time()
    if rows is None:
        rows = [dict(r) for r in db.execute("SELECT * FROM visits WHERE end_ts>? AND weight IS NOT NULL", (now - 60 * 86400,))]
    ids = [c["id"] for c in cats]
    big = [c["id"] for c in cats if (c["weight"] or 0) >= 12]
    tens = [cid for cid in ids if cid not in big]
    pts: dict[tuple, list] = {}                     # (robot, cat) -> [(weight, duration, trust)]
    used: dict[str, str] = {}                     # visit id -> the cat it is known to be
    for r in rows:
        w = r["weight"]
        if w is None or r["id"] == skip or r["label_cat"] in ("none", "unknown"):
            continue
        if r["label_cat"] in ids:
            cat, trust = r["label_cat"], TAP_TRUST
        elif len(big) == 1 and w >= 12:
            cat, trust = big[0], 1.0                  # nobody else weighs 13 lb
        elif 8 <= w < 12:
            pr, _ = house_prior(cats, r["robot"], r["end_ts"])
            live = [cid for cid in tens if pr and pr.get(cid, 1.0) == 1.0]
            if not (pr and len(live) == 1 and all(pr.get(cid, 1.0) < 0.1 for cid in tens if cid != live[0])):
                continue
            cat, trust = live[0], 1.0                 # before the move / night routine: only one 10 lb cat possible
        else:
            continue
        used[r["id"]] = cat
        pts.setdefault((r["robot"], cat), []).append((w, r["duration"], trust))
    src_of = lambda ps: "taps" if all(t < 1 for *_, t in ps) else "certain visits" if all(t == 1 for *_, t in ps) else "taps + certain visits"
    weight = {rb: {} for rb in ROBOTS}
    for (rb, cid), ps in pts.items():
        # 0.15 lb floor: taps skew to the clear-cut weights, so a tapped cat's own spread reads too tight
        if rb in weight and (m := _fit([(w, t) for w, _, t in ps], 0.15)) and m["n"] >= 3:
            weight[rb][cid] = {**m, "src": src_of(ps)}
    # how differently the two scales read, from cats known for certain on both (taps follow the weight rule people
    # tap by, so they skew to clear-cut readings); taps-only pairs only when there is nothing better
    pair = lambda c: (weight["lr4"][c]["mean"] - weight["evo"][c]["mean"], min(weight["lr4"][c]["n"], weight["evo"][c]["n"]))
    both = [c for c in ids if c in weight["lr4"] and c in weight["evo"]]
    certain = [c for c in both if "certain" in weight["lr4"][c]["src"] and "certain" in weight["evo"][c]["src"]]
    diffs = [pair(c) for c in (certain or both)]
    shift = sum(d * n for d, n in diffs) / sum(n for _, n in diffs) if diffs else 0.0
    # a cat known only by taps on one box but for certain on the other is centred from the other box: Buggs's
    # upstairs taps all followed "under 9.6 is Buggs", so her model sat at 9.55 +-0.15 and called a 9.94 Buggs
    # (seen on camera, 2026-10-10) 99.5% Syd; downstairs she reads 9.98, and the scales differ by 0.31 (Rocky)
    if certain:
        for rb in ROBOTS:
            other = "evo" if rb == "lr4" else "lr4"
            for c in both:
                if weight[rb][c]["src"] == "taps" and "certain" in weight[other][c]["src"]:
                    o = weight[other][c]
                    weight[rb][c] = {**weight[rb][c], "mean": o["mean"] + (shift if rb == "lr4" else -shift),
                                     "sd": max(weight[rb][c]["sd"], o["sd"]),
                                     "src": f"taps, centred from the {HOUSE.get('places', {}).get(other, other).lower()} box"}
    # a cat with too little data on one box: its other box, shifted by how differently the two scales read
    old = HOUSE.get("old_account", {})
    for rb in ROBOTS:
        other = "evo" if rb == "lr4" else "lr4"
        for c in cats:
            cid = c["id"]
            if cid in weight[rb]:
                continue
            if cid in weight[other]:
                o = weight[other][cid]
                weight[rb][cid] = {"mean": o["mean"] + (shift if rb == "lr4" else -shift), "sd": max(o["sd"], 0.25),
                                   "n": 0, "src": f"estimated from the {HOUSE.get('places', {}).get(other, other).lower()} box"}
            elif old.get("robot") == rb and c["name"].strip().lower() in old.get("weights", {}):
                weight[rb][cid] = {"mean": old["weights"][c["name"].strip().lower()], "sd": 0.25, "n": 1, "src": "old account"}
            else:
                weight[rb][cid] = {"mean": c["weight"] or 10.0, "sd": 0.3, "n": 0, "src": "whisker"}
    # time in the box: log seconds per cat per box, only where that cat has 5+ known visits there
    dur = {rb: {} for rb in ROBOTS}
    for rb in ROBOTS:
        allp = [(math.log(d), t) for (r_, _), ps in pts.items() if r_ == rb for _, d, t in ps if d and d > 0]
        if len(allp) >= 10:
            dur[rb]["*"] = _fit(allp, 0.3)
            for cid in ids:
                cp = [(math.log(d), t) for _, d, t in pts.get((rb, cid), []) if d and d > 0]
                if len(cp) >= 5:
                    dur[rb][cid] = _fit(cp, 0.3)
    models = {"weight": weight, "dur": dur, "shift": shift, "mix": {rb: {cid: 1 / len(ids) for cid in ids} for rb in ROBOTS}}
    # how often each cat uses each box, outside the known windows: estimated from the visits themselves (EM)
    free = [r for r in rows if r["id"] not in used and r["id"] != skip and r["label_cat"] not in ("none", "unknown")
            and r["weight"] and 8 <= r["weight"] < 16 and not house_prior(cats, r["robot"], r["end_ts"])[0]]
    known = [r for r in rows if r["id"] in used and not house_prior(cats, r["robot"], r["end_ts"])[0]]
    for rb in ROBOTS:
        fr, kn = [r for r in free if r["robot"] == rb], [used[r["id"]] for r in known if r["robot"] == rb]
        if not fr and not kn:
            continue
        for _ in range(25):
            tot = {cid: 1.0 + kn.count(cid) for cid in ids}
            for r in fr:
                lk = likelihood(models, rb, r["weight"], r["duration"])
                if lk:
                    p = {cid: models["mix"][rb][cid] * lk[cid] for cid in ids}
                    z = sum(p.values()) or 1.0
                    for cid in ids:
                        tot[cid] += p[cid] / z
            z = sum(tot.values())
            models["mix"][rb] = {cid: tot[cid] / z for cid in ids}
    photo = {}
    feats: dict[tuple, list] = {}
    for vid, look in clips_by_visit.items():
        row = next((r for r in rows if r["id"] == vid), None)
        if row and row["label_cat"] and row["id"] != skip and look:
            feats.setdefault((look["mode"], row["label_cat"]), []).append((look["lum"], look["warm"]))
    for (mode, cat), ps in feats.items():
        if len(ps) >= 2:
            photo.setdefault(mode, {})[cat] = (statistics.fmean(p[0] for p in ps), statistics.fmean(p[1] for p in ps), len(ps))
    models["photo"] = photo
    return models


WEIGHT_MISS, WEIGHT_MISS_SD = 0.08, 0.75
# The scale misses now and then: a paw on the step, a cat shifting as it reads. Rocky (the only 13 lb cat, so never
# in doubt) read 13.1-13.8 lb 17 times and 11.9 once (tapped 2026-10-10); Buggs, seen on camera, read 9.94
# upstairs where her other readings sit at 9.4-9.6. So each reading is trusted with a fat tail: 8% of readings may
# be off by about 3/4 lb. Without it one reading between two cats counted as 20:1 and a Buggs visit read 99.5% Syd.


def likelihood(models, robot: str, weight: float | None, duration: int | None) -> dict | None:
    """p(weight, time in box | cat) on that box; None when no cat weighs this (a person at the box)."""
    wm = models["weight"].get(robot) or next(iter(models["weight"].values()))
    if weight is None:
        return {c: 1.0 for c in wm}
    z = {c: (weight - m["mean"]) / m["sd"] for c, m in wm.items()}
    if min(abs(v) for v in z.values()) > 4:
        return None
    p = {c: (1 - WEIGHT_MISS) * math.exp(-0.5 * z[c] ** 2) / m["sd"]
         + WEIGHT_MISS * math.exp(-0.5 * ((weight - m["mean"]) / WEIGHT_MISS_SD) ** 2) / WEIGHT_MISS_SD for c, m in wm.items()}
    dm = models["dur"].get(robot) or {}
    if duration and duration > 0 and "*" in dm:
        x = math.log(duration)
        for c in p:
            d = dm.get(c) or dm["*"]
            p[c] *= (math.exp(-0.5 * ((x - d["mean"]) / d["sd"]) ** 2) / d["sd"]) ** 0.7   # supporting, not deciding
    return p


def posterior(models, weight: float | None, look: dict | None, vprob: dict | None = None,
              prior: dict | None = None, robot: str = "evo", duration: int | None = None,
              seen: bool = False) -> tuple[dict, str]:
    how = []
    lk = likelihood(models, robot, weight, duration)
    odd = lk is None
    if odd and seen and weight is not None:
        # a weight no cat has, but a cat on camera at the box: a paw off the scale, not a person (10/10 9:18 AM
        # upstairs: 11.9 lb, a dark cat walking out). Only a loose lean toward the nearest weights; never "sure".
        p = {c: math.exp(-0.5 * ((weight - m["mean"]) / max(1.0, m["sd"])) ** 2) for c, m in models["weight"][robot].items()}
        how.append("odd weight")
    elif odd:
        p = {c: 1.0 for c in models["weight"][robot]}
    else:
        mix = models["mix"].get(robot, {})
        p = {c: v * mix.get(c, 1.0) for c, v in lk.items()}
        if weight is not None:
            how.append("weight")
            dm = models["dur"].get(robot) or {}
            if duration and sum(1 for c in p if c in dm) >= 2:
                how.append("time in the box")
    if vprob:
        for c in p:
            p[c] *= max(vprob.get(c, 0.0), 1e-6) ** 0.8
        how.append("camera")
    elif look and (cents := models["photo"].get(look["mode"])) and len(cents) >= 2:
        scale = 18.0  # colour units; lum/warm spread inside one cat's clips
        for c in p:
            if c in cents:
                d2 = (look["lum"] - cents[c][0]) ** 2 + (look["warm"] - cents[c][1]) ** 2
                p[c] *= math.exp(-0.5 * d2 / scale ** 2) + 1e-3
        how.append("photo colour")
    if prior and not odd:
        prior, tags = prior
        for c in p:
            p[c] *= prior.get(c, 1.0)
        how += tags
    if odd and not vprob and not seen:
        # A weight no cat has, with no cat on camera: a person at the box (changing the litter,
        # emptying the drawer), not a cat.
        return {}, "odd weight"
    z = sum(p.values()) or 1.0
    return {c: v / z for c, v in p.items()}, "+".join(how) or "nothing"


def why(models, cats: list[dict], v: dict, ranked: list) -> list[str]:
    """Short, plain reasons for a best guess or a double-check, the way you'd say them out loud."""
    if len(ranked) < 2 or v["weight"] is None:
        return []
    nm = {c["id"]: c["name"][:1].upper() + c["name"][1:] for c in cats}
    a, b = ranked[0][0], ranked[1][0]
    wm = models["weight"][v["robot"]]
    rng = lambda c: f"{wm[c]['mean'] - 1.5 * wm[c]['sd']:.1f}-{wm[c]['mean'] + 1.5 * wm[c]['sd']:.1f}"
    out = [f"Weighed {v['weight']:.1f} lb. On this box {nm[a]} is usually {rng(a)} lb, {nm[b]} {rng(b)}."]
    dm = models["dur"].get(v["robot"]) or {}
    if v.get("duration") and a in dm and b in dm:
        x = math.log(v["duration"])
        la = -0.5 * ((x - dm[a]["mean"]) / dm[a]["sd"]) ** 2 - math.log(dm[a]["sd"])
        lb = -0.5 * ((x - dm[b]["mean"]) / dm[b]["sd"]) ** 2 - math.log(dm[b]["sd"])
        if abs(la - lb) > 0.4:
            fav, oth = (a, b) if la > lb else (b, a)
            longer = math.exp(dm[fav]["mean"]) > math.exp(dm[oth]["mean"])
            out.append(f"Stayed {round(v['duration'] / 60, 1):g} min. {nm[fav]} {'usually stays longer' if longer else 'is usually quicker'}.")
    mix = models["mix"].get(v["robot"], {})
    if mix.get(a) and mix.get(b) and (r := max(mix[a], mix[b]) / min(mix[a], mix[b])) > 2.5:
        rare = a if mix[a] < mix[b] else b
        out.append(f"{nm[rare]} {'rarely uses' if r > 6 else 'uses'} this box{'' if r > 6 else ' less often'}.")
    return out


# ---------- web ----------

LOCKED = (STATIC / "login.html")
FAILS: dict[str, list] = {}          # ip -> recent failed family-code tries (rate limit)
PUBLIC = {"/healthz", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png", "/apple-touch-icon.png", "/login"}


def client_ip(request) -> str:
    """The proxy's X-Forwarded-For counts only when the request came from this machine (Tailscale Serve/Funnel,
    a local reverse proxy). From anywhere else it is client-controlled and would let one sender dodge the
    per-IP login limit."""
    remote = request.remote or "?"
    xff = request.headers.get("X-Forwarded-For")
    if xff and remote in ("127.0.0.1", "::1"):
        return xff.split(",")[-1].strip()
    return remote


def set_login_cookie(resp, key: str):
    resp.set_cookie("cb", key, max_age=400 * 86400, httponly=True, secure=True, samesite="Lax")


def norm_code(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


async def login(request: web.Request):
    """Plain URL + the family code word works on any device.
    Case, spaces and punctuation are ignored, so "Blue Fern" and "bluefern " both work.
    8 wrong tries per IP per 15 min, 60 per hour overall, then it waits."""
    now = time.time()
    ip = client_ip(request)
    recent = [t for t in FAILS.get(ip, []) if now - t < 900]
    total = sum(1 for ts in FAILS.values() for t in ts if now - t < 3600)
    if len(recent) >= 8 or total >= 60:
        return web.json_response({"ok": False, "error": "Too many tries. Wait 15 minutes."}, status=429)
    body = await request.json() if request.can_read_body else {}
    code = norm_code(str(body.get("code", "")))
    if code and hmac.compare_digest(code.encode(), norm_code(request.app["family_code"]).encode()):
        resp = web.json_response({"ok": True})
        set_login_cookie(resp, request.app["web_key"])
        log.info("login ok from %s", ip)
        return resp
    FAILS[ip] = recent + [now]
    log.warning("login failed from %s", ip)
    return web.json_response({"ok": False, "error": "That code didn't work."}, status=401)


@web.middleware
async def access_log(request: web.Request, handler):
    status = 500
    try:
        resp = await handler(request)
        status = resp.status
    except web.HTTPException as e:
        status = e.status
        raise
    finally:
        quiet = request.path.startswith("/media/") or request.path in ("/api/state", "/healthz")
        if not quiet or status >= 400:
            ua = request.headers.get("User-Agent", "")
            dev = "iPhone" if "iPhone" in ua else "iPad" if "iPad" in ua else "Android" if "Android" in ua else "Mac" if "Macintosh" in ua else ua[:24]
            log.info("http %s %s %s %s %s cookie=%s", request.method, request.path, status, client_ip(request), dev, "cb" in request.cookies)
    return resp


@web.middleware
async def auth(request: web.Request, handler):
    if request.path in PUBLIC:
        return await handler(request)
    if request.path.startswith("/internal/"):
        if not hmac.compare_digest(request.headers.get("x-catbox-key", ""), request.app["internal_key"]):
            raise web.HTTPForbidden()
        return await handler(request)
    key = request.app["web_key"]
    if hmac.compare_digest(request.query.get("k", ""), key):
        resp = web.HTTPFound("/")
        set_login_cookie(resp, key)
        raise resp
    if not hmac.compare_digest(request.cookies.get("cb", ""), key):
        if request.path.startswith("/api/") or request.path.startswith("/media/"):
            return web.json_response({"error": "locked"}, status=401)
        return web.FileResponse(LOCKED, status=401, headers={"Cache-Control": "no-cache"})
    return await handler(request)


def only_dog(c: dict) -> bool:
    """The camera model saw a pet in this clip and every one was a dog (Finn walking past during a visit)."""
    v = json.loads(c["vision"]) if c.get("vision") else None
    pets = (v or {}).get("cats") or []
    return bool(pets) and all(k.get("species") == "dog" for k in pets)


def ir_coat_at(cam: str, ts: float) -> bool:
    """Does the night coat rule hold for this camera's clip from this time? cams.json "ir_coat": true, or
    "ir_coat_until": "<ISO time>" for a camera since moved up close, where the infrared lamp lights tabbies too
    (a camera moved from across a room to right at the box; up close Rocky's
    crops read as light as Syd's). Its older clips keep the rule."""
    c = CAMS.get(cam, {})
    if c.get("ir_coat_until"):
        return ts < datetime.fromisoformat(c["ir_coat_until"]).timestamp()
    return bool(c.get("ir_coat"))


def video_url(path: str) -> str | None:
    """A clip's video, or None once it's deleted. ?v= is the file's mtime: media is cached as immutable, and a clip
    re-encoded after the fact (damaged frames, 2026-10-12) must not be served from a browser's old copy."""
    try:
        return f"/media/{path}.mp4?v={int((MEDIA / (path + '.mp4')).stat().st_mtime)}"
    except FileNotFoundError:
        return None


def visit_window(v) -> tuple[int, int]:
    """Clips, photos and frames within this window belong to the visit (the cat walking up and walking away)."""
    return v["start_ts"] - 75, v["end_ts"] + 60


def visit_clips(clips: list[dict], v) -> list[dict]:
    lo, hi = visit_window(v)
    return [c for c in clips if c["start_ts"] <= hi and c["end_ts"] >= lo and c.get("reason") != "nap"
            and CAM_ROBOT.get(c["cam"], v["robot"]) == v["robot"] and not only_dog(c)]


def pick_frames(cl: list[dict], v) -> list[dict]:
    """Up to 4 frames: the busiest frame near the entry, the busiest near the exit, then other busy ones."""
    lo, hi = visit_window(v)
    cand, seen = [], set()
    for c in cl:
        pets = (json.loads(c["vision"]) if c.get("vision") else {}).get("cats") or []
        cat_t = {round(k["t"]) for k in pets if k.get("species", "cat") == "cat"}
        dog_t = {round(k["t"]) for k in pets if k.get("species") == "dog"} - cat_t
        for f in json.loads(c["frames"] or "[]"):
            if not lo <= c["start_ts"] + f["t"] <= hi:
                continue
            near_cat = any(abs(f["t"] - t) <= 1 for t in cat_t)
            if (f.get("s", 0) < 0.004 and not near_cat) or (not near_cat and any(abs(f["t"] - t) <= 1 for t in dog_t)):
                continue
            seen.add(near_cat)
            cand.append({"ts": c["start_ts"] + f["t"], "s": f["s"] + (1 if near_cat else 0), "cat": near_cat,
                         "url": f"/media/{c['path']}/{f['f']}"})
    if True in seen:   # the camera model found the cat: show those frames, not the busiest ones
        cand = [f for f in cand if f["cat"]]
    if not cand:
        return []
    chosen = []
    near_in = [f for f in cand if abs(f["ts"] - v["start_ts"]) <= 60]
    near_out = [f for f in cand if abs(f["ts"] - v["end_ts"]) <= 60]
    if near_in:
        chosen.append({**max(near_in, key=lambda f: f["s"]), "kind": "in"})
    if near_out:
        best = max(near_out, key=lambda f: f["s"])
        if not chosen or best["url"] != chosen[0]["url"]:
            chosen.append({**best, "kind": "out"})
    for f in sorted(cand, key=lambda f: -f["s"]):
        if len(chosen) >= 4:
            break
        if all(abs(f["ts"] - c["ts"]) >= 3 for c in chosen):
            chosen.append({**f, "kind": ""})
    return sorted(chosen, key=lambda f: f["ts"])


TAP_CACHE: dict = {}


def tap_checks(app, db, cats, looks, rows, vm, crops, cat_ids) -> dict:
    """Re-check every tap against a model built without it. Taps are ~90% sure, so the ones
    everything else disagrees with get a second look. Cached until a visit or a tap changes."""
    sig = hash((len(rows), max((r["end_ts"] for r in rows), default=0),
                tuple(sorted((r["id"], r["label_cat"] or "") for r in rows if r["label_cat"]))))
    if (c := TAP_CACHE.get("v")) and c[0] == sig:
        return c[1]
    out = {}
    for r in rows:
        if r["label_cat"] not in cat_ids:
            continue
        cam = merge_probs(vision_probs(vm, crops.get(r["id"], []), cat_ids),
                          coat_probs(crops.get(r["id"], []), cat_ids, vm.get("orange", set()), vm.get("coat"), r))
        m = build_models(db, cats, looks, rows, skip=r["id"])
        hp = house_prior(cats, r["robot"], r["end_ts"])
        p, how = posterior(m, r["weight"], looks.get(r["id"]), cam, hp if hp[0] else None, r["robot"], r["duration"])
        ranked = sorted(p.items(), key=lambda kv: -kv[1])
        if ranked and ranked[0][0] != r["label_cat"] and p.get(r["label_cat"], 0) < 0.25:
            if cam and cam.get(r["label_cat"], 0) > 2 * cam.get(ranked[0][0], 0):
                continue      # the camera saw the tapped cat walk out, not the scale's pick: a person and the camera beat the scale
            out[r["id"]] = {"cat": ranked[0][0], "p": round(ranked[0][1], 3), "how": how, "why": why(m, cats, r, ranked)}
    TAP_CACHE["v"] = (sig, out)
    return out


async def api_state(request: web.Request):
    db = request.app["db"]
    days = min(int(request.query.get("days", "14")), 120)
    now = int(time.time())
    since = now - days * 86400
    cats = [dict(r) for r in db.execute("SELECT * FROM cats ORDER BY weight DESC")]
    for c in cats:
        c["name"] = c["name"][:1].upper() + c["name"][1:]
        key = c["name"].lower()
        c["color"] = CAT_COLORS.get(key) or PALETTE[cats.index(c) % len(PALETTE)]
        c["photo"] = f"/cat/{c['id']}.jpg"
    rows = [dict(r) for r in db.execute("SELECT * FROM visits WHERE end_ts>? ORDER BY end_ts DESC", (since,))]
    clips = [dict(r) for r in db.execute("SELECT * FROM clips WHERE end_ts>? ORDER BY start_ts", (since - 600,))]
    looks, vclips, crops = {}, {}, {}
    for v in rows:
        cl = visit_clips(clips, v)
        vclips[v["id"]] = cl
        best = max((json.loads(c["look"]) for c in cl if c["look"]), key=lambda l: l["area"], default=None)
        looks[v["id"]] = best
        crops[v["id"]] = visit_crops(cl, v)
    known_rows = [dict(r) for r in db.execute("SELECT * FROM visits WHERE end_ts>? AND weight IS NOT NULL", (now - 60 * 86400,))]
    models = build_models(db, cats, looks, known_rows)
    auto = certain_visits(cats, rows, crops, models)
    vm = build_vision_model(db, cats, crops, auto)
    vm["orange"], vm["mix"] = orange_cats(db), models["mix"]
    vm["coat"] = coat_check(rows, crops, auto, vm["orange"])
    cat_ids = [c["id"] for c in cats]
    checks = tap_checks(request.app, db, cats, looks, known_rows, vm, crops, cat_ids)
    kept = kv_get(db, "kept", {}) or {}
    reviewed = set(kv_get(db, "reviewed", []) or [])   # unsorted visits someone already looked at: stop asking
    out, omitted = [], []
    for v in rows:
        hp = house_prior(cats, v["robot"], v["end_ts"])
        p, how = posterior(models, v["weight"], looks[v["id"]], merge_probs(vision_probs(vm, crops[v["id"]], cat_ids),
                           coat_probs(crops[v["id"]], cat_ids, vm["orange"], vm["coat"], v)),
                           hp if hp[0] else None, v["robot"], v["duration"], seen=bool(crops[v["id"]]))
        ranked = sorted(p.items(), key=lambda kv: -kv[1])
        if v["label_cat"] == "unknown":
            continue                                          # someone tapped "Don't know": drop it everywhere
        if v["label_cat"] == "none":
            cat, conf, state = None, 1.0, "not_cat"           # a person tapped "Not a cat"
        elif v["label_cat"]:
            cat, conf, state = v["label_cat"], 1.0, "confirmed"
        elif not ranked:
            cat, conf, state = None, 0.0, "not_cat"           # odd weight, nothing on camera
        else:
            cat, conf = ranked[0]
            state = "sure" if conf >= 0.9 and "odd weight" not in how else "unsure"
            # A 99% call is assigned, not asked. A wrong one is one tap to fix under its photo,
            # which is less friction than sorting every near-certain visit by hand. (0.985 is what the page shows as 99%.)
            if state == "sure" and conf < ASSIGN_AT and len(ranked) > 1 and not any(t in how for t in ("camera", "before the move", "night routine")):
                # weight (and time in the box) alone may only separate cats far apart on that box (a 13 lb cat vs
                # two 10 lb cats), or two cats both known on that box from taps/certain visits whose means sit 3+
                # spreads apart (e.g. 9.5 vs 10.2 on one box). Anything closer is a best guess, never "sure".
                wm = models["weight"][v["robot"]]
                a_m, b_m = wm[ranked[0][0]], wm[ranked[1][0]]
                gap = abs(a_m["mean"] - b_m["mean"])
                known = a_m["n"] >= 3 and b_m["n"] >= 3
                if gap < 2 and not (known and gap >= 3 * max(a_m["sd"], b_m["sd"])):
                    state = "unsure"
        if state == "not_cat":
            continue
        cl = vclips[v["id"]]
        frames, vcrops = pick_frames(cl, v), crops[v["id"]]
        if state == "unsure" and not frames and not vcrops:
            # a toss-up with no photo: listed for review as a question, never under a guessed name;
            # the absence monitor still counts it for both candidates
            omitted.append({"id": v["id"], "start": v["start_ts"], "end": v["end_ts"], "robot": v["robot"],
                            "weight": v["weight"], "duration": v["duration"], "state": "unsure", "cat": None, "how": how,
                            "options": [{"cat": c, "p": round(x, 3)} for c, x in ranked[:2]],
                            "why": why(models, cats, v, ranked), "reviewed": v["id"] in reviewed})
            continue
        # the clips with this cat in them, in time order (the walk in and the walk out can be two clips: a
        # motion clip from the camera itself starts about a minute before Whisker reports the cat in); with no
        # cat on camera, the visit recording still shows, so the family can see the camera missed it
        saved = [c for c in cl if video_url(c["path"])]
        first = {}                                    # clip id -> when the cat first shows in it
        for cr in vcrops:
            first[cr["trk"][0]] = min(first.get(cr["trk"][0], cr["ts"]), cr["ts"])
        # each video opens 3 s before the cat shows: a visit recording starts with a minute of empty room
        vids = [video_url(c["path"]) + (f"#t={max(0, int(first[c['id']] - c['start_ts']) - 3)}" if c["id"] in first else "")
                for c in ([c for c in saved if c["id"] in first] or saved[:1])]
        out.append({"id": v["id"], "robot": v["robot"], "start": v["start_ts"], "end": v["end_ts"],
                    "weight": v["weight"], "duration": v["duration"], "cat": cat, "conf": round(conf, 3),
                    "state": state, "how": how, "options": [{"cat": c, "p": round(x, 3)} for c, x in ranked[:2]],
                    "whisker": v["whisker_cat"], "label_by": v["label_by"],
                    "why": why(models, cats, v, ranked) if state == "unsure" else [],
                    "reviewed": state == "unsure" and v["id"] in reviewed,
                    "check": checks.get(v["id"]) if state == "confirmed" and kept.get(v["id"]) != v["label_cat"] else None,
                    "frames": frames, "clip": vids[0] if vids else None, "clips": vids,
                    "crops": [{"url": c["url"], "ts": c["ts"], "score": c["score"]}
                              for c in sorted(sorted(crops[v["id"]], key=lambda c: -c["score"])[:4], key=lambda c: c["ts"])]})
    health = dict(request.app["health"])
    health["whisker_sync"] = kv_get(db, "whisker_sync")
    cam = kv_get(db, "cam") or {}
    naps = build_naps(db, vm, cat_ids, int(cam.get("nap_every_min") or 10), now)
    naps["cams"] = [c for c, x in CAMS.items() if x.get("naps")]
    sightings = build_sightings(db, vm, cat_ids, [c for c in clips if c["end_ts"] > now - 86400], rows,
                                request.app["whisker"].status, now)
    return web.json_response({"now": now, "version": int((STATIC / "index.html").stat().st_mtime),
                              "places": HOUSE.get("places", {}),
                              "cats": [{**c, "lb": round(m["mean"], 1) if (m := models["weight"]["lr4"].get(c["id"])) else None}
                                       for c in cats],
                              "visits": out, "omitted": omitted, "naps": naps, "sightings": sightings,
                              "robots": list(request.app["whisker"].status.values()),
                              "controls": (kv_get(db, "controls", []) or [])[-10:],
                              "feeder": feeder_view(request.app),
                              "cam": kv_get(db, "cam"), "health": health,
                              "model": {rb: {c: {k: round(x, 2) if isinstance(x, float) else x for k, x in m.items()}
                                             for c, m in wm.items()} for rb, wm in models["weight"].items()},
                              "box_use": {rb: {c: round(x, 3) for c, x in mx.items()} for rb, mx in models["mix"].items()},
                              "scale_shift": round(models["shift"], 2),
                              "photo_model": bool(models["photo"]),
                              "camera_model": {"refs": {c: sum(1 for _, ir, _ in x if not ir) for c, x in vm["protos"].items()},
                                               "night_refs": {c: sum(1 for _, ir, _ in x if ir) for c, x in vm["protos"].items()},
                                               "night": vm["night"], "day": vm["day"], "coat": vm["coat"],
                                               "tapped": vm["n_labeled"][False], "night_tapped": vm["n_labeled"][True],
                                               "learned_from_visits": len(auto),
                                               "photo_groups": len(photo_groups()),
                                               "photo_labels": len(kv_get(db, "photo_labels", {}) or {})}})


async def api_label(request: web.Request):
    body = await request.json()
    db = request.app["db"]
    cat = body.get("cat") or None
    if cat and cat not in ("none", "unknown") and not db.execute("SELECT 1 FROM cats WHERE id=?", (cat,)).fetchone():
        raise web.HTTPBadRequest(text="unknown cat")
    by = (str(body.get("by") or "").strip()[:24]) or None
    kept = kv_get(db, "kept", {}) or {}
    if body.get("keep") and cat:
        kept[str(body.get("id"))] = cat                # "yes, I'm sure": stop asking about this tap
    else:
        kept.pop(str(body.get("id")), None)
    kv_set(db, "kept", kept)
    n = db.execute("UPDATE visits SET label_cat=?, label_by=?, label_at=? WHERE id=?",
                   (cat, by if cat else None, int(time.time()) if cat else None, body.get("id"))).rowcount
    db.commit()
    if not n:
        raise web.HTTPNotFound()
    return web.json_response({"ok": True})


async def api_nap_wrong(request: web.Request):
    """"Not right" on a nap: drop that pet from the nap looks between start and end, and log who said so."""
    body = await request.json()
    db = request.app["db"]
    try:
        start, end, who = int(body["start"]), int(body["end"]), str(body["who"])
    except (KeyError, TypeError, ValueError):
        raise web.HTTPBadRequest(text="start, end and who are needed")
    by = (str(body.get("by") or "").strip()[:24]) or None
    cats = [dict(r) for r in db.execute("SELECT id FROM cats")]
    cat_ids = [c["id"] for c in cats]
    ids, _ = cat_prompts(db)
    vm = build_vision_model(db, cats, {})
    removed = 0
    for r in db.execute("SELECT id, animals FROM nap_checks WHERE ts BETWEEN ? AND ?", (start - 5, end + 5)).fetchall():
        animals = json.loads(r["animals"] or "[]")
        keep = [k for k in animals if nap_who(vm, k, cat_ids, ids) != who]
        for k in animals:
            if k not in keep and k.get("crop"):
                crop = (MEDIA / k["crop"]).resolve()
                if crop.is_relative_to(MEDIA.resolve()):
                    crop.unlink(missing_ok=True)
        if len(keep) != len(animals):
            removed += len(animals) - len(keep)
            db.execute("UPDATE nap_checks SET animals=? WHERE id=?", (json.dumps(keep), r["id"]))
    db.commit()
    log.info("nap removed: %s %s-%s by %s (%s looks)", who, start, end, by, removed)
    return web.json_response({"ok": True, "removed": removed})


async def api_reviewed(request: web.Request):
    """Mark unsorted visits as looked-at (Skip, or "hide the ones I went through"): they keep their best guess
    in the list but stop showing in "Who's this?"."""
    body = await request.json()
    ids = [str(x) for x in (body.get("ids") or [])][:500]
    db = request.app["db"]
    reviewed = set(kv_get(db, "reviewed", []) or []) | set(ids)
    kv_set(db, "reviewed", sorted(reviewed))
    return web.json_response({"ok": True, "reviewed": len(reviewed)})


CONTROL_NAMES = {"clean": "Clean now", "night_light": "Night light", "panel_lock": "Button lock",
                 "wait": "Wait time", "reset_drawer": "Drawer emptied"}


FEEDER_OPS = {"feed", "meal", "remove", "lock", "voice", "refill", "desiccant"}


def feeder_view(app) -> dict | None:
    """The feeder's last reading (refreshed every 5 min and after each button), its recent button history, and the
    house's quiet hours for meals (house.json "feeder_quiet": {"from": "22:00", "to": "09:00"}; the page asks twice
    before setting a meal inside them)."""
    f = app["feeder"]
    if not f:
        return None
    return {**f.state, "quiet": HOUSE.get("feeder_quiet"), "log": (kv_get(app["db"], "feeder_log", []) or [])[-10:]}


async def api_feeder(request: web.Request):
    """The feeder buttons: {op, by, ...}; see Feeder.act. Sent to HGsmart as its own app would."""
    f: Feeder | None = request.app["feeder"]
    if not f:
        raise web.HTTPNotFound(text="No feeder is set up.")
    body = await request.json()
    op, by = str(body.get("op") or ""), (str(body.get("by") or "").strip()[:24]) or None
    if op not in FEEDER_OPS:
        raise web.HTTPBadRequest(text="unknown feeder button")
    try:
        ok = await f.act(op, body)
    except (ValueError, TypeError):
        raise web.HTTPBadRequest(text="bad value")
    except Exception as e:
        log.warning("feeder %s failed: %s", op, e)
        raise web.HTTPBadGateway(text="The feeder didn't take that. Try again in a minute.")
    detail = {k: body[k] for k in ("portions", "slot", "time", "on", "value") if k in body}
    log.info("feeder %s %s by %s: %s", op, json.dumps(detail), by or "?", "ok" if ok else "refused")
    db = request.app["db"]
    hist = (kv_get(db, "feeder_log", []) or [])[-49:]
    hist.append({"op": op, **detail, "by": by, "at": int(time.time()), "ok": ok})
    kv_set(db, "feeder_log", hist)
    if not ok:
        raise web.HTTPBadGateway(text="The feeder said no. Try again in a minute.")
    await asyncio.sleep(2)                        # let the feeder apply it before reading it back
    await f.refresh()
    return web.json_response({"ok": True, "feeder": feeder_view(request.app)})


async def api_control(request: web.Request):
    """The box buttons: {robot: "lr4"|"evo", op, value, by}. Sent to Whisker exactly as its own app would."""
    body = await request.json()
    key, op, value = str(body.get("robot") or ""), str(body.get("op") or ""), body.get("value")
    by = (str(body.get("by") or "").strip()[:24]) or None
    w: Whisker = request.app["whisker"]
    r = w.robot(key) if w.account else None
    if r is None:
        raise web.HTTPServiceUnavailable(text="That box isn't connected right now.")
    if op not in CONTROL_NAMES:
        raise web.HTTPBadRequest(text="unknown control")
    lock = w.locks.setdefault(key, asyncio.Lock())
    if lock.locked():
        raise web.HTTPConflict(text="Still sending the last change to that box.")
    async with lock:
        try:
            if op == "clean":
                if (w.status.get(key) or {}).get("cat_inside") or r.status in CAT_INSIDE:
                    raise web.HTTPConflict(text="A cat is inside. The box cleans by itself after the cat leaves.")
                ok = await r.start_cleaning()
            elif op == "night_light":
                ok = await r.set_night_light_mode(NightLightMode(str(value).upper()))
            elif op == "panel_lock":
                ok = await r.set_panel_lockout(bool(value))
            elif op == "wait":
                if not isinstance(value, int) or value not in r.VALID_WAIT_TIMES:
                    raise web.HTTPBadRequest(text="wait must be one of %s" % r.VALID_WAIT_TIMES)
                ok = await r.set_wait_time(value)
            else:
                fn = getattr(r, "reset_waste_drawer", None)
                if fn is None:
                    raise web.HTTPBadRequest(text="This box resets its drawer level by itself.")
                ok = await fn()
        except web.HTTPException:
            raise
        except ValueError:
            raise web.HTTPBadRequest(text="bad value")
        except Exception as e:
            log.warning("control %s %s failed: %s", key, op, e)
            raise web.HTTPBadGateway(text="Whisker didn't take that. Try again in a minute.")
        ok = ok is not False
        log.info("control %s %s=%s by %s: %s", key, op, value, by or "?", "ok" if ok else "refused")
        db = request.app["db"]
        hist = (kv_get(db, "controls", []) or [])[-49:]
        hist.append({"robot": key, "op": op, "value": value, "by": by, "at": int(time.time()), "ok": ok})
        kv_set(db, "controls", hist)
        if not ok:
            raise web.HTTPBadGateway(text="The box said no. Try again in a minute.")
        await asyncio.sleep(2)                    # let the robot apply it before reading it back
        try:
            await r.refresh()
        except Exception:
            pass
        w.on_robot_update(r)
    return web.json_response({"ok": True, "robot": w.status.get(key)})


async def internal_clip(request: web.Request):
    b = await request.json()
    rel = Path(b["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise web.HTTPBadRequest()
    frames = b.get("frames") or []
    look, areas = (None, {}) if b.get("reason") == "nap" else \
        await asyncio.get_running_loop().run_in_executor(None, analyze_frames, MEDIA / rel, frames)
    for f in frames:
        f["s"] = areas.get(f["f"], 0)
    db = request.app["db"]
    db.execute("INSERT INTO clips(cam,start_ts,end_ts,reason,robot,path,frames,look) VALUES(?,?,?,?,?,?,?,?)",
               (b.get("cam"), int(b["start_ts"]), int(b["end_ts"]), b.get("reason"), b.get("robot"), str(rel),
                json.dumps(frames), json.dumps(look) if look else None))
    db.commit()
    request.app["vision_kick"].set()
    return web.json_response({"ok": True, "look": look})


async def internal_cam(request: web.Request):
    b = await request.json()
    for c in b.get("cams", []):
        c["label"] = CAMS.get(c["name"], {}).get("label", c["name"])
    kv_set(request.app["db"], "cam", {**b, "at": int(time.time())})
    return web.json_response({"ok": True})


async def media(request: web.Request):
    rel = Path(request.match_info["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise web.HTTPNotFound()
    f = (MEDIA / rel).resolve()
    if not f.is_relative_to(MEDIA.resolve()) or not f.is_file():   # a symlink out of media/ is not served
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={"Cache-Control": "private, max-age=31536000, immutable"})


async def cat_photo(request: web.Request):
    f = DATA / "cats" / f"{Path(request.match_info['id']).name}.jpg"
    if not f.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(f, headers={"Cache-Control": "private, max-age=86400"})


async def index(request):
    return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


async def healthz(request):
    w = request.app["health"].get("whisker", {})
    return web.json_response({"ok": bool(w.get("ok")), "whisker_at": w.get("at")})


def static_file(name):
    async def h(request):
        return web.FileResponse(STATIC / name, headers={"Cache-Control": "public, max-age=86400"})
    return h


def drop_clip(db, r):
    folder = MEDIA / r["path"]
    for f in folder.glob("*") if folder.is_dir() else []:
        f.unlink(missing_ok=True)
    if folder.is_dir():
        folder.rmdir()
    (MEDIA / (r["path"] + ".mp4")).unlink(missing_ok=True)
    db.execute("DELETE FROM clips WHERE id=?", (r["id"],))


async def janitor_loop(app):
    while True:
        await asyncio.sleep(300)
        now = time.time()
        db = app["db"]
        # a nap check the camera model never got to (GPU worker down) must not linger
        for r in db.execute("SELECT id, path FROM clips WHERE reason='nap' AND start_ts<?", (now - 1200,)).fetchall():
            drop_clip(db, r)
        # a clip that lines up with no litter-box visit within 30 min AND shows no cat or dog (people walking
        # by, an empty room) is deleted. Pet sightings stay.
        for r in db.execute("SELECT id, cam, path, start_ts, end_ts, vision FROM clips WHERE end_ts<? AND end_ts>?",
                            (now - 1800, now - 3 * 86400)).fetchall():
            v = json.loads(r["vision"]) if r["vision"] else None
            if v is None or (v.get("cats") and not v.get("error")):
                continue  # not analysed yet, or it has a pet in it
            robot = CAM_ROBOT.get(r["cam"])
            hit = db.execute("SELECT 1 FROM visits WHERE start_ts-75<=? AND end_ts+60>=? AND (? IS NULL OR robot=?) LIMIT 1",
                             (r["end_ts"], r["start_ts"], robot, robot)).fetchone()
            if not hit:
                drop_clip(db, r)
        for f in MEDIA.glob("*/*.mp4"):
            if now - f.stat().st_mtime > CLIP_KEEP_DAYS * 86400:
                f.unlink(missing_ok=True)
        for r in db.execute("SELECT id, path FROM clips WHERE end_ts<?", (now - FRAME_KEEP_DAYS * 86400,)).fetchall():
            drop_clip(db, r)
        db.commit()


async def on_startup(app):
    app["tasks"] = [asyncio.create_task(whisker_loop(app)), asyncio.create_task(janitor_loop(app)),
                    asyncio.create_task(vision_loop(app))]
    if app["feeder"]:
        app["tasks"].append(asyncio.create_task(feeder_loop(app)))


async def on_cleanup(app):
    for t in app["tasks"]:
        t.cancel()
    if app["whisker"].account:
        await app["whisker"].account.disconnect()
    if app["feeder"]:
        await app["feeder"].client.close()


def make_app() -> web.Application:
    app = web.Application(middlewares=[access_log, auth])
    app["db"] = db_open()
    app["web_key"] = secret("web-key")
    app["internal_key"] = secret("internal-key")
    app["vision_key"] = secret("vision-key")
    fc = SECRETS / "family-code"
    if not fc.exists():                      # first run: a code a person can type (change it to a word if you like)
        SECRETS.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_private(fc, f"{secrets.randbelow(10 ** 6):06d}\n")
        log.warning("first run: the family code is %s (edit secrets/family-code to change it)", fc.read_text().strip())
    app["family_code"] = secret("family-code")
    app["vision_sem"] = asyncio.Semaphore(1)
    app["vision_kick"] = asyncio.Event()
    app["health"] = {}
    app["whisker"] = Whisker(app)
    fs = SECRETS / "hgsmart.json"             # optional feeder (feeder.py): its tokens, never a password
    app["feeder"] = Feeder(fs, NY, kv_get, kv_set, app["db"]) if fs.exists() else None
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_post("/login", login)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/label", api_label)
    app.router.add_post("/api/reviewed", api_reviewed)
    app.router.add_post("/api/nap-wrong", api_nap_wrong)
    app.router.add_post("/api/control", api_control)
    app.router.add_post("/api/feeder", api_feeder)
    app.router.add_post("/internal/clip", internal_clip)
    app.router.add_post("/internal/cam", internal_cam)
    app.router.add_get("/media/{path:.+}", media)
    app.router.add_get("/cat/{id}.jpg", cat_photo)
    for name in ("manifest.webmanifest", "icon-192.png", "icon-512.png", "apple-touch-icon.png"):
        app.router.add_get(f"/{name}", static_file(name))
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # 127.0.0.1 behind a proxy or Tailscale Funnel; CATBOX_BIND=0.0.0.0 to serve the home network directly
    web.run_app(make_app(), host=os.environ.get("CATBOX_BIND", "127.0.0.1"), port=PORT, access_log=None)
