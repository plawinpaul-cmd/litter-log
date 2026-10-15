"""Optional: an HGsmart / HoneyGuaridan feeder (tested on the S25D), with the buttons of its own app, for anyone
with the family code.

Turned on by secrets/hgsmart.json: {"email": ..., "refresh_token": ..., "device_id": ...}. Only tokens are kept,
never the password. Each use of a refresh token returns a new one, which is saved; an older one keeps working
(measured 2026-10-12), so another client on the same account, like the HGsmart app or a script, is unaffected.

The feeder stores meal times in UTC. The family sets local times, so the local time of each meal is remembered
here with the UTC offset it was set under. When the offset changes (daylight saving) and the feeder still has the
old UTC time, the meal is sent again, or a 9:00 meal would become 8:00 in November. If the HGsmart cloud has
moved it already, or someone changed it in the app, the feeder's time is taken as the new one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from hgsmart_client.api import HGSmartApiClient
from hgsmart_client.helpers import parse_plan_value

log = logging.getLogger("catbox")
SLOTS = 6
REFRESH_S = 300


def _offset_min(tz) -> int:
    return int(datetime.now(tz).utcoffset().total_seconds() // 60)


def _local(hour: int, minute: int, offset: int) -> str:
    m = (hour * 60 + minute + offset) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _utc(local: str, offset: int) -> tuple[int, int]:
    h, m = (int(x) for x in local.split(":"))
    u = (h * 60 + m - offset) % 1440
    return u // 60, u % 60


class Feeder:
    def __init__(self, secret_path, tz, kv_get, kv_set, db):
        self.path, self.tz, self.kv_get, self.kv_set, self.db = secret_path, tz, kv_get, kv_set, db
        cfg = json.loads(secret_path.read_text())
        self.email, self.device = cfg["email"], cfg["device_id"]
        self.client = HGSmartApiClient(username=self.email, refresh_token=cfg["refresh_token"], timezone="UTC")
        self.lock = asyncio.Lock()
        self.state: dict = {"ok": False, "at": None}
        self.slots: dict[int, dict] = {}

    def _save_token(self):
        cfg = json.loads(self.path.read_text())
        if self.client.refresh_token and cfg.get("refresh_token") != self.client.refresh_token:
            cfg["refresh_token"] = self.client.refresh_token
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cfg))
            tmp.chmod(0o600)
            tmp.replace(self.path)

    async def _auth(self):
        if not self.client.access_token:
            if not await self.client.refresh_access_token():
                raise RuntimeError("HGsmart sign-in failed")
        self._save_token()

    async def refresh(self) -> dict:
        async with self.lock:
            try:
                await self._auth()
                attrs = await self.client.get_device_attributes(self.device) or {}
                stats = await self.client.get_feeder_stats(self.device) or {}
                today = await self.client.get_device_today_events(self.device) or {}
                devs = await self.client.get_devices() or []
                self._save_token()
                if not attrs:
                    raise RuntimeError("the feeder sent no settings")
                dev = next((d for d in devs if str(d.get("deviceId")) == self.device), {})
                self.slots = self._read_slots(attrs)
                await self._keep_local_times()
                self.state = {"ok": True, "at": int(time.time()), "online": dev.get("online"),
                              "name": dev.get("name") or "Feeder", "meals": self._meals(),
                              "lock": str(attrs.get("child", "")) == "1",
                              "voice": "custom" if str(attrs.get("choosevoice")) == "1" else "default",
                              "food": stats.get("remaining"), "desiccant_days": stats.get("desiccantExpire"),
                              "today": self._today(today.get("events") or [])}
            except Exception as e:
                log.warning("feeder refresh failed: %s", e)
                self.state = {**self.state, "ok": False, "error": "Can't reach the feeder right now."}
            return self.state

    def _read_slots(self, attrs: dict) -> dict[int, dict]:
        out = {}
        for k, v in attrs.items():
            if k.startswith("plan") and k != "plan" and (p := parse_plan_value(str(v))):
                out[p["slot"]] = p
        return out

    async def _keep_local_times(self):
        """kv "feeder_local": {slot: {"local": "HH:MM", "offset": minutes}}, see the module docstring."""
        now_off, kept = _offset_min(self.tz), self.kv_get(self.db, "feeder_local", {}) or {}
        for slot, p in self.slots.items():
            k = kept.get(str(slot))
            if k and (p["hour"], p["minute"]) == _utc(k["local"], k["offset"]) and k["offset"] != now_off:
                h, m = _utc(k["local"], now_off)       # unchanged on the feeder, but the clocks moved
                if await self.client.set_schedule(self.device, slot, h, m, p["portions"], p["enabled"]):
                    log.info("feeder: meal %d kept at %s local after the clock change", slot, k["local"])
                    p["hour"], p["minute"] = h, m
            kept[str(slot)] = {"local": _local(p["hour"], p["minute"], now_off), "offset": now_off}
        self.kv_set(self.db, "feeder_local", kept)

    def _meals(self) -> list[dict]:
        off = _offset_min(self.tz)
        out = []
        for slot in range(SLOTS):
            p = self.slots.get(slot)
            empty = p is None or (not p["enabled"] and p["hour"] == 0 and p["minute"] == 0)
            out.append({"slot": slot, "empty": empty, "time": _local(p["hour"], p["minute"], off) if p else None,
                        "portions": p["portions"] if p else 1, "on": bool(p and p["enabled"])})
        return out

    def _today(self, events: list[dict]) -> list[dict]:
        out = []
        for e in events:
            try:
                at = datetime.strptime(e["createTime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            desc = str(e.get("eventDesc") or "")
            n = next((int(t) for t in desc.replace("(", " ").split() if t.isdigit()), None)
            out.append({"at": int(at.timestamp()), "portions": n, "scheduled": desc.lower().startswith("scheduled"),
                        "text": desc})
        return sorted(out, key=lambda x: x["at"])

    async def act(self, op: str, body: dict) -> bool:
        async with self.lock:
            await self._auth()
            d, c = self.device, self.client
            if op == "feed":
                n = int(body.get("portions") or 1)
                if not 1 <= n <= 9:
                    raise ValueError("portions 1-9")
                ok = await c.send_feed_command(d, n)
            elif op == "lock":
                ok = await c.set_button_lockout(d, bool(body.get("on")))
            elif op == "voice":
                ok = await c.set_meal_call_sound_mode(d, body.get("value") == "custom")
            elif op == "refill":
                ok = await c.set_food_remaining(d, 100)
            elif op == "desiccant":
                ok = await c.reset_desiccant(d)
            elif op in ("meal", "remove"):
                slot = int(body.get("slot"))
                if not 0 <= slot < SLOTS:
                    raise ValueError("slot 0-5")
                off = _offset_min(self.tz)
                if op == "remove":
                    ok = await c.set_schedule(d, slot, 0, 0, 1, False)
                else:
                    local, n = str(body.get("time") or ""), int(body.get("portions") or 1)
                    datetime.strptime(local, "%H:%M")
                    if not 1 <= n <= 9:
                        raise ValueError("portions 1-9")
                    h, m = _utc(local, off)
                    ok = await c.set_schedule(d, slot, h, m, n, bool(body.get("on", True)))
                    if ok:
                        kept = self.kv_get(self.db, "feeder_local", {}) or {}
                        kept[str(slot)] = {"local": local, "offset": off}
                        self.kv_set(self.db, "feeder_local", kept)
            else:
                raise ValueError("unknown feeder button")
            self._save_token()
            return bool(ok)


async def feeder_loop(app):
    f = app.get("feeder")
    while f:
        await f.refresh()
        await asyncio.sleep(REFRESH_S)
