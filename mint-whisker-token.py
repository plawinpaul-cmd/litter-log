#!/usr/bin/env python3
"""One-time Whisker sign-in: saves only the session tokens to secrets/whisker-token.json (mode 600).

    WHISKER_EMAIL=you@example.com python mint-whisker-token.py
Password: WHISKER_PASSWORD env, else macOS Keychain (service "whisker"), else a hidden prompt.
Whisker locks accounts after repeated failures, so this tries once.
"""
import asyncio, getpass, json, os, subprocess, sys
from pathlib import Path
from pylitterbot import Account

USER = os.environ.get("WHISKER_EMAIL") or (sys.argv[1] if len(sys.argv) > 1 else "")
OUT = Path(__file__).resolve().parent / "secrets" / "whisker-token.json"


def password() -> str:
    if os.environ.get("WHISKER_PASSWORD"):
        return os.environ["WHISKER_PASSWORD"]
    if sys.platform == "darwin":
        r = subprocess.run(["security", "find-generic-password", "-s", "whisker", "-a", USER, "-w"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.rstrip("\n")
    return getpass.getpass("Whisker password: ")


async def main():
    if not USER:
        sys.exit("set WHISKER_EMAIL (the account that owns your Litter-Robots)")
    acct = Account()
    try:
        await acct.connect(username=USER, password=password())
        OUT.parent.mkdir(mode=0o700, exist_ok=True)
        OUT.write_text(json.dumps(acct.session.tokens) + "\n")
        os.chmod(OUT, 0o600)
        print("saved", OUT.name)
    finally:
        await acct.disconnect()

asyncio.run(main())
