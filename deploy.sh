#!/usr/bin/env bash
# Ship Litter Log from this checkout to a Linux box over SSH and restart it (systemd user services).
# Settings come from the environment or ./deploy.env:
#   CATBOX_HOST         ssh host that runs the app (required)
#   CATBOX_GPU_HOST     ssh host with the GPU that runs vision/ (optional; no camera model without it)
#   CATBOX_FUNNEL_PORT  publish the app on this Tailscale Funnel port (optional)
#   CATBOX_VISION, CATBOX_TZ, CATBOX_BIND, CATBOX_RING  app settings, written to the box's app.env
#                       (CATBOX_VISION defaults to the GPU host's Tailscale address, port 8904)
# First run also moves the secrets (tokens only, never passwords) and sets up the venv.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/deploy.env" ] && set -a && . "$SRC/deploy.env" && set +a
HOST=${CATBOX_HOST:?set CATBOX_HOST (in deploy.env or the environment)}
GPU=${CATBOX_GPU_HOST:-}
[ -z "${CATBOX_SKIP_GPU:-}" ] || GPU=   # CATBOX_SKIP_GPU=1: app only (e.g. while the GPU box is busy)
# camera model worker on the GPU box (venv + models: see vision/vision.py)
# restart the worker only when its code changed: a restart costs a 1-2 min model reload on the next clip
if [ -n "$GPU" ]; then
  CHANGED=$(rsync -ai "$SRC/vision/vision.py" "$SRC/vision/catvision.service" "$GPU:catvision/" | wc -l)
  scp -q -p "$SRC/secrets/vision-key" "$GPU:catvision/key"
  ssh "$GPU" "chmod 600 ~/catvision/key && mkdir -p ~/.config/systemd/user && cp ~/catvision/catvision.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable catvision.service >/dev/null 2>&1 && { [ $CHANGED -eq 0 ] || systemctl --user restart catvision.service; }"
fi

# the app's own settings: without CATBOX_VISION it would look for the camera model on its own box
if [ -z "${CATBOX_VISION:-}" ] && [ -n "${CATBOX_GPU_HOST:-}" ]; then
  GPU_IP=$(ssh "$CATBOX_GPU_HOST" 'tailscale ip -4' 2>/dev/null | head -1)
  [ -n "$GPU_IP" ] && CATBOX_VISION="http://$GPU_IP:8904"
fi
APP_ENV=$(for k in CATBOX_VISION CATBOX_TZ CATBOX_BIND CATBOX_RING; do [ -z "${!k:-}" ] || printf '%s=%s\n' "$k" "${!k}"; done)

SYNCED=$(rsync -ai --delete --exclude data --exclude secrets --exclude vision --exclude node_modules --exclude .venv --exclude '*.pyc' \
  --exclude .git --exclude deploy.env --exclude app.env --exclude RUNBOOK.local.md "$SRC/" "$HOST:catbox/")
# restarting the Ring sidecar cuts off a clip being recorded, so only when its own files changed
RING_CHANGED=$(printf '%s\n' "$SYNCED" | grep -cE ' ring/[^/]+$| systemd/catbox-ring\.service$' || true)
DEPS_CHANGED=$(printf '%s\n' "$SYNCED" | grep -cE ' ring/package(-lock)?\.json$' || true)
if [ -d "$SRC/secrets" ] && ls "$SRC/secrets"/* >/dev/null 2>&1; then
  ssh "$HOST" 'mkdir -p ~/catbox/secrets && chmod 700 ~/catbox/secrets'
  scp -q -p "$SRC/secrets/"* "$HOST:catbox/secrets/"
  ssh "$HOST" 'chmod 600 ~/catbox/secrets/*'
  # tokens live only on the box that uses them (Ring rotates its token; two copies would fight)
  rm -f "$SRC/secrets/"*-token.json
fi
ssh "$HOST" FUNNEL="${CATBOX_FUNNEL_PORT:-}" RING_CHANGED="$RING_CHANGED" DEPS_CHANGED="$DEPS_CHANGED" APP_ENV="$(printf %q "$APP_ENV")" bash -s <<'REMOTE'
set -euo pipefail
cd ~/catbox
printf '%s\n' "$APP_ENV" > app.env
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt
{ [ "$DEPS_CHANGED" = 0 ] && [ -d ring/node_modules ]; } || (cd ring && npm ci --omit=dev --silent --no-audit --no-fund)
mkdir -p ~/.config/systemd/user
cp systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now catbox.service catbox-ring.service >/dev/null 2>&1
systemctl --user restart catbox.service
[ "$RING_CHANGED" = 0 ] || systemctl --user restart catbox-ring.service   # waits for a recording in progress
if [ -n "$FUNNEL" ]; then
  tailscale funnel status 2>/dev/null | grep -q ":$FUNNEL" || tailscale funnel --bg --https="$FUNNEL" http://127.0.0.1:8902 >/dev/null
fi
sleep 4
systemctl --user --no-pager --lines=0 status catbox.service catbox-ring.service | grep -E 'Active:|●'
REMOTE
