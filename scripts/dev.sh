#!/usr/bin/env bash
#
# Start the backend and frontend on freshly-allocated ports, wired together.
#
# Why this exists
# ---------------
# The frontend used to hardcode http://127.0.0.1:8000. Any backend started
# elsewhere — because 8000 was busy, because a previous run never died — made
# the splash page report "engine offline" while the API was perfectly healthy.
# The failure is silent and looks like a broken model, which is the worst
# possible way for a port conflict to present itself.
#
# So: ask the OS for two free ports, tell the frontend where the backend
# actually landed, and reap whatever the last run left behind. There is no
# fixed port left to collide with.
#
# Usage:
#   ./scripts/dev.sh          start both, wired together
#   ./scripts/dev.sh --stop   kill a previous run and exit
set -euo pipefail

# Job control, so each background job below becomes its own PROCESS GROUP.
# This is load-bearing, not incidental — see the note above reap().
set -m

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$ROOT/shot-vision-engine-main"
RUNTIME="$ROOT/.dev-runtime"
PIDFILE="$RUNTIME/pids"
ENVFILE="$FRONTEND/.env.local"

mkdir -p "$RUNTIME"

# --- reaping the previous run -------------------------------------------------
# Killing the recorded pid is not enough. `npm run dev` spawns vite as a CHILD,
# and killing npm alone orphans vite still holding its port — measured, not
# hypothetical. So each server runs in its own process group and the whole
# group is signalled.
#
# That makes the guard essential rather than paranoid. Process-group ids are
# recycled like pids, and a negative kill against a recycled group is far worse
# than a stray single kill: on a dev machine the editor itself is typically one
# process group with dozens of members.
#
# The guard therefore keys on the PORT, not on command-line text. Matching
# command lines was tried and is too loose — any process in the group merely
# MENTIONING the pattern satisfies it, which during testing matched the test
# harness's own argv and killed the wrong group. The port is the resource we
# actually want back, so ask who is holding it: only if the current listener on
# that port is genuinely inside the recorded group do we signal the group.
# If nobody holds the port, there is nothing to reclaim and we do nothing.
reap() {
  [ -f "$PIDFILE" ] || return 0
  local pgid label port listener lpgid
  while IFS=$'\t' read -r pgid label port; do
    [ -n "${pgid:-}" ] && [ -n "${port:-}" ] || continue
    case "$pgid$port" in ''|*[!0-9]*) continue ;; esac   # ignore malformed lines
    [ "$pgid" -gt 1 ] || continue                         # never signal group 0/1

    listener="$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
    if [ -z "$listener" ]; then
      continue                                            # port already free
    fi
    lpgid="$(ps -o pgid= -p "$listener" 2>/dev/null | tr -d ' ' || true)"
    if [ "$lpgid" = "$pgid" ]; then
      echo "  reaping stale $label on port $port (process group $pgid)"
      kill -- "-$pgid" 2>/dev/null || true
    else
      echo "  port $port is held by pid $listener outside group $pgid — leaving it alone"
    fi
  done < "$PIDFILE"
  rm -f "$PIDFILE"
}

if [ "${1:-}" = "--stop" ]; then
  echo "Stopping any previous run:"
  reap
  rm -f "$ENVFILE"
  echo "Done."
  exit 0
fi

echo "Checking for leftovers from a previous run:"
reap

# --- port allocation ----------------------------------------------------------
# Bind to port 0 and let the kernel pick, rather than probing a fixed candidate
# list. There is a small window between closing this socket and the server
# binding it; in practice the kernel does not reissue the same ephemeral port
# that fast, and a collision is a loud bind error rather than silent breakage.
free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

API_PORT="$(free_port)"
WEB_PORT="$(free_port)"

# --- the sync step ------------------------------------------------------------
# src/lib/api.ts reads VITE_API_BASE and falls back to :8000. Vite reads env
# files ONLY at startup, so this must be written before the frontend launches.
cat > "$ENVFILE" <<ENVEOF
# Written by scripts/dev.sh — regenerated on every run, do not edit by hand.
VITE_API_BASE=http://127.0.0.1:$API_PORT
ENVEOF

cleanup() {
  echo ""
  echo "Shutting down."

  # Were we superseded? A second `dev.sh` started while this one was still
  # alive would, in ITS OWN startup reap(), kill OUR backend/frontend
  # process groups and then overwrite $PIDFILE with ITS OWN fresh entries.
  # We (the superseded run) would then see our children gone, exit our wait
  # loop normally, and land here — and if we called reap() unconditionally,
  # we'd read the SECOND run's entries and kill ITS live, wanted processes
  # too, thinking they're leftovers. That cross-instance kill is almost
  # certainly why servers kept dying across unrelated dev.sh invocations
  # tonight: not a single bad shutdown, but each one taking out the next.
  #
  # The tell: our own (pid, port) lines are gone from $PIDFILE, replaced by
  # someone else's. If that's happened, back off entirely — don't reap,
  # don't touch the env file, just exit. The newer run owns cleanup now.
  superseded=0
  if [ -f "$PIDFILE" ]; then
    grep -qF "$(printf '%s\t' "$API_PID")" "$PIDFILE" 2>/dev/null || superseded=1
  fi

  if [ "$superseded" = "1" ]; then
    echo "  superseded by a newer run — leaving its processes and config alone"
  else
    reap
    # Only remove the env file if it still describes THIS run. Even here,
    # without the supersession race, this stays the safe check: deleting
    # unconditionally could otherwise still race a run that started in the
    # instant between our check above and this line.
    if [ -f "$ENVFILE" ] && grep -q ":$API_PORT\$" "$ENVFILE" 2>/dev/null; then
      rm -f "$ENVFILE"
    elif [ -f "$ENVFILE" ]; then
      echo "  $ENVFILE now belongs to a newer run — leaving it in place"
    fi
  fi
}
trap cleanup EXIT INT TERM

# --- launch -------------------------------------------------------------------
echo "Backend  → port $API_PORT"
( cd "$ROOT" && exec uvicorn src.inference.api:app --reload --port "$API_PORT" ) &
printf '%s\t%s\t%s\n' "$!" "backend" "$API_PORT" >> "$PIDFILE"
API_PID=$!

# --strictPort turns a taken port into a hard error instead of a silent walk to
# the next one — if vite wandered, the URL printed below would be a lie.
echo "Frontend → port $WEB_PORT"
( cd "$FRONTEND" && exec npm run dev -- --port "$WEB_PORT" --strictPort ) &
printf '%s\t%s\t%s\n' "$!" "frontend" "$WEB_PORT" >> "$PIDFILE"
WEB_PID=$!

# --- wait for the API, then report -------------------------------------------
for _ in $(seq 1 60); do
  curl -sf -m 2 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 && break
  kill -0 "$API_PID" 2>/dev/null || { echo "Backend died on startup."; exit 1; }
  sleep 1
done

echo ""
echo "─────────────────────────────────────────────"
echo "  App    http://localhost:$WEB_PORT"
echo "  API    http://127.0.0.1:$API_PORT"
curl -sf -m 5 "http://127.0.0.1:$API_PORT/health" 2>/dev/null \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  Model  %s (%s features)" % (d["model_name"], d["features"]))' \
  2>/dev/null || echo "  Model  health check not answering yet"
echo "─────────────────────────────────────────────"
echo "  Ctrl-C stops both."
echo ""

# Exit as soon as EITHER process dies, so a crashed backend does not leave a
# frontend running against nothing — which is the "engine offline" state again.
# `wait -n` would say this in one line, but macOS ships bash 3.2, where it does
# not exist. Polling is the portable equivalent.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
echo "One of the two processes exited."
