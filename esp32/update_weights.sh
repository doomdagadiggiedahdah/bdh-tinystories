#!/bin/bash
# Point this at a checkpoint, get it onto the board.
#
#   update_weights.sh checkpoints/step_010000.pt
#   update_weights.sh --port /dev/ttyACM1 step_000999.pt
#
# The port is a FLAG, not a positional arg, so shell brace expansion
# (step_000{999,800}.pt) can't silently land a second checkpoint in the
# port slot. Passing several checkpoints flashes the last one and says so.
set -euo pipefail

PORT=""
FORCE=0
CHECKPOINTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --port|-p) PORT="${2:?--port needs a value}"; shift 2 ;;
        --force|-f) FORCE=1; shift ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)  CHECKPOINTS+=("$1"); shift ;;
    esac
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Usage: update_weights.sh [--port DEV] [--force] <checkpoint.pt>" >&2
    exit 1
fi

# Several checkpoints given (usually brace expansion). The board holds one,
# and guessing which is wrong: with step_000{999,800}.pt the "last" is the
# OLDER checkpoint. Refuse and let the caller pick.
if [ ${#CHECKPOINTS[@]} -gt 1 ]; then
    echo "ERROR: ${#CHECKPOINTS[@]} checkpoints given, but the board holds one:" >&2
    for c in "${CHECKPOINTS[@]}"; do echo "  $c" >&2; done
    echo >&2
    echo "Pick one. (Brace expansion like step_000{999,800}.pt expands to two" >&2
    echo "arguments — and the last one is the older checkpoint.)" >&2
    exit 1
fi
CHECKPOINT="${CHECKPOINTS[0]}"

[ -f "$CHECKPOINT" ] || { echo "No such checkpoint: $CHECKPOINT" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS="$SCRIPT_DIR/weights.bin"
VENV=~/Documents/ProgramExperiments/exp/.venv/bin/python
[ -x "$VENV" ] || { echo "Python venv not found at $VENV" >&2; exit 1; }

# Auto-detect the port if not given, so the common case needs no flag.
if [ -z "$PORT" ]; then
    mapfile -t PORTS < <(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true)
    case ${#PORTS[@]} in
        0) echo "No /dev/ttyACM* or /dev/ttyUSB* found — is the board plugged in?" >&2; exit 1 ;;
        1) PORT="${PORTS[0]}" ;;
        *) echo "Multiple boards: ${PORTS[*]}" >&2
           echo "Pick one with --port (check which is the ESP32-S3: arduino-cli board list)" >&2
           exit 1 ;;
    esac
fi

echo "=== BDH Weight Update ==="
echo "Checkpoint: $CHECKPOINT"
echo "Port:       $PORT"
echo

# A serial monitor holding the port makes esptool fail with a confusing
# "port is busy". Say so plainly instead.
if command -v fuser >/dev/null && fuser "$PORT" >/dev/null 2>&1; then
    echo "ERROR: $PORT is held by another process (serial monitor?):" >&2
    fuser -v "$PORT" >&2 || true
    echo "Close it and retry." >&2
    exit 1
fi

OLD_HASH=""
[ -f "$WEIGHTS" ] && OLD_HASH="$(sha256sum "$WEIGHTS" | cut -d' ' -f1)"

echo "Exporting weights..."
"$VENV" "$SCRIPT_DIR/export_weights.py" "$CHECKPOINT" "$WEIGHTS"

NEW_HASH="$(sha256sum "$WEIGHTS" | cut -d' ' -f1)"
if [ "$FORCE" -eq 0 ] && [ "$OLD_HASH" = "$NEW_HASH" ]; then
    echo
    echo "Weights identical to what was last flashed — skipping the ~50s write."
    echo "(Use --force to flash anyway.)"
    exit 0
fi

echo
echo "Flashing to $PORT at 0x100000..."
uv tool run esptool --port "$PORT" --baud 921600 write_flash 0x100000 "$WEIGHTS"

echo
echo "Done. Talk to it with (same commands as PROGRESS.md):"
echo "  uv run --with pyserial esp32/bench.py --reset --prompt \"Once upon a time\""
echo "  arduino-cli monitor -p $PORT -c baudrate=115200"
