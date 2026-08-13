#!/bin/bash
set -e

CHECKPOINT="${1:?Usage: update_weights.sh <checkpoint.pt> [port]}"
PORT="${2:-/dev/ttyACM0}"
SCRIPT_DIR="$(dirname "$0")"
VENV=~/Documents/ProgramExperiments/exp/.venv/bin/python

echo "=== BDH Weight Update ==="
echo "Checkpoint: $CHECKPOINT"
echo "Port:       $PORT"

echo "Exporting weights..."
"$VENV" "$SCRIPT_DIR/export_weights.py" "$CHECKPOINT" "$SCRIPT_DIR/weights.bin"

echo "Flashing to $PORT at 0x100000..."
uv tool run esptool --port "$PORT" --baud 921600 write_flash 0x100000 "$SCRIPT_DIR/weights.bin"

echo ""
echo "Done. Connect with:"
echo "  arduino-cli monitor -p $PORT -c baudrate=115200"
