#!/bin/bash
set -e
FQBN="esp32:esp32:XIAO_ESP32S3:PSRAM=opi,PartitionScheme=max_app_8MB"
PORT="${1:-/dev/ttyACM0}"
DIR="$(dirname "$0")/bdh_tinystories"
WEIGHTS="$(dirname "$0")/weights.bin"

echo "=== BDH TinyStories ESP32 Flash ==="

echo "Compiling..."
arduino-cli compile --fqbn "$FQBN" "$DIR"

echo "Uploading firmware to $PORT..."
arduino-cli upload --fqbn "$FQBN" -p "$PORT" "$DIR"

if [ -f "$WEIGHTS" ]; then
    echo "Flashing weights to 0x100000..."
    sleep 2
    uv tool run esptool --port "$PORT" --baud 921600 write_flash 0x100000 "$WEIGHTS"
    echo "Weights flashed."
else
    echo "WARNING: $WEIGHTS not found. Run export_weights.py first, then:"
    echo "  uv tool run esptool --port $PORT write_flash 0x100000 esp32/weights.bin"
fi

echo ""
echo "Done. Open serial monitor:"
echo "  arduino-cli monitor -p $PORT -c baudrate=115200"
