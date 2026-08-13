#!/usr/bin/env python3
"""Drive the ESP32 BDH REPL over serial: send a prompt, capture the reply.

Opens the port with DTR/RTS deasserted — asserting either resets the board
into the bootloader instead of talking to the firmware.

Usage:
    uv run --with pyserial esp32/bench.py --prompt "Once upon a time"
"""
import argparse
import sys
import time

import serial


def open_port(port, baud):
    """Open without toggling DTR/RTS, which would reset the board."""
    s = serial.Serial()
    s.port = port
    s.baudrate = baud
    s.timeout = 1
    s.dtr = False
    s.rts = False
    s.open()
    return s


def read_until(ser, needle, timeout, echo=True):
    """Read until `needle` appears or `timeout` elapses. Returns text seen."""
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = ser.read(256).decode("utf-8", errors="replace")
        if chunk:
            buf += chunk
            if echo:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            if needle in buf:
                return buf
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--timeout", type=float, default=400.0,
                    help="seconds to wait for generation to finish")
    ap.add_argument("--reset", action="store_true",
                    help="pulse DTR/RTS to reboot the board first")
    args = ap.parse_args()

    ser = open_port(args.port, args.baud)

    if args.reset:
        # Standard ESP32 reset: pull EN low briefly via RTS.
        ser.rts = True
        time.sleep(0.1)
        ser.rts = False
        time.sleep(0.5)

    # Drain the banner. The firmware prints "Ready!" once weights are mapped.
    banner = read_until(ser, "Ready!", timeout=15, echo=True)
    if "Ready!" not in banner:
        print("\n[warn] no 'Ready!' banner — board may already be mid-session",
              file=sys.stderr)

    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write((args.prompt + "\n").encode())
    ser.flush()

    # Generation ends with the timing line, then the profile block.
    out = read_until(ser, "lm_head", timeout=args.timeout, echo=True)
    # Let the last lines land.
    trailing = read_until(ser, "\x00", timeout=1.5, echo=True)
    ser.close()

    if "tok/s" not in out + trailing:
        print("\n[warn] never saw a tok/s line — timed out?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
