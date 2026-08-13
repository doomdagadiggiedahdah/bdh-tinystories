# BDH on ESP32-S3 — Progress

**Status (2026-08-13): WORKING.** The BDH model runs on the XIAO ESP32-S3 Sense,
generating text over serial at **0.49 tok/s**.

First on-device output (step-200 checkpoint, loss 2.05):
```
> Once upon a timed and Sand sho ay sam has Bore and mabilgs and and ave sa a jom
[prompt 16 B in 29.1s | 64 tokens in 131.5s = 0.49 tok/s]
```

## What's here

| File | Purpose |
|---|---|
| `export_weights.py` | checkpoint `.pt` → `weights.bin` (int8, layout v2: transposed encoders) |
| `bdh_tinystories/bdh_tinystories.ino` | incremental BDH inference + serial REPL |
| `flash.sh` | compile + upload firmware + flash weights |

## Board / model facts

- Board: XIAO ESP32-S3 Sense — 8MB flash, 8MB PSRAM
  (if you have more than one ESP32 plugged in, confirm the port with
  `arduino-cli board list` before flashing — it's easy to overwrite the
  wrong board)
- Model: D=192, mlp_mult=60, nh=4, 6 shared layers → 6.7M params, 6.42MB int8
- Weights live at raw flash offset `0x100000`, memory-mapped (no partition table)
- Firmware ~300KB; PSRAM caches 5.86MB at T_MAX=80
- Verified: C logits match PyTorch to 0.019 (pure int8 cost); with identical
  int8 weights the match is 3e-6 (exact)

## Update weights after training (checkpoint → board)

```bash
python esp32/export_weights.py checkpoints/step_010000.pt esp32/weights.bin
esp32/flash.sh /dev/ttyACM0        # compiles, uploads firmware, flashes weights
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200   # talk to it
```

If only weights changed (no .ino edits), skip the firmware step:
```bash
uv tool run esptool --port /dev/ttyACM0 write_flash 0x100000 esp32/weights.bin
```

Host-side sanity check before flashing (compares C forward pass vs PyTorch —
harness lives in the job tmp dir, rebuild from `PROGRESS` notes if gone):
compile the .ino with `-DARDUINO_HOST_TEST` stubs, run same prompt through
both, compare logits. Worth redoing after any .ino math change.

## Speed: where the time goes & what to try

Currently 0.49 tok/s (~2s/token). Per token: ~13MB flash reads (encoder +
encoder_v + decoder × 6 layers) + attention over int8 PSRAM caches.
Already done: incremental inference (was 10+ min/token naive), transposed
weight layouts for sequential flash reads, decoder rows skipped when the
ReLU gate is zero.

Ideas, roughly by expected payoff:

1. **ESP32-S3 SIMD (PIE)** — 128-bit vector int8 MACs via `ee.vmulas.s8`
   etc. The inner dot products are scalar float; quantizing activations to
   int8 per-token and using PIE could give 2–4x. esp-dsp library has
   primitives (`dsps_dotprod_s8`).
2. **Quantize activations to int8 throughout** — even without SIMD, int
   math beats float on this core; also halves PSRAM cache traffic.
3. **Exploit more sparsity** — x_sparse is ReLU output (~50%+ zeros).
   Encoder_v gate already skips when x_sparse[n]==0; could also skip
   encoder dot products via a threshold, or process only top-k neurons.
4. **Both cores** — inference is single-core; FreeRTOS task on core 0
   could take half the heads/neurons. Near-2x if memory bandwidth allows.
5. **Flash → PSRAM copy of hot weights** — PSRAM (OPI, 80MHz) is faster
   than DIO flash. Weights (6.42MB) + caches (5.86MB) > 8MB, so can't
   hold both fully; but at T_MAX=48 caches shrink to ~3.4MB, leaving room
   to host encoder+encoder_v (4.4MB) in PSRAM. Tradeoff: shorter stories.
6. **Higher flash clock** — sketch runs DIO; check if board supports QIO
   80MHz (`arduino-cli` FlashMode option) for ~2x flash bandwidth.
7. **Compiler flags** — Arduino builds with `-Os` (size). Force `-O3
   -ffast-math` for the sketch (build_opt.h / platform.local.txt). Trivial
   to try, possibly 1.5x+ on the float inner loops. Do this FIRST.
8. **Skip lm_head during prompt** — logits are computed every step but only
   needed for the last prompt byte and during generation. Guard with a flag.
   Small (~2%) but free.
9. **Cheaper RoPE trig** — ~34k `cosf`/`sinf` calls per token. Replace with
   a per-frequency incremental rotation (angle addition formula: rotate the
   previous position's (cos,sin) by a precomputed per-freq delta) or a
   lookup table. Maybe 30-50ms/token.
10. **Verify after any change** — rebuild the host harness (compile .ino
    with `-DARDUINO_HOST_TEST`, feed same weights.bin + prompt, diff logits
    vs PyTorch). A speed hack that breaks math should fail here, not on
    the board.
## Longer generations (currently capped at T_MAX=80 bytes)

1. **Sliding-window attention (ring buffer)** — when t reaches T_MAX, evict
   the oldest position from kr_cache/x_cache (circular index) and keep
   generating forever with the last ~80 bytes as context. RoPE positions
   keep counting up (model trained at context 512, so absolute positions
   up to 512 are in-distribution). This is the highest-value change:
   unbounded story length, no extra memory.
2. **int8 x_cache** — x_cache is float (T_MAX·D·4·6 = 590KB at T_MAX=80).
   Quantizing per-position to int8 frees ~440KB → T_MAX +6 or so. Minor.
3. **int4 kr_cache** — halves the big cache (5.5MB → 2.8MB) → T_MAX ~160,
   or free PSRAM for hot weights (speed idea #5). Needs a quality check on
   host first: int4 attention keys may hurt; measure logit error.
4. **Rebalance T_MAX vs PSRAM** — kr_cache costs ~69KB per position at
   D=192/mult=60. T_MAX=100 ≈ 7.2MB total caches: fits, barely, if
   nothing else moves into PSRAM. Combine with idea 2 for headroom.
5. **Trained context is 512** — if training used block_size=512, nothing
   architectural stops T_MAX=512 except PSRAM; with int4 kr_cache and
   int8 x_cache it's ~3.7MB → actually reachable. Full-length stories.

## Pipeline improvements (non-speed)

- `flash.sh` reflashes weights even when unchanged (~50s) — add a hash check
- Serial REPL has no way to tune temperature / length at runtime — trivial
  to add `!temp 0.5`-style commands
- T_MAX=80 caps stories at 80 bytes; raising it costs ~71KB PSRAM per byte
  (T_MAX=100 ≈ 7.1MB, tight but possible if weights stay in flash)
- Prompt processing (1.8s/byte) could show a progress indicator
- Export could store per-row (not per-tensor) int8 scales for slightly
  better quality — measure first; current error (0.019 logits) is tiny
- Chat driver script on the laptop side (nicer than raw serial monitor);
  remember: open port with `dtr=False, rts=False` or the board resets into
  bootloader (that cost us an hour)

## Training handoff

- D=192 run in progress (separate session), target step 10000
- `step_000200.pt` (loss 2.05) is what's on the board now — replace when
  a better checkpoint lands

  ## Final thoughts
  A few things I'd want the next agent (or future us) to know and try:

Testing — the philosophy: hardware debugging is miserable (2-minute flash cycles, buffered serial, boards resetting when you look at them wrong). The host harness exists so that math bugs never reach the board — compile the same .ino on the laptop, feed identical weights, diff against PyTorch. What I'd add to it:

1. Generation parity test — my check only verified one forward pass. Run 100+ tokens of greedy generation on host-C vs PyTorch; both are deterministic, so outputs should be byte-identical. This catches incremental-cache bugs that only appear at longer sequence lengths (the scariest bug class in what I wrote).
2. Quantization quality measurement — compute validation loss with fp32 vs int8 weights in PyTorch. Currently we know logit error is 0.019 but not what that costs in perplexity. This also tells you upfront whether int4 kr_cache (the T_MAX=512 enabler) is viable or ruinous.
3. On-device profiling — before anyone writes SIMD, wrap each stage (encoder / attention / encoder_v / decoder / lm_head) in micros() timers and print a per-token breakdown. Optimize what's actually slow; my "flash bandwidth dominates" claim is an educated guess, not a measurement.
4. Soak test — let it generate for an hour. Watchdog timers, heap fragmentation, and PSRAM edge cases only show up over time. Related: the multi-second compute in loop() may eventually trip the task watchdog — if it ever reboots mid-story, that's the suspect.

Explorations I'd find interesting:
- Digital twin chats — the host harness runs the same weights.bin ~100x faster than the board. Tune temperature/top-k/repetition behavior there, then hardcode the winners into firmware. Sampling settings matter enormously for tiny models.
- Standalone toy — the XIAO Sense has no display, but a $4 I2C OLED + a button = a pocket story machine that needs no laptop. That's the demo that makes people gasp.
- Checkpoint ladder — keep step 200 next to the final weights and let a serial command switch between them. "Watch the same chip get smarter" is a great artifact of this project.
- Commit the esp32 work — it's all untracked in git right now. Worth a commit before anyone else starts editing.
