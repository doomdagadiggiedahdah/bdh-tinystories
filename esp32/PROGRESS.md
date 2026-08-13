# BDH on ESP32-S3 — Progress

**Status (2026-08-13): WORKING.** The BDH model runs on the XIAO ESP32-S3 Sense,
generating text over serial at **0.95 tok/s** (was 0.49 — see the speed
section and `EXPERIMENTS.md`).

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
| `bdh_tinystories/build_opt.h` | `-O3` for the sketch (Arduino defaults to `-Os`; this is worth 1.64x) |
| `flash.sh` | compile + upload firmware + flash weights |
| `bench.py` | drive the REPL over serial, print the per-stage profile |
| `EXPERIMENTS.md` | append-only log of speed experiments, including dead ends |

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

Point it at a checkpoint. That's the whole workflow — it exports, finds the
port, and flashes:

```bash
esp32/update_weights.sh step_000999.pt
```

Then talk to it:
```bash
uv run --with pyserial esp32/bench.py --reset --prompt "Once upon a time"
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200
```

Notes on the script:
- The port is a **flag**, not a positional: `--port /dev/ttyACM1`. (It used
  to be the second positional arg, which meant a shell brace expansion like
  `step_000{999,800}.pt` silently passed the second checkpoint as the port.)
- Pass several checkpoints and it refuses rather than guessing — with
  `step_000{999,800}.pt` the "last" one is the *older* checkpoint.
- Skips the ~50s write if the exported weights are byte-identical to last
  time; `--force` overrides.
- Fails with a clear message if a serial monitor is holding the port.

Only needed if you edited the `.ino` (weights alone don't need a firmware
rebuild):
```bash
esp32/flash.sh /dev/ttyACM0        # compiles, uploads firmware, flashes weights
```

**Checkpoints must be D=192 / mlp_mult=60** (~6.7M params, 6.42MB int8) to
fit the 6.9MB flash region. The older D=256 / mlp_mult=50 run (9.96M params)
needs ~9.9MB and cannot run on this board.

Host-side sanity check before flashing (compares C forward pass vs PyTorch —
harness lives in the job tmp dir, rebuild from `PROGRESS` notes if gone):
compile the .ino with `-DARDUINO_HOST_TEST` stubs, run same prompt through
both, compare logits. Worth redoing after any .ino math change.

## Speed: where the time goes & what to try

**Now 0.95 tok/s** (~1.05s/token), up from 0.55 measured / 0.49 previously
reported. Two changes got there: `-O3` (1.64x) and hoisting the encoder
tensor into PSRAM (+12%). Full history, including three dead ends, is in
`EXPERIMENTS.md` — **read it before trying anything below.**

Measured per-stage breakdown at 0.95 tok/s (`PROFILE 1` in the .ino prints
this after every generation):

```
encoder      543 ms/tok  52.9%
encoder_v    167 ms/tok  16.3%
attention    147 ms/tok  14.4%
rope+quant   110 ms/tok  10.7%
decoder       56 ms/tok   5.4%
lm_head+embed  3 ms/tok   0.3%
```

The encoder dominates because weights are shared across layers: all 6
layers re-read the whole 2.11MB tensor, ~13.3MB of reads per token.

**What the bottleneck actually is.** Not flash bandwidth, not arithmetic
throughput — both were tested and neither moved it (see `EXPERIMENTS.md`).
The remaining explanation is cache-miss latency against a 13.3MB/token
working set streamed through a 32KB cache. **So the thing to optimize is
bytes read per token.** Ideas that make bytes move *faster* have twice
disappointed; ideas that make bytes *fewer* are untested and promising.

Ideas, reprioritized by that finding:

1. **int4 encoder weights** — halves the encoder's 13.3MB/token, hitting
   the actual constraint. Best remaining idea. Measure quality on host
   first: int4 per-row scales on the encoder only, check logit drift and
   validation loss before flashing.
2. **Exploit more sparsity** — ~74% of encoder_v rows and ~93% of decoder
   rows are already skipped via the ReLU gate (measured). The encoder
   itself can't use this — its output *defines* the sparsity pattern, so
   all N rows must be read before you know which are zero. A cheap
   *predictor* of which neurons will fire (low-rank probe, or an int4
   first pass to threshold) would break that circularity and is the only
   way to cut the encoder's row count.
3. **Both cores** — inference is single-core; a FreeRTOS task on core 0
   could take half the heads. The one "go faster" idea still standing,
   since it adds a second cache path rather than a faster one.
4. **Cheaper RoPE trig** — rope+quant is 110ms (10.7%), ~34k `cosf`/`sinf`
   per token. Incremental rotation (angle-addition from the previous
   position) or a LUT. Straightforward ~50-80ms.
5. **Skip lm_head during prompt** — only needed for the last prompt byte.
   Tiny (3ms/tok) but free.
6. **esp-dsp SIMD (`dsps_dotprod_s8`)** — was idea #1 in the old list.
   Demoted: run #4 showed the core isn't starved on arithmetic, so SIMD
   likely helps less than the 2-4x once hoped. Try after int4.
7. **Verify after any change** — rebuild the host harness (compile .ino
   with `-DARDUINO_HOST_TEST`, feed same weights.bin + prompt, diff logits
   vs PyTorch). A speed hack that breaks math should fail here, not on
   the board. Nothing in this session changed the math (the one change
   that did, run #4, was reverted), so no re-verification was needed —
   but int4 will absolutely need it.

Dead ends — do not retry, reasons in `EXPERIMENTS.md`:
- **QIO flash mode** — impossible with OPI PSRAM; they share SPI pins, the
  core silently forces DIO.
- **int8 activation quantization** — measurably *slower* than float on the
  LX7, and costs accuracy.

## Benchmarking

`esp32/bench.py` drives the REPL over serial and prints the profile
(opens the port with DTR/RTS deasserted, which is what stops the board
from resetting into the bootloader):

```bash
uv run --with pyserial esp32/bench.py --reset --prompt "Once upon a time"
```
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
