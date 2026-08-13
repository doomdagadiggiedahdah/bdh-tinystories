# ESP32 speed experiments — append-only log

Chronological record of what was tried on the board, **including what didn't
work**. `PROGRESS.md` is the curated state + backlog; this file is the raw
history, so a future agent doesn't re-run a dead end.

Each data point costs a ~4 minute compile/flash/generate cycle on real
hardware. Please add rows rather than rewriting them.

Board: XIAO ESP32-S3 Sense, 8MB flash / 8MB OPI PSRAM.
Model: D=192, nh=4, N=2880, 6 shared layers. Prompt "Once upon a time",
80 total bytes (16 prompt + 64 generated), temperature 0.8.

## Results

| # | Change | tok/s | Encoder ms/tok | Note |
|---|---|---|---|---|
| 0 | Baseline (`-Os`, DIO flash) | 0.55 | 1052 | Instrumented build. Pre-existing doc claimed 0.49 — the gap is run-to-run noise / prompt differences, not a regression. |
| 1 | `+ -O3` via `build_opt.h` | 0.90 | 608 | **1.64x.** Sketch grew 300389 → 300857 bytes. Cheapest win available. |
| 2 | `+ FlashMode=qio` | 0.90 | 607.8 | **No effect — dead end, don't retry.** See below. |
| 3 | `+ encoder hoisted to PSRAM` | 0.95 | 543 | **Kept, but only +12%** — far below the ~4x the bandwidth ratio predicted. See "the bandwidth theory is wrong" below. |
| 4 | `+ int8 activation quantization` | 0.90 | 588 | **Slower. Reverted — don't retry.** See below. |

**Current best: run #3 — `-O3` + encoder in PSRAM, float dot products.
0.55 → 0.95 tok/s, 1.7x overall.**

## First per-stage profile (at `-Os`, run #0)

This is what redirected the whole effort — the encoder, not attention, is
the cost centre:

```
encoder      1052 ms/tok  59.4%
encoder_v     281 ms/tok  15.9%
attention     241 ms/tok  13.6%
rope+quant    118 ms/tok   6.6%
decoder        78 ms/tok   4.4%
lm_head+embed   3 ms/tok   0.2%
```

The encoder is expensive because weights are **shared across layers**: all
6 layers re-read the entire `nh*N*D` = 2.11MB tensor (activations differ,
weights don't), so it costs ~13.3MB of reads per token.

## A model of the machine

Working back from 13.3MB of encoder reads in 1052ms gives **~12.6 MB/s**
effective flash bandwidth at `-Os`. That same figure then predicted
`encoder_v` and `decoder` times almost exactly, once ReLU row-skipping was
accounted for. This is reusable: *estimate a stage's cost as bytes-read /
bandwidth*, and it holds.

Measured ReLU sparsity, implied by that fit (a model property, worth having):

- **encoder_v: ~74% of rows skipped** (`x_sparse[n] == 0`)
- **decoder: ~93% of rows skipped** (needs both `x_sparse > 0` and the
  gate `> 0`)

That much sparsity is why "exploit more sparsity" in the `PROGRESS.md`
backlog is promising — but note the encoder itself *cannot* use it: its
output is what defines the sparsity pattern, so all N rows must be read
before you know which are zero.

## Correction worth preserving

From the 12.6 MB/s figure I first concluded "flash-bandwidth-bound, not
compute-bound." **`-O3` buying 1.64x proved that overstated.** At `-Os`,
scalar float compute was co-limiting; the bandwidth number was polluted by
compute stalls. Only *after* `-O3` — 13.3MB at ~22 MB/s, against DIO
80MHz's ~20MB/s theoretical ceiling — is it genuinely flash-bound.

The lesson, which outlives the numbers: **a bandwidth estimate derived from
an unoptimized build cannot distinguish "memory-bound" from "compute-bound."**
Optimize the compute first, *then* measure bandwidth.

## Dead end: QIO flash mode (run #2)

`arduino-cli board details` lists `FlashMode=qio` (QIO 80MHz) as the
*default* for this board, and the boot ROM was printing `mode:DIO` — which
looked like a free 2x on a flash-bound workload (4 data lines vs 2).

It isn't. Compiling and uploading with `FlashMode=qio` (which rewrites the
bootloader, where flash mode lives in the image header at offset 0) produced
a board that **still boots `mode:DIO`**, with timings identical to run #1 to
within 0.2ms.

Cause: on ESP32-S3, **octal (OPI) PSRAM and QIO flash contend for the same
SPI data lines.** With `PSRAM=opi` — which this sketch requires, as it needs
all 8MB — the core forces flash to DIO regardless of the requested mode.

**Consequence:** flash bandwidth is permanently capped at ~20MB/s here. The
only way to go faster on the encoder is to stop reading it from flash, which
is what run #3 tries: OPI PSRAM is ~80MB/s, ~4x DIO flash.

## The bandwidth theory is wrong (runs #3 and #4)

Two experiments in a row failed to move the needle, and together they rule
out both of the obvious explanations for why the encoder is slow.

**Run #3 — moving the encoder to PSRAM gave only 12%** (608 → 543ms), not
the ~4x implied by OPI PSRAM's ~80MB/s vs DIO flash's ~20MB/s. At 543ms for
13.3MB the encoder reads at ~24.5 MB/s — barely above flash. If raw
bandwidth were the constraint, PSRAM would have shown it. The likely real
constraint is **cache-miss servicing**: both flash and PSRAM are external
memory behind the same 32KB cache, and streaming 2.11MB per layer misses on
essentially every line. The backing store changes, the miss path doesn't.

(Kept anyway: 12% is free at runtime, the fallback is safe, and it makes the
encoder independent of flash mode.)

**Run #4 — int8 activations made it slower** (543 → 588ms encoder, 167 →
177ms encoder_v). The hypothesis was that `dot_f32_i8`'s per-element
`int8 -> float` convert dominated, so quantizing `cur_x`/`cur_ykv` to int8
and accumulating in int32 would remove it. It backfired: the LX7's FPU does
a pipelined 1-cycle float FMA, while the integer path needs sign-extend +
32-bit multiply and the compiler vectorizes it less well. Integer math is
*not* automatically cheaper here. It also costs accuracy, so there is no
version of this worth keeping.

**Where that leaves it.** The encoder is not flash-bound (run #2 + #3), not
bandwidth-bound (#3), and not int-conversion-bound (#4). What remains is
cache-miss latency on a 13.3MB/token working set. The lever that follows
from that is **reducing bytes read**, not moving or converting them faster:

- **int4 encoder weights** — halves the encoder's 13.3MB to 6.6MB. Directly
  attacks the actual constraint, and is the single most promising remaining
  idea. Needs a host-side quality check first (see `PROGRESS.md`).
- **esp-dsp SIMD (`dsps_dotprod_s8`)** — worth a try, but note run #4 is
  evidence the core is *not* starved on arithmetic, so temper expectations;
  SIMD helps most when compute-bound.
- Attention (147ms, 14%) still uses a float dot over N=2880 against data
  that is *already* int8 in `kr_cache`. Left alone deliberately to keep run
  #4 attributable — but given #4's result, expect little from changing it.

## PSRAM budget (for run #3)

Tight enough that the margin is worth recording rather than recomputing:

```
kr_cache   6*4*80*2880  = 5,529,600 B
x_cache    6*80*192*4   =   368,640 B
kr_scale   6*4*80*4     =     7,680 B
scratch    (~cur_*)     =   118,000 B
                          -----------
                          6,024,448 B = 5.75 MB   (~5.86 MB with heap overhead)

PSRAM total               8,388,608 B = 8.00 MB
free                                  ~2.25 MB
encoder    4*2880*192   = 2,211,840 B = 2.11 MB   -> fits, ~150KB to spare
```

So the encoder fits **only just**, and only at T_MAX=80. Raising T_MAX costs
~69KB/position and will evict it.

**This margin bites.** After flashing a different checkpoint the board came
up with `ERROR: bad weights (magic mismatch or PSRAM alloc failed)` — the
weights were fine (header verified 0x42444802, D=192, correct size); the
PSRAM allocation had intermittently failed at 7.98MB of 8.00MB used.

The first fallback was inadequate: it only retried if `kr_cache` failed, so
if the hoist succeeded and a *later, smaller* allocation failed, it returned
false and bricked the boot. It's now a two-attempt loop — try the whole
buffer set with the encoder hoisted, and on *any* failure free everything
and retry reading the encoder from flash. A tight budget now costs 12%
speed, never a dead board.

The error messages are also split: a magic mismatch and an allocation
failure need opposite fixes (reflash weights vs lower T_MAX), and
conflating them cost a diagnosis cycle.

**If you raise T_MAX, the encoder hoist is what stops fitting first** — and
since it's only worth 12%, trading it for longer stories is a reasonable
deal. The firmware makes that trade automatically.
