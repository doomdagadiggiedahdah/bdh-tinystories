# BDH TinyStories — Train Your First Language Model

This repo trains a ~10M parameter language model to write children's stories,
from random bytes to coherent sentences. The model learns directly from raw
UTF-8 bytes — no tokenizer needed.

The user may know nothing about ML. Guide them through each step, explaining
what's about to happen and why before doing it. Wait for their go-ahead at
each stage. Use TaskCreate to show progress through the pipeline.

## What's here

- `bdh.py` — the BDH (Dragon Hatchling) model architecture ([source](https://github.com/pathwaycom/bdh))
- `train.py` — training script with data download, batch size detection, checkpointing, and sample logging
- `chat.py` — interactive REPL for talking to a trained model
- `samples.log` — created during training; shows the model's output evolving step by step

## The Pipeline

When the user asks to set up, start training, or get going, create a task list
and walk them through each phase. **Before each step, explain what it does and
why, then ask for a go-ahead.** Don't just silently run commands.

### Phase 1: Environment Setup

**Explain**: "First I'll create an isolated Python environment and install
PyTorch (the ML framework), NumPy (for data handling), and the HuggingFace
datasets library (to download the training stories). This won't affect your
system Python."

```bash
uv venv .venv
source .venv/bin/activate  # or: source .venv/bin/activate.fish
uv pip install torch numpy datasets
```

After install, verify the device:
```bash
python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'MPS' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'CPU')"
```

Report what was detected. If CPU-only, set expectations: "Training will work
but take roughly 10x longer than with a GPU."

If **MPS** (Apple Silicon), read the "Apple Silicon specifics" section near the
bottom of this file before continuing — batch sizing and memory behave
differently there, and skipping it will cost you two failed training runs.
Set expectations: roughly 0.4 steps/s at batch 8, so ~90 min for 2000 steps.

### Phase 2: Download Training Data

**Explain**: "Next I'll download TinyStories — a dataset of 2.1 million short
children's stories (~1.9 GB). The model will learn to write by reading these.
The stories use simple vocabulary so a small model can make real progress."

```bash
python train.py --step data
```

Report the byte count when done.

### Phase 3: Find Your Batch Size

**Explain**: "Now I'll figure out how much your GPU can handle at once. A
'batch' is how many text chunks the model processes simultaneously — bigger
batches mean faster, more stable training, but need more memory. I'll test
increasing sizes until your GPU runs out of memory, then use the largest
that works."

```bash
python train.py --step find_batch
```

This runs a full forward+backward pass (not just forward — the backward pass
roughly doubles memory usage due to gradients).

Note it does **not** allocate AdamW's optimizer state, which real training adds
on the first `optimizer.step()`. So its answer is optimistic. On Apple Silicon
use one or two below what it reports (see "Apple Silicon specifics"); on CUDA
the reported value is usually fine.

Report the result: "Your GPU can handle batch_size=<N>, processing <N×512>
bytes per training step."

### Phase 4: Train the Model

**Explain**: "Now the main event. The model starts with ~10 million random
numbers as its 'brain' and will read through the stories, adjusting those
numbers to get better at predicting the next character. I'll log the model's
output at specific checkpoints so you can watch it learn — every single step
from 0-10 (where it goes from random garbage to discovering common letters),
then at wider intervals as it builds up to writing sentences and stories.

This will take roughly <estimate based on batch size and steps/sec> hours.
I'll run it in the background so we can keep talking."

```bash
python train.py --batch_size <BATCH> --max_iters 10000 \
  --sample_steps "0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,75,100,150,200,300,500,750,1000,2000,3000,5000,7500,10000"
```

Run this in the background so you stay responsive to the user.

On Apple Silicon, prefix the command with the memory watermarks (see "Apple
Silicon specifics") and budget ~22 s per sample step on top of training time —
a 29-entry `--sample_steps` list adds ~11 minutes.

### Phase 5: Show Early Results

Once training has been running for a minute or two, pull the first samples
from `samples.log` and show the progression. This is the magic moment. Show
it as a table tracking just the first prompt's output across steps:

| Step | Loss | Generated (after prompt) |
|------|------|-------------------------|
| 0    | ~5.5 | `\xcd\x8a\xf4[\x9d...` |
| 5    | ~3.9 | `e  e   eae aae ee` |
| 10   | ~3.3 | `a   aa a   e  a  e` |
| 20   | ~2.9 | `ee te t se t a toeot` |
| 50   | ~2.6 | `se t arit t t arirind` |

To extract this, parse `samples.log` for the first prompt's output at each
step and truncate to ~60 chars.

### Phase 6: Monitoring (only when asked or after showing early results)

IMPORTANT: Don't set up monitoring until training is running AND the user has
seen initial results, or they explicitly ask to check progress.

When the user asks to monitor:
1. Read the latest entries from `samples.log` and show a preview
2. Show loss trend from `loss.csv`
3. If they want recurring checks, offer to use `/loop`:

```
/loop 10m check training: read the last sample from samples.log, report step, loss, and truncated output
```

### Phase 7: Chat With Your Model

After training (or once there's a checkpoint), offer the interactive REPL:

**Explain**: "Your model is trained! Want to talk to it? You can type the
beginning of a story and the model will continue it. This uses the same model
that just learned from those 2 million children's stories."

```bash
python chat.py
```

Or with a specific checkpoint:
```bash
python chat.py --checkpoint checkpoints/step_010000.pt
```

## Task List Template

When starting the pipeline, create these tasks:

1. "Set up Python environment" — install uv venv + dependencies
2. "Download TinyStories dataset" — ~1.9 GB from HuggingFace
3. "Find max batch size" — test GPU memory limits
4. "Start training run" — 10k steps with sample logging
5. "Show learning progression" — display early samples table
6. "Chat with the model" — interactive REPL after training

Mark each as in_progress when starting, completed when done.

## Explaining things when asked

The user may ask questions at any point. Here are the key concepts
you should be ready to explain clearly:

### "What is a training step?"
The model sees a batch of text chunks (e.g. 8 x 512 bytes). It predicts the
next byte at each position, measures how wrong it was (loss), then adjusts its
~10M parameters slightly to be less wrong next time. One step = one batch of
predictions + one adjustment.

### "What is loss?"
How surprised the model is by the correct answer. Loss = -log(probability
assigned to the correct next byte). Random guessing over 256 bytes gives
loss ~5.5 (= ln(256)). Lower = better. Loss of 1.0 means the model assigns
roughly 37% probability to the right answer on average.

### "What is batch size?"
How many text chunks the model processes simultaneously before adjusting weights.
Bigger = more stable learning but uses more GPU memory. The find_batch step
determines the maximum your hardware can handle.

### "How does it learn without a tokenizer?"
Raw bytes. Every character is its byte value (0-255). The model has to learn
that 't','h','e' go together to form "the" — it figures out spelling from
scratch just by seeing enough examples.

### "What's special about BDH?"
The weights are shared across all 6 layers. Normal transformers have separate
weights per layer, so 6 layers = 6x the parameters. BDH reuses the same
weights, getting 6 layers of computational depth for the parameter cost of 1.

## After training completes

When training finishes (or the user asks about results):

1. Show the final samples from all 3 prompts
2. Show the loss progression summary (start -> end)
3. Point out `checkpoints/` if they want to resume or experiment later
4. Show the full progression table from samples.log — the journey from
   random bytes to stories is the main artifact of this project
5. Offer to run `chat.py` to interact with the trained model

## Apple Silicon specifics

Measured on an M4 Mac mini, 16 GB unified memory. Read before Phase 3 if the
device is MPS.

**Set both memory watermarks.** Ratios are relative to
`torch.mps.recommended_max_memory()` (12.71 GB on a 16 GB machine), not physical
RAM. The default high watermark of 1.7 allows ~21 GB, which swaps instead of
failing cleanly. Setting only the high ratio errors with
`invalid low watermark ratio` — set both.

```bash
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 \
  python train.py ...
```

**`find_batch` overshoots.** It never calls `optimizer.step()`, so it misses
AdamW's state (two extra copies of all 10M params). It reported 10; 8 is what
survives. Subtract one or two from whatever it reports.

**Generation memory trap (already fixed — don't remove it).** `bdh.py:generate`
and `chat.py:generate_streaming` call `torch.mps.empty_cache()` every 10 tokens.
This is required, not an optimization. With no KV cache, generating N bytes means
N forward passes at N different sequence lengths; the MPS allocator caches per
shape and never reuses, so one 200-byte sample grew the pool to 12 GB while
holding <50 MB live, then starved training at *any* batch size. Symptom: an OOM
during a training step that doesn't respond to lowering the batch. Diagnose by
comparing `torch.mps.current_allocated_memory()` (live) against
`torch.mps.driver_allocated_memory()` (claimed) — a large gap is allocator cache.
Calling `empty_cache()` afterwards barely helps; it must run *during* generation.
Peak 12.03 → 1.09 GB, for ~4% generation speed.

**Performance:** ~0.43 steps/s at batch 8 (~2.4 s/step), 9.7 GB peak, ~90 min for
2000 steps including ~22 s per sample point. Don't bother with bfloat16 — stable
but only 11% faster; the workload is memory-bandwidth bound. The float32 autocast
warning at startup is expected.

## Model details

- **Architecture**: BDH (Dragon Hatchling) — weight-sharing across layers
- **Parameters**: ~9.96M (shared weights mean 6 layers of depth for the cost of 1)
- **Vocab**: 256 (raw bytes, no tokenizer)
- **Config**: D=256, 4 heads, 6 layers, mlp_mult=50
- **Data**: TinyStories (2.1M children's stories, 1.9 GB)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.1)
