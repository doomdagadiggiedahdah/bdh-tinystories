# Train Your First Language Model

Train a ~10M parameter model to write children's stories — from random bytes to coherent sentences in a few hours. Watch it learn to spell, form words, build sentences, and eventually write story fragments with characters and dialogue.

Built on the [BDH (Dragon Hatchling)](https://github.com/pathwaycom/bdh) architecture with [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) data.

## Quickstart with Claude Code

Open this repo in [Claude Code](https://claude.ai/code) and say:

> Set me up and start training

Claude will create a virtual environment, install dependencies, detect your GPU memory, and start training with sample logging.

## Manual setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch numpy datasets
python train.py --step find_batch        # detect max batch size
python train.py --batch_size 8 --max_iters 10000 \
  --sample_steps "0,1,2,3,4,5,10,20,50,100,500,1000,5000,10000"
```

## What you'll see

Open `samples.log` to watch the model learn in real time:

**Step 0** — random bytes
```
Alice was so tired when she got back home so she went\xcdj\x8a\xf4[\x9d\x9d\x92H
```

**Step 10** — discovers common letters
```
Alice was so tired when she got back home so she went e  e   eae aae ee    ae
```

**Step 50** — letter combinations emerge
```
Alice was so tired when she got back home so she wento tone se s t and tone ane
```

**Step 500** — real sentences
```
Alice was so tired when she got back home so she went to share. She wanted.
Timmy asked her friends that he walked to play. She was very happy and tried
to play with her mom and the box.
```

**Step 1300** — dialogue and characters
```
Alice was so tired when she got back home so she went home. She says, "Thank
you, Lily." Lily said, "Yes, Lily. I want to play with tall the park when you
are too stronger and soons in the park."
```

## Requirements

- Python 3.10+
- A GPU with 4GB+ VRAM (NVIDIA or Apple Silicon), or CPU (much slower)
- ~2 GB disk for the dataset

## How it works

The model reads raw bytes (no tokenizer) and predicts the next byte. It has 9.96M parameters that are **shared across all 6 layers** — so it gets 6 layers of depth for the parameter cost of 1. Training runs through 1.9 GB of children's stories, learning English from scratch: first letter frequencies, then spelling patterns, then words, grammar, and eventually story structure.

## Chat with your model

After training, start an interactive REPL:

```bash
python chat.py
```

Type the beginning of a story and watch the model continue it, one character at a time. Ctrl+C to stop, Ctrl+D to quit.

```
>>> Once upon a time there was a little
Once upon a time there was a little girl named Lily. She loved to play in the
park with her friends. One day, she saw a big red ball...
```

## Files

| File | Purpose |
|------|---------|
| `bdh.py` | Model architecture (unmodified from [pathwaycom/bdh](https://github.com/pathwaycom/bdh)) |
| `train.py` | Training script with data download, batch detection, checkpointing, sample logging |
| `chat.py` | Interactive REPL — type a prompt, model continues it |
| `samples.log` | Generated outputs at each sample step (created during training) |
| `loss.csv` | Loss over time (created during training) |
| `checkpoints/` | Model checkpoints every 1000 steps |

## License

BDH model code is from [Pathway](https://github.com/pathwaycom/bdh). Training script and samples infrastructure by the authors of this repo.
