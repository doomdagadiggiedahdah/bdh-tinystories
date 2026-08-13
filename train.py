#!/usr/bin/env python3
"""
Train a ~10M parameter BDH (Dragon Hatchling) model on TinyStories.

Byte-level language model — no tokenizer, just raw UTF-8. Learns to write
children's stories from scratch. Watch it go from random bytes to coherent
sentences in the samples.log file.

Usage:
    # Auto-detect batch size, then train:
    python train.py --step find_batch
    python train.py --step train --batch_size <result> --max_iters 10000

    # Or let the setup command handle it for you.
"""

import argparse
import csv
import os
import sys
import time
from contextlib import nullcontext

import bdh
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed evaluation prompts (from the TinyStories paper, no 6-gram overlap with dataset)
EVAL_PROMPTS = [
    "Alice was so tired when she got back home so she went",
    "Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, 'Look, Lily. A rainbow has",
    "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked",
]


def setup_device():
    """Detect the best available device and dtype."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = "float32"  # MPS doesn't support float16 scaler well
    else:
        device = torch.device("cpu")
        dtype = "float32"

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if device.type in ("cuda", "mps")
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))

    return device, dtype, ptdtype, ctx, scaler


def download_tinystories(data_path):
    """Download TinyStories from HuggingFace and concatenate to a single bytes file."""
    if os.path.exists(data_path):
        size = os.path.getsize(data_path)
        print(f"TinyStories data already exists: {size:,} bytes")
        return

    print("Downloading TinyStories from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Downloaded {len(ds)} stories, concatenating...")
    with open(data_path, "wb") as f:
        for i, example in enumerate(ds):
            if i > 0:
                f.write(b"\n")
            f.write(example["text"].encode("utf-8"))
    size = os.path.getsize(data_path)
    print(f"TinyStories written: {size:,} bytes")


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def generate_samples(model, device, prompts, max_new_tokens=200, temperature=1.0, top_k=3):
    model.eval()
    results = []
    for prompt_text in prompts:
        prompt = torch.tensor(
            bytearray(prompt_text, "utf-8"), dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            ret = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
        results.append(decoded)
    model.train()
    return results


def save_checkpoint(model, optimizer, scaler, step, loss, checkpoint_dir, max_keep=5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }, path)
    ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")])
    for old in ckpts[:-max_keep]:
        os.remove(os.path.join(checkpoint_dir, old))
    print(f"Checkpoint saved: {path}")


def find_batch_size(model, train_data, block_size, device, ctx, scaler):
    """Binary search for the largest batch size that fits in memory (forward + backward)."""
    print("Finding largest batch size that fits in memory...")
    last_good = None
    for bs in [32, 16, 8, 4, 2, 1]:
        torch.cuda.empty_cache() if device.type == "cuda" else None
        try:
            x, y = get_batch(train_data, block_size, bs, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  batch_size={bs:3d} — fits! (peak {mem:.2f} GB)")
            else:
                print(f"  batch_size={bs:3d} — fits!")
            last_good = bs
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "NVML_SUCCESS" in str(e):
                print(f"  batch_size={bs:3d} — OOM")
                torch.cuda.empty_cache() if device.type == "cuda" else None
            else:
                raise

    if last_good is None:
        print("ERROR: Even batch_size=1 doesn't fit. Need more memory or smaller block_size.")
        sys.exit(1)

    # Search upward from last_good
    best = last_good
    for bs in range(last_good + 1, 65):
        torch.cuda.empty_cache() if device.type == "cuda" else None
        try:
            model.zero_grad(set_to_none=True)
            x, y = get_batch(train_data, block_size, bs, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  batch_size={bs:3d} — fits! (peak {mem:.2f} GB)")
            else:
                print(f"  batch_size={bs:3d} — fits!")
            best = bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "NVML_SUCCESS" in str(e):
                print(f"  batch_size={bs:3d} — OOM")
                torch.cuda.empty_cache() if device.type == "cuda" else None
                break
            else:
                raise

    print(f"\nMax batch_size: {best} (with block_size={block_size})")
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Train a BDH model on TinyStories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", choices=["data", "find_batch", "train"], default="train",
                        help="data: download only. find_batch: detect max batch size. train: full training.")
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample_steps", type=str, default=None,
                        help="Comma-separated list of steps to sample at (e.g. '0,10,50,100,500,1000')")
    args = parser.parse_args()

    sample_steps_set = None
    if args.sample_steps:
        sample_steps_set = set(int(s) for s in args.sample_steps.split(","))

    device, dtype, ptdtype, ctx, scaler = setup_device()
    print(f"Device: {device}, dtype: {dtype}")

    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ~10M param config
    config = bdh.BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        mlp_internal_dim_multiplier=50,
        vocab_size=256,
        dropout=0.1,
    )

    # Download data
    data_path = os.path.join(SCRIPT_DIR, "tinystories.bin")
    download_tinystories(data_path)

    data_all = np.memmap(data_path, dtype=np.uint8, mode="r")
    split = int(0.9 * len(data_all))
    train_data = data_all[:split]
    val_data = data_all[split:]
    print(f"Train: {len(train_data):,} bytes, Val: {len(val_data):,} bytes")

    if args.step == "data":
        print("Data pipeline OK.")
        return

    # Create model
    model = bdh.BDH(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.2f}M)")
    print(f"Config: D={config.n_embd}, heads={config.n_head}, layers={config.n_layer}, mlp_mult={config.mlp_internal_dim_multiplier}")

    if args.step == "find_batch":
        best = find_batch_size(model, train_data, args.block_size, device, ctx, scaler)
        print(f"\nRecommended command:")
        print(f"  python train.py --batch_size {best} --max_iters 10000")
        return

    # === Training ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    loss_csv_path = os.path.join(SCRIPT_DIR, "loss.csv")
    samples_path = os.path.join(SCRIPT_DIR, "samples.log")

    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "timestamp"])

    LOG_FREQ = 100
    SAMPLE_FREQ = 500
    CHECKPOINT_FREQ = 1000

    def write_samples(step, loss_val=None):
        samples = generate_samples(model, device, EVAL_PROMPTS)
        with open(samples_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Step {step}")
            if loss_val is not None:
                f.write(f"  |  loss={loss_val:.4f}")
            f.write(f"\n{'='*60}\n")
            for prompt_text, completion in zip(EVAL_PROMPTS, samples):
                f.write(f"\n--- INPUT: \"{prompt_text}\" ---\n")
                f.write(f"OUTPUT: {completion}\n")
        print(f"[step {step}] Samples written to {samples_path}")

    loss_acc = 0.0
    loss_steps = 0
    best_loss = float("inf")
    stall_start_step = 0
    stall_start_loss = float("inf")

    t_start = time.time()

    # Step 0 sample (before any training)
    if sample_steps_set is not None and 0 in sample_steps_set:
        write_samples(0, loss_val=None)

    print(f"\nTraining for {args.max_iters} steps (batch_size={args.batch_size}, block_size={args.block_size})...")
    print(f"Samples will be written to: {samples_path}")
    print(f"Loss log: {loss_csv_path}\n")

    for step in range(args.max_iters):
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)

        loss_val = loss.item()
        loss_acc += loss_val
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # NaN guard
        if not np.isfinite(loss_val):
            print(f"FATAL: loss is {loss_val} at step {step}, stopping.")
            sys.exit(1)

        if step % LOG_FREQ == 0 and step > 0:
            avg_loss = loss_acc / loss_steps
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed
            bytes_per_sec = steps_per_sec * args.batch_size * args.block_size
            print(f"Step {step}/{args.max_iters} | loss {avg_loss:.4f} | {steps_per_sec:.2f} steps/s | {bytes_per_sec:.0f} bytes/s")

            with open(loss_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{avg_loss:.6f}", f"{time.time():.0f}"])

            loss_acc = 0.0
            loss_steps = 0

            # Plateau guard
            if avg_loss < best_loss:
                best_loss = avg_loss
            if step - stall_start_step >= 2000:
                improvement = (stall_start_loss - avg_loss) / max(stall_start_loss, 1e-8)
                if improvement < 0.05:
                    print(f"PLATEAU: loss improved only {improvement*100:.1f}% over 2000 steps. Stopping.")
                    break
                stall_start_step = step
                stall_start_loss = avg_loss

        # Samples — at explicit steps or regular interval
        should_sample = False
        if sample_steps_set is not None and (step + 1) in sample_steps_set:
            should_sample = True
        elif sample_steps_set is None and step > 0 and step % SAMPLE_FREQ == 0:
            should_sample = True
        if should_sample:
            write_samples(step + 1, loss_val=loss_val)

        # Checkpoints
        if step > 0 and step % CHECKPOINT_FREQ == 0:
            save_checkpoint(model, optimizer, scaler, step, loss_val, checkpoint_dir)

    elapsed = time.time() - t_start
    steps_per_sec = args.max_iters / elapsed
    bytes_per_sec = steps_per_sec * args.batch_size * args.block_size

    print(f"\n{'='*60}")
    print(f"Completed {args.max_iters} steps in {elapsed:.1f}s")
    print(f"Steps/sec: {steps_per_sec:.2f}")
    print(f"Bytes/sec: {bytes_per_sec:.0f}")

    # Final samples
    print("\n--- Final samples ---")
    samples = generate_samples(model, device, EVAL_PROMPTS)
    for prompt_text, completion in zip(EVAL_PROMPTS, samples):
        print(f"\nINPUT: \"{prompt_text}\"")
        print(f"OUTPUT: {completion}\n")

    # Save final checkpoint
    save_checkpoint(model, optimizer, scaler, args.max_iters, loss_val, checkpoint_dir)


if __name__ == "__main__":
    main()
