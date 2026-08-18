#!/usr/bin/env python3
"""
Interactive REPL for a trained BDH TinyStories model.

Type the beginning of a story and watch the model continue it.
Press Ctrl+C to interrupt generation, Ctrl+D to quit.

Usage:
    python chat.py                                    # uses latest checkpoint
    python chat.py --checkpoint checkpoints/step_010000.pt
    python chat.py --temperature 0.8 --top_k 5       # adjust creativity
"""

import argparse
import glob
import os
import sys

import bdh
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint file."""
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    config = bdh.BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        mlp_internal_dim_multiplier=50,
        vocab_size=256,
        dropout=0.0,  # no dropout at inference
    )
    model = bdh.BDH(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    step = checkpoint.get("step", "?")
    loss = checkpoint.get("loss", "?")
    if isinstance(loss, float):
        loss = f"{loss:.4f}"
    print(f"Loaded checkpoint: {os.path.basename(checkpoint_path)} (step {step}, loss {loss})")

    return model


def generate_streaming(model, device, prompt_text, max_new_tokens=500, temperature=1.0, top_k=3):
    """Generate tokens one at a time, printing as they're produced."""
    idx = torch.tensor(
        bytearray(prompt_text, "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)

    # Print the prompt
    sys.stdout.write(prompt_text)
    sys.stdout.flush()

    with torch.no_grad():
        for i in range(max_new_tokens):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Decode and print the new byte
            byte_val = idx_next.item()
            try:
                char = bytes([byte_val]).decode("utf-8")
            except UnicodeDecodeError:
                char = f"\\x{byte_val:02x}"
            sys.stdout.write(char)
            sys.stdout.flush()

            # See bdh.BDH.generate — the MPS allocator caches per sequence length,
            # so a long generation balloons the pool without this.
            if idx.device.type == "mps" and (i + 1) % 10 == 0:
                torch.mps.empty_cache()

    print()  # newline after generation


def main():
    parser = argparse.ArgumentParser(description="Chat with your trained BDH model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (default: latest in checkpoints/)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature (lower = more predictable, higher = more creative)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Sample from top K most likely next bytes")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum bytes to generate per response")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            print("No checkpoints found in checkpoints/. Train a model first:")
            print("  python train.py --step find_batch")
            print("  python train.py --batch_size <N> --max_iters 10000")
            sys.exit(1)

    # Load
    model = load_model(checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params on {device}")
    print(f"Settings: temperature={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
    print()
    print("Type the beginning of a story and press Enter.")
    print("The model will continue it. Ctrl+C to stop generating, Ctrl+D to quit.")
    print()

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip():
            continue

        print()
        try:
            generate_streaming(model, device, prompt, args.max_tokens, args.temperature, args.top_k)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print()


if __name__ == "__main__":
    main()
