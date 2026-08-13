"""Export BDH checkpoint to int8 binary for ESP32 flash."""
import struct
import sys
import numpy as np
import torch

def export(checkpoint_path, output_path="weights.bin"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Extract model dimensions from weight shapes
    embed_w = sd["embed.weight"]       # [vocab_size, D]
    encoder_w = sd["encoder"]          # [nh, D, N]
    encoder_v_w = sd["encoder_v"]      # [nh, D, N]
    decoder_w = sd["decoder"]          # [nh*N, D]
    lm_head_w = sd["lm_head"]          # [D, vocab_size]

    vocab_size, D = embed_w.shape
    nh = encoder_w.shape[0]
    N = encoder_w.shape[2]
    n_layer = 6  # BDH default, shared weights

    print(f"Model: D={D}, nh={nh}, N={N}, n_layer={n_layer}, vocab={vocab_size}")

    def quantize(tensor, name):
        t = tensor.float().numpy().ravel()
        absmax = np.max(np.abs(t))
        scale = absmax / 127.0 if absmax > 0 else 1.0
        q = np.clip(np.round(t / scale), -127, 127).astype(np.int8)
        print(f"  {name}: {tensor.shape} -> {len(q)} bytes, scale={scale:.6f}")
        return scale, q

    # Layout v2: encoders transposed [nh, D, N] -> [nh, N, D] so the ESP32's
    # per-neuron inner loop reads D consecutive bytes from flash.
    e_scale, e_q = quantize(embed_w, "embed")
    enc_scale, enc_q = quantize(encoder_w.transpose(1, 2).contiguous(), "encoder^T")
    encv_scale, encv_q = quantize(encoder_v_w.transpose(1, 2).contiguous(), "encoder_v^T")
    dec_scale, dec_q = quantize(decoder_w, "decoder")
    lm_scale, lm_q = quantize(lm_head_w, "lm_head")

    with open(output_path, "wb") as f:
        # 64-byte header
        header = struct.pack("<I I I I I I 5f",
            0x42444802,  # magic (layout v2: transposed encoders)
            D, nh, N, n_layer, vocab_size,
            e_scale, enc_scale, encv_scale, dec_scale, lm_scale)
        header += b'\x00' * (64 - len(header))
        f.write(header)
        # Weight data
        f.write(e_q.tobytes())
        f.write(enc_q.tobytes())
        f.write(encv_q.tobytes())
        f.write(dec_q.tobytes())
        f.write(lm_q.tobytes())

    total = 64 + sum(len(q) for q in [e_q, enc_q, encv_q, dec_q, lm_q])
    print(f"\nWrote {output_path}: {total:,} bytes ({total/1024/1024:.2f} MB)")
    if total > 6.9 * 1024 * 1024:
        print(f"WARNING: weights exceed 6.9MB flash partition!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> [output.bin]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "esp32/weights.bin"
    export(sys.argv[1], out)
