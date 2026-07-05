#!/usr/bin/env python3
"""
gen_hex_workload.py
====================
Converts real LLM INT8-quantised Q/K attention activations into the .hex
format consumed by edsa_hex_file_seq in the UVM testbench.

Supported LLM sources
---------------------
1. Hugging Face transformers (GPT-2, LLaMA-2, BERT, Mistral, etc.)
   Requires: pip install transformers torch
2. ONNX Runtime INT8 exported model
   Requires: pip install onnxruntime numpy
3. Raw numpy .npy / .npz activation dump

Output
------
  workload/q_tokens.hex   – one INT8 byte per line (hex, no prefix)
  workload/k_tokens.hex   – same format
  workload/meta.json      – sequence length, model name, layer index

Usage
-----
  # From HuggingFace (downloads model weights):
  python gen_hex_workload.py --source hf --model gpt2 --layer 0 --seq_len 64

  # From raw numpy dump:
  python gen_hex_workload.py --source npy --q_path q.npy --k_path k.npy

  # Synthetic INT8 (no model download needed – for quick bring-up):
  python gen_hex_workload.py --source synthetic --seq_len 128 --seed 42
"""

import argparse
import json
import os
import struct
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("workload")


def to_int8_hex_file(array: np.ndarray, path: Path) -> None:
    """Write a 1-D array of INT8 values as hex lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(array.flatten().astype(np.int16), -128, 127).astype(np.int8)
    with open(path, "w") as f:
        f.write("// EDSA hex workload – INT8 signed activations\n")
        f.write(f"// Shape: {arr.shape}  dtype: int8\n")
        for byte in arr:
            # Reinterpret as unsigned for hex formatting
            ubyte = struct.unpack("B", struct.pack("b", int(byte)))[0]
            f.write(f"{ubyte:02X}\n")
    print(f"  Wrote {len(arr)} tokens → {path}")


def source_synthetic(args) -> tuple[np.ndarray, np.ndarray]:
    """Generate random INT8 Q/K activations with realistic statistics."""
    rng = np.random.default_rng(args.seed)
    # LLM attention activations after INT8 quantisation are approximately
    # Normal(0, 20) clipped to [-128, 127]
    q = rng.normal(0, 20, size=(args.seq_len, args.head_dim)).astype(np.int8)
    k = rng.normal(0, 20, size=(args.seq_len, args.head_dim)).astype(np.int8)
    print(f"[synthetic]  Q shape={q.shape}  K shape={k.shape}")
    return q, k


def source_npy(args) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-extracted activations from .npy / .npz files."""
    q = np.load(args.q_path)
    k = np.load(args.k_path)
    # Accept (seq_len, head_dim) or (batch, seq_len, head_dim) → take batch[0]
    if q.ndim == 3:
        q = q[0]
    if k.ndim == 3:
        k = k[0]
    # Quantise to INT8 if not already
    if q.dtype != np.int8:
        scale_q = 127.0 / (np.abs(q).max() + 1e-9)
        q = (q * scale_q).clip(-128, 127).astype(np.int8)
    if k.dtype != np.int8:
        scale_k = 127.0 / (np.abs(k).max() + 1e-9)
        k = (k * scale_k).clip(-128, 127).astype(np.int8)
    print(f"[npy]  Q shape={q.shape}  K shape={k.shape}")
    return q, k


def source_huggingface(args) -> tuple[np.ndarray, np.ndarray]:
    """
    Hook into a HuggingFace transformer model, run a forward pass, and
    extract the raw Q/K projection outputs from a specified attention layer.
    Uses hooks – no model surgery required.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise SystemExit(
            "transformers and torch are required.\n"
            "  pip install transformers torch"
        )

    print(f"[hf]  Loading model '{args.model}' ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModel.from_pretrained(args.model, output_attentions=True)
    model.eval()

    # Use a realistic English sentence as the workload token stream
    prompt = (
        "The Event-Driven Spiking Attention architecture achieves massive "
        "MAC sparsity by filtering predictable tokens before they reach the "
        "attention computation, reducing energy consumption significantly."
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        max_length=args.seq_len, truncation=True, padding="max_length"
    )

    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                captured[name] = out[0].detach().cpu().numpy()
            else:
                captured[name] = out.detach().cpu().numpy()
        return hook

    # Attach hooks to the target layer's Q and K projection
    try:
        # GPT-2 / GPT-J style
        layer = model.transformer.h[args.layer].attn
        layer.c_attn.register_forward_hook(make_hook("qkv"))
        with torch.no_grad():
            model(**inputs)
        qkv = captured["qkv"]          # shape (1, seq, 3*d_model)
        d   = qkv.shape[-1] // 3
        q_f = qkv[0, :, :d]            # (seq, d_model)
        k_f = qkv[0, :, d:2*d]

    except AttributeError:
        try:
            # BERT style
            layer = model.encoder.layer[args.layer].attention.self
            layer.query.register_forward_hook(make_hook("q"))
            layer.key.register_forward_hook(make_hook("k"))
            with torch.no_grad():
                model(**inputs)
            q_f = captured["q"][0]
            k_f = captured["k"][0]
        except AttributeError:
            raise SystemExit(
                f"Cannot auto-detect Q/K projections for model '{args.model}'.\n"
                "Use --source npy and provide pre-extracted activations."
            )

    # Per-tensor symmetric INT8 quantisation (standard PTQ)
    def quantise(arr):
        scale = 127.0 / (np.abs(arr).max() + 1e-9)
        return (arr * scale).clip(-128, 127).astype(np.int8)

    # Extract single attention head (head 0 of first args.head_dim dimensions)
    head_dim = min(args.head_dim, q_f.shape[-1])
    q = quantise(q_f[:args.seq_len, :head_dim])
    k = quantise(k_f[:args.seq_len, :head_dim])
    print(f"[hf]  Q shape={q.shape}  K shape={k.shape}")
    return q, k


def write_meta(args, q: np.ndarray, k: np.ndarray) -> None:
    meta = {
        "model":    getattr(args, "model", "synthetic"),
        "source":   args.source,
        "layer":    getattr(args, "layer", 0),
        "seq_len":  q.shape[0],
        "head_dim": q.shape[1] if q.ndim > 1 else 1,
        "q_file":   str(OUTPUT_DIR / "q_tokens.hex"),
        "k_file":   str(OUTPUT_DIR / "k_tokens.hex"),
    }
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta → {OUTPUT_DIR / 'meta.json'}")


def main():
    ap = argparse.ArgumentParser(description="Generate EDSA hex workload from LLM")
    ap.add_argument("--source",   choices=["synthetic", "npy", "hf"],
                    default="synthetic")
    ap.add_argument("--model",    default="gpt2",   help="HuggingFace model name")
    ap.add_argument("--layer",    type=int, default=0, help="Attention layer index")
    ap.add_argument("--seq_len",  type=int, default=64)
    ap.add_argument("--head_dim", type=int, default=64,
                    help="Tokens per head written to hex (flattened)")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--q_path",   default="q.npy")
    ap.add_argument("--k_path",   default="k.npy")
    args = ap.parse_args()

    print(f"\n=== EDSA Hex Workload Generator ===")
    print(f"  source={args.source}  seq_len={args.seq_len}")

    if args.source == "synthetic":
        q, k = source_synthetic(args)
    elif args.source == "npy":
        q, k = source_npy(args)
    elif args.source == "hf":
        q, k = source_huggingface(args)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    # For the UVM driver each token is a single INT8 scalar.
    # Flatten head_dim dimension: each row becomes head_dim consecutive tokens.
    q_flat = q.flatten()
    k_flat = k.flatten()

    to_int8_hex_file(q_flat, OUTPUT_DIR / "q_tokens.hex")
    to_int8_hex_file(k_flat, OUTPUT_DIR / "k_tokens.hex")
    write_meta(args, q, k)
    print("\nDone. Run UVM sim with: xsim edsa_sim +UVM_TESTNAME=edsa_hex_test")


if __name__ == "__main__":
    main()
