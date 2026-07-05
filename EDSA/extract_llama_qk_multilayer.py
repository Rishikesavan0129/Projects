"""
Extract real Q/K activations from Llama 3.2 1B across MULTIPLE layers,
writing one q_tokens.hex/k_tokens.hex pair per layer for a layer-sweep
benchmark. RTL/DUT is unchanged -- this just produces more test vectors.

Usage:
    huggingface-cli login
    python extract_llama_qk_multilayer.py --layers 0 4 8 12 15 \
        --out_dir ./sim_data_multilayer
"""

import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"


def quantize_int8(x: torch.Tensor):
    x = x.float().detach().cpu().numpy()
    scale = np.abs(x).max() / 127.0
    if scale == 0:
        scale = 1.0
    q = np.round(x / scale).clip(-128, 127).astype(np.int8)
    return q, scale


def write_hex(path: str, values: np.ndarray, comment: str):
    with open(path, "w") as f:
        f.write(f"// {comment}\n")
        for v in values:
            f.write(f"{int(v) & 0xFF:02X}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 15],
                     help="List of layer indices to extract")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--prompt", type=str,
                     default=("In a distant future, humanity has spread across "
                              "the stars, building vast networks of habitats "
                              "orbiting distant suns. Engineers and scientists "
                              "collaborate across light-years to solve the "
                              "greatest challenges of interstellar civilization. "))
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--out_dir", type=str, default="./sim_data_multilayer")
    args = ap.parse_args()

    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    full_prompt = args.prompt * args.repeat
    inputs = tok(full_prompt, return_tensors="pt", truncation=True, max_length=4094)
    n_tokens = inputs["input_ids"].shape[1]
    print(f"Tokenized to {n_tokens} tokens")

    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.hidden_size // num_heads
    q_head = args.head % num_heads
    k_head = args.head % num_kv_heads

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []

    for layer_idx in args.layers:
        captured = {}

        def hook_q(module, inp, out):
            captured["q"] = out

        def hook_k(module, inp, out):
            captured["k"] = out

        layer = model.model.layers[layer_idx].self_attn
        h1 = layer.q_proj.register_forward_hook(hook_q)
        h2 = layer.k_proj.register_forward_hook(hook_k)

        with torch.no_grad():
            model(**inputs)

        h1.remove()
        h2.remove()

        q_reshaped = captured["q"].view(1, n_tokens, num_heads, head_dim)
        k_reshaped = captured["k"].view(1, n_tokens, num_kv_heads, head_dim)

        q_stream = q_reshaped[0, :, q_head, :].mean(dim=-1)
        k_stream = k_reshaped[0, :, k_head, :].mean(dim=-1)

        q_q8, q_scale = quantize_int8(q_stream)
        k_q8, k_scale = quantize_int8(k_stream)

        q_path = os.path.join(args.out_dir, f"q_tokens_layer{layer_idx}.hex")
        k_path = os.path.join(args.out_dir, f"k_tokens_layer{layer_idx}.hex")

        write_hex(q_path, q_q8,
                  f"Llama-3.2-1B layer{layer_idx} head{q_head} Q stream, "
                  f"{n_tokens} tokens, scale={q_scale:.6f}")
        write_hex(k_path, k_q8,
                  f"Llama-3.2-1B layer{layer_idx} head{k_head} K stream, "
                  f"{n_tokens} tokens, scale={k_scale:.6f}")

        std_q = float(q_stream.std())
        std_k = float(k_stream.std())
        print(f"Layer {layer_idx:2d}: Q std={std_q:.4f} K std={std_k:.4f} "
              f"-> wrote {os.path.basename(q_path)}, {os.path.basename(k_path)}")

        manifest.append({
            "layer": layer_idx,
            "q_file": q_path,
            "k_file": k_path,
            "n_tokens": n_tokens,
            "q_scale": float(q_scale),
            "k_scale": float(k_scale),
        })

    import json
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")
    print("Use this manifest to drive a multi-vector UVM sequence, one "
          "vector per layer, with per-layer file paths.")


if __name__ == "__main__":
    main()
