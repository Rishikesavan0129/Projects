#!/usr/bin/env python3
"""
gen_stimulus.py

Generates realistic Q*K^T attention-logit stimulus for the hardware
FlashAttention engine, quantizes it to int8 (matching the RTL's X_WIDTH=8),
and writes:
  - stimulus.hex : UVM-readable score stream  (row_id elem_id hex_byte)
  - golden.csv   : per-row golden softmax denominator computed in double
                   precision Python, for the UVM scoreboard to compare
                   hardware results against.

"Real world" data here means actual Q/K vectors sampled from a small
transformer-like embedding distribution (not hand-picked corner-case
numbers), scaled and quantized the way a real INT8 attention accelerator's
front end would feed this block. This lets the determinism/ML script
downstream evaluate the hardware against statistically realistic inputs.

IMPORTANT ASSUMPTION (must match RTL):
The hardware computes a *base-2* exponential (2^F approx), not e^x. This
implies scores fed to the hardware are expected to already be pre-scaled
by log2(e) upstream (a standard trick: softmax(x) = softmax(x*log2(e))
in base-2 form). This script performs that scaling before quantization so
the golden model and the hardware are computing the same thing.
"""

import numpy as np
import os

# -------------------- Configuration (must match RTL params) ---------------
NUM_TILES   = 16
TILE_DEPTH  = 256
ELEMS_PER_ROW = NUM_TILES * TILE_DEPTH   # = 4096, matches max seq len in spec
NUM_ROWS    = 8          # number of independent attention rows to simulate
D_MODEL     = 64         # embedding dim for synthetic Q/K vectors
SEED        = 1234

OUT_DIR     = "/home/claude/flash_attn_hw/sim"
STIM_PATH   = os.path.join(OUT_DIR, "stimulus.hex")
GOLDEN_PATH = os.path.join(OUT_DIR, "golden.csv")

LOG2_E = np.log2(np.e)
Q_FRAC_BITS = 16  # Q16.16 domain assumed for sum/recip raw values (see scoreboard)


def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    """Symmetric quantization to signed int8 with saturation."""
    q = np.round(x * scale)
    return np.clip(q, -128, 127).astype(np.int8)


def gen_row_scores(rng: np.random.Generator) -> np.ndarray:
    """
    Simulate one row of Q*K^T logits the way a real attention head would
    produce them: random unit-norm query against `ELEMS_PER_ROW` random
    unit-norm keys in D_MODEL dims, scaled by 1/sqrt(D_MODEL) (standard
    attention scaling), then converted to base-2 domain via log2(e).
    """
    q = rng.normal(size=(D_MODEL,))
    q = q / np.linalg.norm(q)
    k = rng.normal(size=(ELEMS_PER_ROW, D_MODEL))
    k = k / np.linalg.norm(k, axis=1, keepdims=True)

    logits = (k @ q) / np.sqrt(D_MODEL)      # standard scaled dot-product
    logits_base2 = logits * LOG2_E           # convert e-domain -> 2-domain

    # Real attention logits are small (~O(1-3)); scale up to use the int8
    # dynamic range reasonably (this mirrors a real quantization calibration
    # step, not an arbitrary corner-case injection).
    scale = 20.0
    scores_i8 = quantize_int8(logits_base2, scale)
    return scores_i8, scale


def golden_softmax_denom(scores_i8: np.ndarray, scale: float):
    """
    Compute the exact (double-precision) base-2 softmax denominator for the
    *quantized* int8 scores (so the golden model reflects the same
    quantization the hardware sees), then express it in the Q16.16 raw
    domain the scoreboard expects.
    """
    x = scores_i8.astype(np.float64)  # already in the "x" domain hardware sees
    m = np.max(x)
    denom = np.sum(np.exp2(x - m))    # exact base-2 softmax denominator

    sum_raw = denom * (1 << Q_FRAC_BITS)
    # Reciprocal assumed Q16.16 as well (see nr_reciprocal format note):
    # recip_raw represents (1/denom) scaled into the same raw domain.
    recip_raw = (1.0 / denom) * (1 << Q_FRAC_BITS) * (1 << Q_FRAC_BITS) / (1 << Q_FRAC_BITS)
    # i.e. recip_raw = (1<<16)/denom, consistent with sum_raw/denom = 65536
    recip_raw = (1 << Q_FRAC_BITS) / denom

    return float(m), sum_raw, recip_raw


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    with open(STIM_PATH, "w") as stim_f, open(GOLDEN_PATH, "w") as gold_f:
        for row_id in range(NUM_ROWS):
            scores_i8, scale = gen_row_scores(rng)

            for elem_id, val in enumerate(scores_i8):
                # two's-complement byte as 2 hex chars
                byte_val = int(val) & 0xFF
                stim_f.write(f"{row_id} {elem_id} {byte_val:02x}\n")

            m, sum_raw, recip_raw = golden_softmax_denom(scores_i8, scale)
            gold_f.write(f"{row_id},{m:.6f},{sum_raw:.6f},{recip_raw:.6f}\n")

            print(f"row {row_id}: max={m:.2f} elems={len(scores_i8)} "
                  f"golden_sum_raw={sum_raw:.1f} golden_recip_raw={recip_raw:.4f}")

    print(f"\nWrote stimulus -> {STIM_PATH}")
    print(f"Wrote golden   -> {GOLDEN_PATH}")
    print(f"Total rows: {NUM_ROWS}, elements/row: {ELEMS_PER_ROW}")


if __name__ == "__main__":
    main()
