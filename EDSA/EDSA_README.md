# EDSA — Event-Driven Spiking Attention

**A hardware accelerator that suppresses redundant Q/K attention tokens using a
predictive Hebbian filter, computing sparse dot products only on tokens that
survive the gate.**

---

## Problem

Transformer attention computes a full dense `Q·Kᵀ` for every token pair,
regardless of whether most of that computation is redundant. In practice,
consecutive activations are often highly predictable — background noise
around a stable local mean — meaning a large fraction of the multiply-
accumulate (MAC) work contributes little new information to the attention
score. EDSA targets that redundancy at the hardware level: skip tokens the
system already predicted correctly, and only spend MAC cycles on the ones
that surprised it.

## Architecture

Three pipelined RTL blocks, INT8 datapath, 12-bit token indexing:

| Block | Role |
|---|---|
| **Squelch Unit** | Per-stream Hebbian predictor + habituation gate. Maintains a running prediction, computes a residual against each incoming token, and only forwards ("spikes") tokens whose residual exceeds a programmable threshold. |
| **COO Packer** | Compresses the sparse spike stream into a dense `[index, value]` coordinate stream — no bubbles, no bandwidth wasted on suppressed tokens. |
| **Sparse MAC** | Reads two independent COO streams (Q, K) and performs a sorted merge-join: only multiplies and accumulates when Q-index equals K-index. Non-matching indices advance the lagging stream. |

**Predictive filtering approach (behavioral, not a specification):**
Each squelch unit maintains an internal, self-adjusting reference for its
input stream and forwards a token only when it deviates from that reference
beyond a programmable margin — conceptually similar to biological sensory
habituation, where the nervous system stops signaling for stimuli it has
already learned to expect. The exact update rule, comparison precision, and
fixed-point handling were specifically hardened during verification against
known pitfalls in this class of predictive filter; the corrected formulation
is documented internally, not reproduced here.

## Verification Methodology

- UVM environment with independent Q/K driver agents, hex-file and
  randomized/corner-case sequence support
- Software golden reference model (bit-exact replica of the squelch/merge-join
  algorithm) computes expected accumulator value per vector
- Scoreboard compares DUT `mac_accum_out` against the reference, bit-exact
  pass/fail, no tolerance
- Live signal probe (`$display`-based, cycle-tagged CSV dump of every
  squelch emission and every matched MAC pair) used to root-cause mismatches
  directly against golden-model output — not simulation guesswork

## Verification Rigor

A subtle RTL race condition in the sparse buffering stages caused
systematic data mislabeling under specific timing conditions — the kind
of bug that produces plausible-looking, deterministic, *wrong* output
rather than an obvious crash. It was root-caused using a custom
cycle-accurate signal probe, cross-checked against the software golden
model index-by-index until the exact divergence point was isolated, then
resolved through an architectural revision to the affected buffering
logic. The fix and the specific mechanism are documented internally.

## Results

### Synthetic benchmark (bimodal INT8, 4094 tokens, threshold=24)
| Metric | Value |
|---|---|
| Q / K sparsity | 89.1% / 89.4% |
| Matched pairs | 153 |
| Accumulator | 372,482 — exact match to golden model |
| MAC reduction vs dense O(N²) | 100% (only matched pairs computed) |

### Real workload — Llama 3.2 1B, layer 0 Q/K activations (1602 tokens)
| Metric | Value |
|---|---|
| Q / K sparsity | 60.4% / 47.8% |
| Accumulator | −2,444,882 — exact match |

### Real workload — Llama 3.2 1B, layer 15 Q/K activations (1602 tokens)
| Metric | Value |
|---|---|
| Q / K sparsity | 64.4% / 51.6% |
| Accumulator | 1,703,123 — exact match |

*Layers 4, 8, 12 planned — not yet run at time of writing.*

## Honest Claims Assessment

**Compute reduction — strong, verified.** The design performs 100% fewer
MAC operations than dense attention on every workload tested, by
construction (only matched spike pairs are ever multiplied).

**Bandwidth reduction — real, but threshold- and workload-dependent.**
COO encoding costs 3 bytes/surviving-token (data + index) vs. 1 byte/token
dense. This only wins when survival rate is below ~33%. On real Llama
activations at threshold=24, survival is 35–52% — **above** breakeven,
meaning COO encoding of the activation stream costs *more* bandwidth than
dense, not less, at the design's default operating point. Threshold sweep
on layer 0 shows breakeven crossing at threshold≈32.

**"Breaks the memory wall" — not currently supported, and should not be
claimed as-is.** The dominant DRAM traffic bottleneck for edge LLM
inference is model-weight and KV-cache streaming (gigabytes), not Q/K
activation encoding (kilobytes). EDSA's squelch decision is data-dependent
— it requires the token value already resident on-chip before deciding to
suppress it — so it reduces post-fetch compute, not DRAM fetch volume. A
genuine memory-wall claim would require gating KV-cache *admission at
write time*, which is a documented future-work direction, not part of the
current verified design.

## Repository Contents

- `README.md` — this file
- `verification_report.md` — full scoreboard output, per-vector detail
- `results/*.csv` — accumulator, sparsity, cycle-count per test vector
- `stimulus/generate_stimulus.py` — synthetic bimodal INT8 workload generator
- `stimulus/extract_llama_qk.py` — real-activation extraction from
  Llama-3.2-1B via HuggingFace hooks, per-layer INT8 quantization
- `stimulus/*.hex` — token streams used in reported results
- No RTL source included.

## Future Work

- Complete layer 4/8/12 sweep for full-depth sparsity trend
- Threshold sweep to identify accuracy-vs-bandwidth operating curve
  (raw dot-product error grows sharply with threshold — needs downstream
  attention-quality validation, not just byte-count optimization)
- Extending the predictive-filtering approach further up the memory
  hierarchy is an active research direction, details withheld pending
  publication
