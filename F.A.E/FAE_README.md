# FAE — Hardware FlashAttention Engine

**A synthesizable, physically-implemented hardware pipeline for the online
(tiled, single-pass) softmax stage of transformer attention — the
numerically stable, streaming normalization step FlashAttention-style
algorithms use to avoid ever materializing a full N×N attention matrix.**

---

## Problem

Standard softmax requires two passes over a score row: one to find the max
(for numerical stability), one to compute and sum the exponentials. At
attention-scale sequence lengths, this means either buffering an entire
row of scores before normalizing, or accepting numerical instability from
skipping the max-subtraction step. FlashAttention-style algorithms solve
this by processing scores in tiles and updating a running max and running
sum *incrementally* — never holding the full row in memory at once. FAE is
a hardware realization of that online update rule, plus the reciprocal
division it requires, built and verified as real synthesizable RTL rather
than a software approximation.

## Architecture

Three pipelined stages, tile-based streaming interface:

| Stage | Role |
|---|---|
| **Online Accumulator Pipe** | Consumes one score at a time. Maintains a running max and running sum across the row using the standard online-softmax rescaling identity — when a new score exceeds the current running max, previously-accumulated sum terms are rescaled rather than discarded and recomputed. |
| **Base-2 Softmax Stage** | Computes a fixed-point approximation of `2^F` (base-2, not natural exponent — scores are pre-scaled by `log2(e)` upstream so this is mathematically equivalent to a standard softmax) for the fractional part of each rescaled score. Fixed-point throughout; the multiply is forced onto a physical DSP slice rather than LUT fabric. |
| **Reciprocal Pipe (Newton-Raphson)** | Computes `1/sum` for the final normalization via iterative Newton-Raphson refinement rather than a hardware divider, avoiding the area/latency cost of true division. |
| **Top-level control (`flash_attention_top`)** | AXI tile-fetch interface for streaming score tiles in, drain interface exposing the final running sum, running max, and reciprocal once a full row has been processed. |

Interface shape: one score (`score_x`, INT8) streamed in per cycle;
one row = 4096 elements across 16 tiles of 256; drain outputs one
`(sum, max, reciprocal)` triple per completed row.

## Verification Methodology

- UVM-driven score streaming against a double-precision Python golden
  softmax model computed on the *same quantized* int8 scores the hardware
  sees (so golden and hardware are compared on identical inputs, not just
  identical math)
- Per-row error metrics: `sum_err_pct`, `recip_err_pct` — relative
  deviation of the hardware's fixed-point running sum and reciprocal from
  the golden double-precision values
- Determinism verified by running identical stimulus twice and diffing
  the full result CSV byte-for-byte — any row-level divergence between
  runs is treated as a bug (uninitialized register, race condition, or
  non-deterministic X-propagation), not simulation noise

## Results

### Functional Accuracy (8-row synthetic attention-logit stimulus)
| Metric | Value |
|---|---|
| Rows passed | 8 / 8 (100%) |
| Mean `sum_err_pct` | ~0.005% |
| Max `sum_err_pct` | 0.023% |
| `recip_err_pct` (all rows) | 0.000% |

Stimulus: realistic Q·K attention logits (unit-norm random Q/K vectors,
standard `1/√d` scaling, base-2 domain conversion), not hand-picked
corner cases.

### Physical Implementation — SkyWater 130nm, OpenLane
| Metric | Value |
|---|---|
| DRC violations (Magic) | **0** |
| LVS | **Clean** (14,990 nets matched) |
| Antenna violations | 1 pin, 1 net *(not yet resolved — see Limitations)* |
| Worst negative slack (WNS) | 0.0 ns at 50 MHz (timing closed) |
| Standard cell count | 14,820 |
| Core utilization | 41.05% |
| Die area | 0.397 mm² |

### Power (SAIF-annotated, 50 MHz, OpenSTA)
| Group | Total Power | Share |
|---|---|---|
| Combinational | 41.8 mW | 95.8% |
| Sequential | 1.84 mW | 4.2% |
| **Total** | **43.6 mW** | 100% |

## Known Limitations

- **1 pin + 1 net antenna violation present** — not zero. If claiming
  "0 DRC violations" elsewhere, this should be scoped precisely (Magic
  DRC rule violations: 0; antenna rule violations: 2 total) rather than
  stated as a blanket zero.
- Accuracy verified against a golden softmax computed on the *quantized*
  int8 input — this validates the hardware correctly implements
  fixed-point online softmax, not that int8 quantization itself preserves
  full attention quality relative to a floating-point reference. That is
  a separate, unaddressed question.
- 8-row synthetic stimulus only; no sweep across score dynamic range,
  row length, or degenerate cases (all-equal scores, single dominant
  score) has been run to date.

## Repository Contents

- `README.md` — this file
- `verification_report.md` — accuracy and determinism methodology, full
  result table
- `stimulus/gen_stimulus.py` — synthetic attention-logit generator
  (realistic Q/K statistics, int8 quantization matching RTL parameters)
- `stimulus/analyze_report.py` — accuracy summary + cross-run determinism
  diff tool
- `stimulus/stimulus.hex`, `results/report.csv` — as-run stimulus and
  results
- `pd/metrics.csv`, `pd/manufacturability.rpt`, `pd/power_report.txt` —
  physical implementation signoff data
- No RTL source, no GDSII included.

## Future Work

- Resolve outstanding antenna violations for a genuine zero-violation
  signoff
- Broaden stimulus coverage: dynamic range sweep, degenerate score
  distributions, minimum/maximum row length
- Downstream integration with an upstream sparse score-generation stage
  is an open architecture question — see integration note below
