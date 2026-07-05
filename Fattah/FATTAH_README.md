# F.A.T.T.A.H. — Near-Memory Processor

**A JIT data-assembly interface for AI NPUs, generating on-the-fly COO
sparse-format packets from dense activation streams — verified at scale
against a realistic 25,000-block AI workload stress test.**

---

## Problem

NPU-adjacent memory interfaces typically move dense activation tensors
even when a large fraction of the values are zero (a common property of
ReLU-activated neural network layers). Reconstructing or generating a
sparse (index, value) representation on-the-fly, close to memory, can
reduce the data volume the rest of the system has to move — but only if
the encoding/decoding logic is bit-exact across every edge case: dense
blocks, fully-zero blocks, maximum-length zero runs, and boundary data
values.

## Architecture

- On-the-fly pooling and COO (coordinate-format) sparse packet generation
  from dense NPU activation blocks
- Skip-run encoding: each non-zero element is packed with a count of
  preceding zeros, an end-of-block flag, and the value itself
- Combinatorial assembly resolves `TLAST`-adjacent collision timing
  without a stall cycle

## Verification Methodology

- UVM environment with a dual-mode scoreboard:
  - **AI Profiler mode** — full golden-model comparison. Every block's
    hardware-reconstructed dense output is checked byte-for-byte against
    an expected dense reference; results logged per-block for downstream
    analysis. This mode is the tape-out correctness gate.
  - **Random Torture mode** — constrained-random structural/protocol
    stress testing (results not included in this report)
- Functional coverage model targeting the encoding scheme's actual edge
  cases: zero-skip (dense), standard sparsity, maximum skip-run boundary,
  zero-payload and maximum-payload data values, and NPU backpressure
  (stalled vs. running)

## Results

### Correctness — 25,000-Block AI Workload Stress Test
| Metric | Result |
|---|---|
| Total blocks (400,000 elements) | 25,000 |
| Blocks matched (hardware vs. golden dense) | **25,000 / 25,000 (100%)** |
| Mismatches | 0 |

This is a substantially larger stimulus set than used in this portfolio's
other functional verification campaigns — see `verification_report.md`
for the full methodology, including a documented instance of catching
and correcting a stale-results-file discrepancy during the analysis
process.

### Coverage — Closed
| Coverpoint | Result |
|---|---|
| Skip-run (zero / mid / max boundary) | 100.00% |
| Data payload (zero / max / standard) | 100.00% |
| NPU backpressure (stalled / running) | 100.00% |
| Cross coverage (all combinations) | 100.00% |
| **Overall** | **100.00%** |

All defined bins closed, including the maximum skip-run edge case under
both NPU stall states. See `verification_report.md` §4 for full detail.

## Honest Claims Assessment

**Functional correctness — strong, verified.** Bit-exact match across
25,000 independently generated blocks is a solid, large-scale
correctness result.

**"Tape-out ready" — not currently supported by the evidence in this
repository.** That claim requires physical implementation signoff (DRC,
LVS, timing closure, power analysis) of the kind documented for this
portfolio's other projects. No such data has been included here. Until
it is, this project should be described as **functionally verified**,
not tape-out ready.

## Repository Contents

- `README.md` — this file
- `verification_report.md` — full methodology, results, and open items
- `stimulus/gen_stimulus.py` — 25,000-block AI-activation stimulus
  generator (ReLU + quantization, skip-run COO encoding)
- `stimulus/stimulus.hex`, `stimulus/expected_dense.hex` — as-run
  stimulus and golden dense reference
- `results/fattah_ml_telemetry.csv` — per-block hardware vs. golden
  comparison results (25,000 rows)
- `analysis/analyze_telemetry.py` — pass-rate summary and
  sparsity/bandwidth characterization notebook
- No RTL source, no physical implementation files, included.

## Future Work

- If pursuing a tape-out-ready claim: complete and document physical
  implementation signoff (DRC/LVS/timing/power), matching the standard
  used elsewhere in this portfolio
- Fix the stimulus generator's non-saturating quantization step (values
  above the representable range currently wrap rather than clip) for
  data-generation correctness in future stress-test runs
