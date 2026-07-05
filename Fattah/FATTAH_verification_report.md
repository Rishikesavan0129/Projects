# F.A.T.T.A.H. — Verification Report

**Design:** Near-Memory Processor — JIT data assembly / COO sparse-format
generation interface for AI NPUs
**Environment:** UVM, SystemVerilog
**Report scope:** Large-scale stress-test correctness verification against
a real AI-workload stimulus stream

---

## 1. Verification Objectives

1. Confirm the hardware's sparse-encoded output reconstructs to the exact
   dense reference across a large-scale, realistic AI-activation workload
2. Characterize coverage of protocol-relevant edge cases in the sparse
   encoding scheme (skip-run boundaries, data value extremes, downstream
   backpressure)
3. Establish a pass/fail gate suitable for tape-out decision-making

## 2. Test Environment

- UVM environment (`fattah_env`) composing a driving agent, a dual-mode
  scoreboard, and a functional coverage collector
- **Scoreboard modes:**
  - *AI Profiler mode* — full golden-model comparison. Each block's
    hardware-reconstructed dense output is compared byte-for-byte against
    an expected dense reference, with per-block results logged to CSV
    (`Block_ID, Expected_Hex, Actual_Hex, Match_Status`) for downstream
    analysis. This mode gates tape-out: the scoreboard explicitly reports
    "VERIFICATION FAILED: Do NOT Tape-Out" if any mismatch is recorded.
  - *Random Torture mode* — constrained-random structural/protocol stress
    stimulus without a golden reference (no result data available for this
    report)
- **Coverage model** (`fattah_coverage`) — one covergroup with three
  coverpoints targeting the encoding scheme's actual edge cases:
  - Skip-run length: dense (zero-skip), standard mid-range sparsity, and
    the maximum representable skip-run boundary
  - Data value: zero payload, maximum (0xFF) payload, and standard
    mid-range values
  - Downstream NPU backpressure: stalled vs. running

## 3. Results — AI Profiler Stress Test

### 3.1 Stimulus

25,000 independently generated 16-element INT8 blocks (400,000 total
elements), derived from ReLU-activated Gaussian-distributed synthetic
data intended to approximate real NPU activation sparsity patterns.

### 3.2 Correctness

| Metric | Result |
|---|---|
| Total blocks | 25,000 |
| Blocks matched (hardware vs. golden dense) | **25,000 / 25,000 (100.00%)** |
| Mismatches | 0 |
| Block ID continuity | Verified — 1 through 25,000, no gaps |

**Note on verification process:** an initial analysis pass reported a
lower total (14,129 blocks, one mismatch) due to a stale results file
present in the analysis environment from an earlier, incomplete run. The
discrepancy was identified, the correct current-run file was confirmed
directly (25,000 rows, all `Match_Status = 1`, continuous block IDs), and
the result above reflects that direct verification — not the initial
(incorrect) analysis output.

## 4. Coverage — Closed

Following identification that coverage was being sampled correctly but
never read out and reported (a one-line gap in the report phase, not a
sampling defect), the environment was corrected and re-run against the
same 25,000-block stimulus.

| Coverpoint | Coverage |
|---|---|
| `cp_skip` (skip-run: zero / mid-range / max boundary) | **100.00%** |
| `cp_data` (payload: zero / max / standard range) | **100.00%** |
| `cp_npu_stall` (backpressure: stalled / running) | **100.00%** |
| `cr_skip_x_stall` (cross coverage, all 6 combinations) | **100.00%** |
| **Overall** | **100.00%** |

All defined bins closed, including the two hardest-to-hit combinations:
maximum skip-run length occurring during both NPU-stalled and
NPU-running conditions. Coverage is fully closed against the defined
test plan.

## 5. Known Limitations
- **Random Torture mode results unavailable** — structural/protocol
  stress testing exists in the environment but no result data was
  available for this report
- **Stimulus generation note:** the synthetic data generator's
  quantization step does not saturate values to the valid INT8 range
  before casting, which can silently wrap large activation values rather
  than clipping them. This does not affect the correctness result above
  (both the hardware stimulus and the golden dense reference are derived
  from the same generated values, so the comparison remains internally
  consistent), but it means the stimulus does not fully represent a
  properly saturated real-world quantization pipeline
- **No determinism check performed** — this report reflects a single
  simulation run; repeatability across independent runs has not been
  verified for this design, unlike the two-run comparisons performed for
  other projects in this portfolio

## 6. Sign-off Statement

The design's sparse-to-dense reconstruction is verified **bit-exact
correct** across a 25,000-block, 400,000-element realistic AI-workload
stress test, with zero mismatches — a substantially larger stimulus set
than used for other projects in this portfolio. **Functional coverage is
fully closed at 100%** across all defined coverpoints and cross
coverage, including the maximum skip-run boundary under both NPU
backpressure states. This is a complete, closed functional verification
result.

Note: the verification environment's own scoreboard reports "Silicon is
Tape-Out Ready" on zero mismatches — this is a functional-correctness
conclusion, not a physical-implementation signoff. A genuine tape-out-
readiness claim additionally requires DRC/LVS/timing/power signoff data,
which is not part of this report — see the companion README's Honest
Claims Assessment.
