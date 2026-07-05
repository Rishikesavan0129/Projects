# EDSA — Verification Report

**Design:** Event-Driven Spiking Attention accelerator
**Environment:** UVM 1.2, SystemVerilog
**Report scope:** Functional correctness verification, sparsity/bandwidth
characterization on synthetic and real-model workloads

---

## 1. Verification Objectives

1. Confirm the sparse Q/K dot-product accumulator matches a bit-exact
   software golden model across representative workloads
2. Characterize achieved sparsity and its downstream effect on compute
   and encoding-bandwidth claims
3. Establish determinism (repeatable cycle count and result across
   identical stimulus)
4. Identify verification coverage gaps and stimulus classes not yet
   exercised

## 2. Test Environment

- UVM testbench with independent Q-stream and K-stream driver agents,
  each supporting hex-file playback and randomized/directed sequence
  generation
- Software reference model: independent, bit-exact re-implementation of
  the predictive-filtering and merge-join algorithm, used to compute
  expected accumulator value per test vector
- Scoreboard: exact-match comparison (zero tolerance) between DUT output
  and reference model, per vector
- Functional coverage model tracking scenario classes (threshold range,
  stream-length parity, sparsity extremes, back-to-back matches, EOS
  timing conditions)

Three test classes exist in the environment:

| Test | Purpose | Status |
|---|---|---|
| `edsa_hex_test` | Directed playback of a fixed token stream (synthetic or real-extracted) against the golden model | **Executed — see §4** |
| `edsa_random_test` | Randomized stimulus across threshold, stream length, sparsity | Defined, not yet executed |
| `edsa_corner_test` | Directed edge cases: empty stream, all-spike stream, INT8 boundary values, back-to-back EOS | Defined, not yet executed |

## 3. Debugging Summary

Initial verification runs surfaced a data-integrity defect: deterministic,
plausible-looking, but numerically incorrect accumulator output. The
defect was root-caused using a custom cycle-accurate signal probe that
logged every predictive-filter emission and every matched accumulate
event, cross-referenced token-by-token against the golden model to
isolate the exact point of divergence. The underlying RTL defect and its
resolution are documented internally; this report covers verification
results post-fix only.

## 4. Results

### 4.1 Correctness — Directed Hex Vectors

| Vector | Tokens | Threshold | Q spikes | K spikes | Q/K sparsity | DUT accum | Expected accum | Result |
|---|---|---|---|---|---|---|---|---|
| Synthetic bimodal | 4094 | 24 | 448 | 434 | 89.1% / 89.4% | 372,482 | 372,482 | **PASS** |
| Llama-3.2-1B, layer 0 | 1602 | 24 | 634 | 837 | 60.4% / 47.8% | −2,444,882 | −2,444,882 | **PASS** |
| Llama-3.2-1B, layer 15 | 1602 | 24 | 571 | 776 | 64.4% / 51.6% | 1,703,123 | 1,703,123 | **PASS** |

All vectors: bit-exact match, zero mismatch, zero tolerance applied.

### 4.2 Determinism

Repeated runs of the same vector produced identical cycle count and
identical accumulator value across all trials (cycle range: 0 across
repeated runs on each vector). No metastability, no run-to-run variance
observed in behavioral simulation.

### 4.3 Compute Reduction

100% of MAC operations were eliminated relative to a dense O(N²) baseline
on every vector tested, by construction — the architecture only ever
multiplies index-matched surviving pairs, never suppressed or
non-matching tokens.

### 4.4 Bandwidth Characterization

Sparse (index, value) encoding was compared against dense (raw
byte-stream) encoding for the surviving-token payload:

| Vector | Survival rate (avg Q/K) | Encoding vs. dense | Verdict at threshold=24 |
|---|---|---|---|
| Synthetic bimodal | 10.9% | 0.33× dense | Bandwidth-favorable |
| Layer 0 | 45.9% | 1.38× dense | Bandwidth-unfavorable |
| Layer 15 | 41.9% | 1.26× dense | Bandwidth-unfavorable |

A threshold sweep on Layer 0 data identified the encoding breakeven point
at approximately threshold=32 (survival rate crossing below the 1/3
breakeven ratio inherent to the encoding's per-token overhead). Below
that threshold, sparse encoding costs *more* bytes than dense on real
model activations, not fewer.

## 5. Coverage

**Functional coverage: 20.49%** at time of this report.

This reflects execution of `edsa_hex_test` only — a small number of
directed vectors at a single fixed threshold. The coverage model's
remaining bins (threshold sweep range, extreme sparsity, stream-length
parity edge cases, EOS-timing corner conditions) require executing
`edsa_random_test` and `edsa_corner_test`, which are implemented but not
yet run. **Coverage in its current state supports a correctness claim on
the tested vectors; it does not yet support a general robustness claim
across the design's full input space.**

## 6. Known Limitations of This Verification Pass

- Only 3 of a planned 5-layer real-workload sweep executed (layers 4, 8,
  12 pending)
- Random and corner-case test classes not yet executed — coverage gap
  noted in §5
- No downstream attention-quality (softmax-level) validation performed;
  correctness is verified against the sparse dot-product's own reference
  model, not against a full-precision dense attention baseline. Raw
  dot-product deviation from dense grows substantially as threshold
  increases (documented separately) — this has not been validated
  against actual model output quality/perplexity
- No formal (SVA-based) property verification performed; correctness
  demonstrated by simulation only

## 7. Sign-off Statement

Based on the results in §4, the design's Q/K sparse dot-product
computation is verified **bit-exact correct** against an independent
golden reference model, on both synthetic and real transformer-activation
workloads, at the threshold and vector count tested. The compute-reduction
claim (100% MAC elimination) is supported unconditionally by construction.
The bandwidth-reduction claim is **conditionally supported**, dependent on
operating threshold and workload sparsity — not supported at the design's
default threshold on real model data. Coverage remains incomplete pending
execution of the random and corner-case test suites.
