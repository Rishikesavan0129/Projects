# FAE — Verification Report

**Design:** Hardware FlashAttention Engine — online (tiled) softmax pipeline
**Environment:** UVM 1.2, SystemVerilog
**Report scope:** Functional accuracy verification, coverage closure,
physical implementation signoff

---

## 1. Verification Objectives

1. Confirm the hardware's fixed-point online-softmax output (running sum,
   running max, reciprocal) matches a double-precision software golden
   model within acceptable error bounds
2. Achieve closure on score-value and FSM-state functional coverage
3. Establish determinism (repeatable, bit-identical output across
   identical stimulus, independent of simulation scheduling)
4. Confirm physical implementability via full RTL-to-GDSII signoff

## 2. Test Environment

- UVM agent driving quantized INT8 attention-logit scores, file-driven
  from `stimulus.hex` (8 rows × 4096 elements/row)
- Monitor capturing drain events (`m`, `sum`, `recip`) per completed row
- Scoreboard comparing hardware drain output against a golden model
  computed in double-precision Python on the *same quantized* input the
  hardware receives, reporting `sum_err_pct` and `recip_err_pct` per row
- Functional coverage collector tracking score-value distribution and
  FSM-state transitions during the run

## 3. Debugging Summary

A data-integrity defect affecting back-to-back row throughput was
identified in the reciprocal computation stage during initial bring-up:
consecutive rows streamed without a gap could corrupt the reciprocal
pipeline's in-flight state. Root-caused during simulation and resolved
via a pipeline-stage correction in the reciprocal datapath. Results in
this report reflect the corrected design.

## 4. Results

### 4.1 Functional Accuracy — 8-Row Directed Stimulus

| Row | Golden max | HW sum | Golden sum | Sum err % | Recip err % | Result |
|---|---|---|---|---|---|---|
| 0 | 2.0 | 619,708,486 | 619,708,486.0 | 0.00 | 0.00 | PASS |
| 1 | 2.0 | 432,562,021 | 432,659,941.0 | −0.02 | 0.00 | PASS |
| 2 | 2.0 | 638,914,966 | 638,914,966.0 | 0.00 | 0.00 | PASS |
| 3 | 1.0 | 615,672,210 | 615,672,210.0 | 0.00 | 0.00 | PASS |
| 4 | 1.0 | 611,094,856 | 611,094,856.0 | 0.00 | 0.00 | PASS |
| 5 | 2.0 | 628,324,349 | 628,324,349.0 | 0.00 | 0.00 | PASS |
| 6 | 1.0 | 613,066,568 | 613,131,848.0 | −0.01 | 0.00 | PASS |
| 7 | 2.0 | 501,781,492 | 501,814,132.0 | −0.01 | 0.00 | PASS |

**8 / 8 rows passed (100%).** Max observed `sum_err_pct` magnitude:
0.02%. `recip_err_pct`: 0.00% on every row (Newton-Raphson reciprocal
converged to golden precision within the fixed-point output width on
all tested rows).

### 4.2 Functional Coverage

| Coverage type | Result |
|---|---|
| Score-value coverage | **100.00%** |
| FSM-state coverage | **100.00%** |
| **Overall** | **100.00%** |

Full closure achieved on the coverage model defined for this design —
in contrast to a partial-coverage state, this indicates the 8-row
stimulus set exercised every tracked score-value bin and every FSM state
transition at least once.

### 4.3 Determinism

**Confirmed.** Two independent simulation runs on identical stimulus,
each writing to a separate output file (`report.csv`, `report_2.csv`),
were compared directly. All 8 rows are bit-identical across both runs —
`hw_m`, `hw_sum_raw`, and `hw_recip_raw` match exactly, row for row, with
zero delta in either accuracy metric between runs. The design produces
repeatable, deterministic output independent of simulation invocation.

### 4.4 Physical Implementation — SkyWater 130nm, OpenLane

| Metric | Value |
|---|---|
| DRC violations (Magic) | 0 |
| LVS | Clean — 14,990 nets matched |
| Antenna violations | 1 pin + 1 net (minor, typically resolved by adding an antenna diode — see §6) |
| Worst negative slack (WNS) | 0.0 ns @ 50 MHz |
| Total negative slack (TNS) | 0.0 ns |
| Standard cell count | 14,820 |
| Core utilization | 41.05% |
| Die area | 0.397 mm² |

### 4.5 Power (SAIF-annotated, 50 MHz)

| Group | Power | Share |
|---|---|---|
| Combinational | 41.8 mW | 95.8% |
| Sequential | 1.84 mW | 4.2% |
| **Total** | **43.6 mW** | 100% |

## 5. Sign-off Statement

The design's online-softmax computation is verified **functionally
correct** against an independent double-precision golden model across all
8 tested rows, with maximum observed deviation of 0.02% on the running
sum and exact reciprocal convergence on every row. Score-value and
FSM-state functional coverage are both closed at 100% for the defined
coverage model. **Determinism is confirmed** via two independent
simulation runs producing bit-identical results across all rows. Physical
implementation is DRC-clean and LVS-clean with zero timing slack at the
50 MHz target; two minor antenna violations remain, typically resolved
with a standard diode-insertion pass. This design is verified per the
original test plan.

## 6. Known Limitations / Open Items

- **2 minor antenna violations** (1 pin, 1 net) — typically resolved via
  standard antenna-diode insertion; "0 DRC violations" claims should be
  scoped to Magic DRC specifically, distinct from the antenna check
- **8-row stimulus only** — no sweep across score dynamic range, row
  length, or degenerate distributions (all-equal scores, single dominant
  score) performed to date
- **Golden model validates fixed-point correctness, not quantization
  impact** — the reference is computed on the same quantized int8 input
  the hardware receives, confirming the hardware faithfully implements
  online softmax in fixed point. It does not by itself establish that
  int8 quantization preserves attention quality relative to a
  full-precision floating-point reference; that is a separate,
  unaddressed question
