# H.A.F.I.Z. Edge Core

**A Neuromorphic-Inspired RISC-V SoC Integrating Spatial Sparsity Filtering,
an INT8 Neural Processing Unit, and Hardware-Level Sensory Habituation for
Ultra-Low-Power Edge IoT**

📄 **Preprint:** S. Rishikesavan, "H.A.F.I.Z. Edge Core: A
Neuromorphic-Inspired RISC-V SoC Integrating Spatial Sparsity Filtering,
an INT8 Neural Processing Unit, and Hardware-Level Sensory Habituation for
Ultra-Low-Power Edge IoT." Zenodo, Jun. 28, 2026.
**DOI:** [10.5281/zenodo.20995571](https://doi.org/10.5281/zenodo.20995571)

Full methodology, equations, FSM diagrams, and complete reference list are
in the preprint linked above. This README summarizes the architecture and
headline verified results.

---

## Problem

Battery-powered IoT sensor nodes burn power at every stage of the
acquisition pipeline regardless of whether the incoming data carries
useful information. In high-noise RF environments, the majority of
digitized samples fall below any useful signal threshold — meaning most
ADC-to-SRAM-to-CPU-to-NPU cycles are spent processing noise, not signal.
H.A.F.I.Z. targets this directly, inspired by how the human peripheral
nervous system filters sensory input before it ever reaches the brain.

## Architecture — Three-Stage Cascade

| Stage | Block | Function |
|---|---|---|
| 1 | **ED-NMP-512** | Spatial sparsity filter. Computes `E = I² + Q²` energy in a single combinational cycle, suppressing sub-threshold ADC samples before they reach the memory bus — no SRAM write, no CPU interrupt, no NPU wake. |
| 2 | **INT8 NPU** | Pipelined MAC engine, integrated via the PicoRV32 PCPI coprocessor bus. Supports dot-product, fully-connected, and 1D-CNN inference workloads with firmware-controlled accumulator resets. |
| 3 | **Al-Mani SHU** | Temporal habituation filter. Suppresses samples that are energetically valid but informationally redundant (unchanged from the last significant reading), waking the NPU only on genuine environmental change. |

The cascade mirrors the biological pipeline from peripheral receptor
transduction (spatial threshold) through nervous-system habituation
(temporal threshold) to central processing (inference) — the NPU wakes
only when both upstream filters agree the input is novel.

## Verified Results

### Spatial Filtering (ED-NMP-512) — 5,000-sample RF stimulus, 80% noise ratio
| Metric | Result |
|---|---|
| Samples correctly squelched | 4,108 / 5,000 (82.2%) |
| Samples passed to SRAM | 892 / 5,000 (17.8%) |
| ML cross-validation (3 independent classifiers) | AUC = 1.000, zero logic errors |
| Vivado cycle-accurate simulation | 100% PASS across all 5,000 samples |

### NPU Inference — 3,386-sample real ECG waveform (1D-CNN, arrhythmia detection)
| Metric | Result |
|---|---|
| Hardware vs. software accumulator match | 3,386 / 3,386 (100%) |
| Failures | 0 |
| Match holds across `clear_accum` boundaries | Yes |

### Al-Mani SHU — UVM Unit-Level Verification
| Metric | Result |
|---|---|
| Total transactions (7 stimulus classes) | 3,518 |
| Scoreboard mismatches | 0 |
| SVA assertion violations | 0 / 5 |
| Functional coverage | 7 / 8 cover groups at 100%* |

*The 8th group (`cg_delta_magnitude`) closes at 83.3%; the unreached bin
corresponds to a delta value structurally unreachable given the design's
16-bit input width — confirming correct overflow-margin sizing rather
than a verification gap.

### Full SoC — System-Level Integration
| Metric | Result |
|---|---|
| Total transactions | 9,360 |
| Mismatches | 0 |
| Determinism | 100% (confirmed) |

### Power (SAIF-annotated, SkyWater 130nm, post-route)
| Metric | Value |
|---|---|
| Baseline peak active power | 105.29 mW |
| Time-averaged power (with filtering) | 26.44 mW |
| **Reduction** | **74.8%** |
| **Battery life extension** | **3.98×** |

### Physical Implementation — Final Tri-Core Signoff (SkyWater 130nm)
| Metric | Value |
|---|---|
| Clock frequency | 50 MHz |
| Standard cells | 38,087 |
| Die area | 1.0 mm² |
| Total power (SAIF-annotated) | 52.3 mW |
| Magic DRC violations | **0** |
| Antenna violations | 9 (diode-repairable) |
| LVS | Clean |

### Cross-Technology Scaling — ASAP7 7nm (synthesis + static power estimate)
| Metric | Value |
|---|---|
| Clock frequency | 1 GHz |
| Estimated power | 55.0 mW (α = 0.2, synthesis-level, no full place-and-route) |
| Frequency gain vs. 130nm | 20× at comparable power envelope |

*The 7nm figure is a synthesis + OpenSTA static estimate, not a
SAIF-annotated post-route measurement — full PnR was not run at this
node due to router memory constraints. It demonstrates architectural
scalability, not a physical signoff.*

## Repository Contents

- `README.md` — this file
- Full technical detail, equations, FSM specifications, SVA property
  definitions, and complete verification methodology: see the
  [preprint](https://doi.org/10.5281/zenodo.20995571)

No RTL source, GDSII, or additional implementation files are included in
this repository — the preprint is the canonical technical reference.
