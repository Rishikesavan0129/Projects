# Projects
**Rishikesavan S.**

Silicon Design & Architecture

Silicon engineer focused on low-power computer architecture, Domain-Specific Architectures (DSA), and ASIC physical design. Primary objective: neutralizing the memory wall and dynamic power bottlenecks in AI workloads via neuromorphic spatial and temporal sparsity.

**Technical Profile**

**Languages:** SystemVerilog, Verilog, Python, C, Tcl

**Architecture:** RISC-V (PicoRV32), PCPI, AXI4/AXI4-Lite, INT8 NPUs, NoC, Datapath Pipelining

**Verification:** UVM, SystemVerilog Assertions (SVA), Cycle-Accurate Modeling

**Physical Design:** RTL-to-GDSII, OpenLane, Yosys, OpenROAD, Magic, OpenSTA (Sky130, ASAP7)

**Core Architectures**

**H.A.F.I.Z. Edge Core(Hardware Architecture for In-Phase and Quatra-Phase Zeroes)**

Neuromorphic-Inspired Tri-Core RISC-V SoC

Integrated PicoRV32 with custom spatial sparsity filters and an INT8 NPU.


**Event-Driven Spiking Attention (EDSA)**

Dynamic Activation Sparsity IP for LLMs.

Hardware-level token entropy evaluation and dynamic KV-cache pruning.


**Hardware FlashAttention Engine (FAE)**

Single-Pass LLM Attention Accelerator.

**Fast Adaptive Tensor Translation & Assembly Hardware(F.A.T.T.A.H)**
A Near memory processing IP Block.

**Multicast Universal Junction for Event Exchange Bus(M.U.J.E.E.B)**
NoC router (Crossbar + Direct Memory Access(DMA) ) IP Block.

**Methodology**

Mathematical Modeling $\rightarrow$ RTL Prototyping $\rightarrow$ UVM Verification $\rightarrow$ RTL-to-GDSII

Every Project was Verified with strict adherence to timing closure, functional coverage, and physical(Magic) DRC rules.
