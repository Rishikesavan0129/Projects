#!/usr/bin/env python3
"""
analyze_report.py

Post-processing for the UVM verification run. Two jobs:

1. ACCURACY ANALYSIS: load report.csv (written by score_scoreboard.sv) and
   summarize pass/fail rate, per-row error distribution vs the golden
   softmax model -- this answers "how close is the hardware approximation
   to true softmax."

2. DETERMINISM CHECK: hardware/RTL determinism means: same stimulus in ->
   bit-identical outputs out, every single time, with no dependency on
   simulation seed, scheduling, or wall-clock. This script does NOT (and
   cannot) prove that by inspecting one run; you must run the simulator
   twice on the *same* stimulus.hex (run_1/report.csv and run_2/report.csv)
   and diff them here. A synchronous, combinational-loop-free, properly
   reset design (which this RTL was written to be, per the blueprint's
   STA/timing directives) should produce byte-identical CSVs across runs.
   Any difference is itself a verification finding (e.g. an uninitialized
   register, a race, or a non-deterministic X-propagation path) and should
   be treated as a bug, not noise.

Usage:
    python3 analyze_report.py --run1 sim/report_run1.csv --run2 sim/report_run2.csv
    python3 analyze_report.py --run1 sim/report.csv          # accuracy only
"""

import argparse
import csv
import sys


def load_report(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if not r or r[0].startswith("SUMMARY"):
                continue
            rows.append(r)
    return header, rows


def accuracy_summary(header, rows):
    print(f"\n=== ACCURACY ANALYSIS ({len(rows)} rows) ===")
    idx = {name: i for i, name in enumerate(header)}
    sum_errs = [abs(float(r[idx["sum_err_pct"]])) for r in rows]
    recip_errs = [abs(float(r[idx["recip_err_pct"]])) for r in rows]
    statuses = [r[idx["status"]] for r in rows]

    n_pass = statuses.count("PASS")
    n_fail = statuses.count("FAIL")

    print(f"  PASS: {n_pass}/{len(rows)}  FAIL: {n_fail}/{len(rows)}")
    if sum_errs:
        print(f"  sum_err_pct    : mean={sum(sum_errs)/len(sum_errs):.3f}  "
              f"max={max(sum_errs):.3f}")
        print(f"  recip_err_pct  : mean={sum(recip_errs)/len(recip_errs):.3f}  "
              f"max={max(recip_errs):.3f}")

    if n_fail > 0:
        print("\n  Failing rows:")
        for r in rows:
            if r[idx["status"]] == "FAIL":
                print(f"    row {r[idx['row_id']]}: sum_err={r[idx['sum_err_pct']]}% "
                      f"recip_err={r[idx['recip_err_pct']]}%")

    return n_fail == 0


def determinism_check(path1, path2):
    print(f"\n=== DETERMINISM CHECK ===\n  run1: {path1}\n  run2: {path2}")
    h1, r1 = load_report(path1)
    h2, r2 = load_report(path2)

    if h1 != h2:
        print("  FAIL: report headers differ between runs (schema mismatch)")
        return False

    if len(r1) != len(r2):
        print(f"  FAIL: row count differs ({len(r1)} vs {len(r2)})")
        return False

    mismatches = []
    for i, (a, b) in enumerate(zip(r1, r2)):
        if a != b:
            mismatches.append((i, a, b))

    if mismatches:
        print(f"  FAIL: {len(mismatches)}/{len(r1)} rows differ between runs "
              f"-- DESIGN IS NOT DETERMINISTIC")
        for i, a, b in mismatches[:10]:
            print(f"    row idx {i}:\n      run1={a}\n      run2={b}")
        return False

    print(f"  PASS: all {len(r1)} rows bit-identical across runs -- design is deterministic")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True, help="path to report.csv from run 1")
    ap.add_argument("--run2", help="path to report.csv from run 2 (for determinism check)")
    args = ap.parse_args()

    header, rows = load_report(args.run1)
    acc_ok = accuracy_summary(header, rows)

    det_ok = True
    if args.run2:
        det_ok = determinism_check(args.run1, args.run2)
    else:
        print("\n=== DETERMINISM CHECK ===\n  SKIPPED: pass --run2 <path> "
              "(report.csv from a second identical simulation) to check determinism.")

    print("\n=== FINAL VERDICT ===")
    print(f"  Accuracy:     {'PASS' if acc_ok else 'FAIL'}")
    print(f"  Determinism:  {'PASS' if det_ok else ('FAIL' if args.run2 else 'NOT CHECKED')}")

    sys.exit(0 if (acc_ok and det_ok) else 1)


if __name__ == "__main__":
    main()
