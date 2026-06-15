#!/usr/bin/env python3
"""
S* (Plagnol & Wall 2006) ROC curve for ghost archaic introgression.

S* detects introgression by finding unusual LD patterns among variants
private to the target population (absent from a reference panel).

Since all 50 AFR diploid samples have ~3% ghost introgression and S*
needs a non-introgressed reference panel, we split AFR 50/50:
  - Reference: tsk_25..tsk_49  (used as outgroup)
  - Target:    tsk_0..tsk_24   (tested for introgression, 50 haplotypes)

Uses the sstar package (Huang 2022, MBE).

Stages:
  prepare    Create ref/tgt sample lists
  run_sstar  Execute S* computation
  roc        Parse scores, compute and plot ROC

Bruce Rannala, Feb 2026
"""

import os
import sys
import json
import argparse
import numpy as np

from introgression_roc_pipeline import (
    Params,
    tracts_to_window_labels,
    aggregate_for_roc,
    compute_roc,
    plot_roc_curves,
)


# ================================================================
#  Stage 1: prepare — Create ref/tgt sample lists
# ================================================================

def stage_prepare(out_dir):
    """Write ref_samples.txt and tgt_samples.txt."""
    os.makedirs(out_dir, exist_ok=True)

    ref_path = os.path.join(out_dir, "ref_samples.txt")
    tgt_path = os.path.join(out_dir, "tgt_samples.txt")

    with open(ref_path, "w") as f:
        for i in range(25, 50):
            f.write(f"tsk_{i}\n")
    print(f"  Reference samples (25): tsk_25..tsk_49 → {ref_path}")

    with open(tgt_path, "w") as f:
        for i in range(25):
            f.write(f"tsk_{i}\n")
    print(f"  Target samples (25):    tsk_0..tsk_24  → {tgt_path}")


# ================================================================
#  Stage 2: run_sstar — Execute S* computation
# ================================================================

def stage_run_sstar(sim_dir, out_dir):
    """Run sstar cal_s_star on the simulation VCF."""
    from sstar.cal_s_star import cal_s_star

    vcf = os.path.join(sim_dir, "singer_input.vcf")
    ref_ind = os.path.join(out_dir, "ref_samples.txt")
    tgt_ind = os.path.join(out_dir, "tgt_samples.txt")
    output = os.path.join(out_dir, "sstar_scores.txt")

    for f in [vcf, ref_ind, tgt_ind]:
        if not os.path.isfile(f):
            print(f"  ERROR: Missing {f}. Run --stage prepare first.")
            return False

    print(f"  VCF:    {vcf}")
    print(f"  Ref:    {ref_ind}")
    print(f"  Target: {tgt_ind}")
    print(f"  Output: {output}")
    print("  Running S* (50 kb windows, non-overlapping)...")

    cal_s_star(
        vcf=vcf,
        ref_ind_file=ref_ind,
        tgt_ind_file=tgt_ind,
        anc_allele_file=None,
        output=output,
        win_len=50000,
        win_step=50000,
        thread=4,
        match_bonus=5000,
        max_mismatch=1,
        mismatch_penalty=-10000,
    )

    # Quick summary
    n_lines = 0
    n_na = 0
    with open(output) as f:
        header = f.readline()
        for line in f:
            n_lines += 1
            parts = line.strip().split("\t")
            if len(parts) >= 5 and parts[4] == "NA":
                n_na += 1
    print(f"  {n_lines} score rows ({n_na} NA) → {output}")
    return True


# ================================================================
#  Stage 3: roc — Parse scores, compute ROC
# ================================================================

def stage_roc(sim_dir, out_dir, p):
    """Parse S* scores, map to 10 kb windows, compute ROC."""
    scores_file = os.path.join(out_dir, "sstar_scores.txt")
    if not os.path.isfile(scores_file):
        print(f"  ERROR: {scores_file} not found. Run --stage run_sstar first.")
        return

    # --- Load node ID mapping ---
    with open(os.path.join(sim_dir, "node_id_map.json")) as f:
        id_map = json.load(f)  # str(orig_node) → simp_node
    rev_map = {v: int(k) for k, v in id_map.items()}

    # --- Parse S* output ---
    # Columns: chrom, start, end, sample, S*_score, ...
    nw = int(np.ceil(p.seq_length / p.window_size))
    sstar_scores = {}  # orig_node_id → np.array of per-10kb-window scores

    print("  Parsing S* scores...")
    n_parsed = 0
    with open(scores_file) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            score_str = parts[4]
            if score_str == "NA":
                continue

            sample_name = parts[3]
            start = int(parts[1])
            end = int(parts[2])
            score = float(score_str)
            n_parsed += 1

            try:
                ind = int(sample_name.replace("tsk_", ""))
            except ValueError:
                continue

            # Only target individuals (tsk_0..tsk_24)
            if ind >= 25:
                continue

            # Assign score to both haplotypes of this diploid individual
            for hap in (0, 1):
                simp_node = 2 * ind + hap
                if simp_node not in rev_map:
                    continue
                orig = rev_map[simp_node]

                if orig not in sstar_scores:
                    sstar_scores[orig] = np.full(nw, np.nan)

                # Map 50 kb S* window to constituent 10 kb windows
                w0 = max(0, int(start // p.window_size))
                w1 = min(int(np.ceil(end / p.window_size)), nw)
                for w in range(w0, w1):
                    cur = sstar_scores[orig][w]
                    if np.isnan(cur) or score > cur:
                        sstar_scores[orig][w] = score

    print(f"  {n_parsed} non-NA score rows parsed")
    print(f"  {len(sstar_scores)} haplotypes with scores "
          f"(expected 50 for 25 target diploids)")

    # --- Load ground truth (restricted to target haplotypes) ---
    with open(os.path.join(sim_dir, "true_tracts.json")) as f:
        all_tracts = {int(k): v for k, v in json.load(f).items()}

    # Restrict to the 50 target haplotype nodes
    target_tracts = {nid: all_tracts[nid] for nid in sstar_scores
                     if nid in all_tracts}
    labels = tracts_to_window_labels(
        target_tracts, p.seq_length, p.window_size, p.introg_label_threshold)

    # --- Compute ROC ---
    al, asc = aggregate_for_roc(labels, sstar_scores)
    fpr, tpr, auroc = compute_roc(al, asc)
    print(f"  AUROC (S* vs truth) = {auroc:.4f}  "
          f"({int(al.sum()):,} pos / {len(al):,} windows)")
    np.savez(os.path.join(sim_dir, "roc_sstar.npz"),
             fpr=fpr, tpr=tpr, auroc=auroc)

    # --- Plot combined ROC ---
    roc_dict = {}
    for name, fname in [
        ("CR (true trees)",       "roc_cr_true.npz"),
        ("CR (SINGER posterior)", "roc_cr_singer.npz"),
        ("IBDmix",                "roc_ibdmix.npz"),
        ("BPP MSC-I",            "roc_bpp.npz"),
        ("CR (Relate)",           "roc_cr_relate.npz"),
        ("S*",                    "roc_sstar.npz"),
    ]:
        path = os.path.join(sim_dir, fname)
        if os.path.exists(path):
            rd = np.load(path)
            roc_dict[name] = (rd["fpr"], rd["tpr"], float(rd["auroc"]))

    if roc_dict:
        plot_roc_curves(roc_dict,
                        os.path.join(sim_dir, "roc_comparison_all.png"))
        print()
        print("  ┌─────────────────────────────┬─────────┐")
        print("  │ Method                      │  AUROC  │")
        print("  ├─────────────────────────────┼─────────┤")
        for nm, (_, _, a) in roc_dict.items():
            print(f"  │ {nm:<27s} │ {a:.4f}  │")
        print("  └─────────────────────────────┴─────────┘")


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="S* (Wall 2006) ROC curve for ghost introgression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python %(prog)s --stage prepare      # create sample lists
  python %(prog)s --stage run_sstar    # run S* computation
  python %(prog)s --stage roc          # compute and plot ROC
  python %(prog)s --stage all          # all stages
        """)
    parser.add_argument("--sim-dir", default="introg_sim",
                        help="Directory with simulation outputs")
    parser.add_argument("--out-dir", default="sstar_output",
                        help="Working directory for S* files")
    parser.add_argument("--stage", default="all",
                        choices=["all", "prepare", "run_sstar", "roc"])
    args = parser.parse_args()

    p = Params()
    sim_dir = args.sim_dir
    out_dir = args.out_dir
    run_all = args.stage == "all"

    if run_all or args.stage == "prepare":
        print("\n" + "=" * 60)
        print("  STAGE: prepare")
        print("=" * 60)
        stage_prepare(out_dir)

    if run_all or args.stage == "run_sstar":
        print("\n" + "=" * 60)
        print("  STAGE: run_sstar")
        print("=" * 60)
        stage_run_sstar(sim_dir, out_dir)

    if run_all or args.stage == "roc":
        print("\n" + "=" * 60)
        print("  STAGE: roc")
        print("=" * 60)
        stage_roc(sim_dir, out_dir, p)

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
