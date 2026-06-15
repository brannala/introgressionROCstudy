#!/usr/bin/env python3
"""
BPP MSC-I introgression detection ROC pipeline.

Extracts 10 kb window alignments from the msprime simulation, runs BPP
with an MSC-I (introgression) model, and computes ROC curves comparing
BPP's per-locus introgression posterior probabilities against the
simulation ground truth.

Stages:
  prepare  — Extract 10 kb window alignments from sim.trees for BPP
  run_bpp  — Execute BPP inference
  roc      — Parse BPP output, compute and plot ROC curves

Usage:
  python run_bpp_roc.py --stage prepare
  python run_bpp_roc.py --stage run_bpp
  python run_bpp_roc.py --stage roc
  python run_bpp_roc.py --stage all

  # Quick smoke test (10 loci, ~seconds):
  python run_bpp_roc.py --stage prepare --nloci 10 --bppdir bpp_test
  python run_bpp_roc.py --stage run_bpp --bppdir bpp_test

  # Multi-chain ROC (between-chain convergence visualization):
  #   --bppdir points at a directory holding r1/, r2/ subdirs, each
  #   containing the BPP ancestry output.
  python run_bpp_roc.py --stage roc \\
      --simdir sim_A \\
      --bppdir sim_A/bpp_combo \\
      --chains r1 r2
"""

import os
import sys
import glob
import json
import argparse
import subprocess
import numpy as np
from collections import defaultdict

import tskit

from introgression_roc_pipeline import (
    Params, tracts_to_window_labels, aggregate_for_roc, compute_roc,
    plot_roc_curves,
)


# ================================================================
#  1.  PREPARE BPP INPUT
# ================================================================

def prepare_bpp_data(simdir, bppdir, n_afr, nloci, threads,
                     no_ghost=False):
    """
    Extract 10 kb window alignments from sim.trees for BPP MSC-I.

    Subsamples n_afr AFR (+ 2 GHOST unless no_ghost=True) haplotypes, writes:
      sequences.phy   — multi-locus phylip (nloci × n_samples × 10000 bp)
      imap.txt        — sample-to-species mapping
      infer.ctl       — BPP control file
      sample_ids.json — BPP name → original node ID
    """
    p = Params()
    window_size = p.window_size

    ts = tskit.load(os.path.join(simdir, "sim.trees"))
    print(f"  Loaded: {ts.num_samples} samples, {ts.num_sites:,} sites")

    # Select samples
    afr_orig = list(range(n_afr))
    if no_ghost:
        keep_nodes = afr_orig
        print(f"  Subsampling: {n_afr} AFR (no GHOST) = {len(keep_nodes)} haplotypes")
    else:
        ghost_orig = [2 * p.n_afr_diploid, 2 * p.n_afr_diploid + 1]
        keep_nodes = afr_orig + ghost_orig
        print(f"  Subsampling: {n_afr} AFR + 2 GHOST = {len(keep_nodes)} haplotypes")
    n_samples = len(keep_nodes)

    # BPP name → original node ID
    sample_ids = {}
    bpp_names = []
    for i, nid in enumerate(afr_orig):
        name = f"afr{i+1}"
        sample_ids[name] = nid
        bpp_names.append(f"AFR^{name}")
    if not no_ghost:
        for i, nid in enumerate(ghost_orig):
            name = f"ghost{i+1}"
            sample_ids[name] = nid
            bpp_names.append(f"GHOST^{name}")

    # Simplify (must clear migration records first)
    tables = ts.dump_tables()
    tables.migrations.clear()
    ts_sub = tables.tree_sequence().simplify(samples=keep_nodes)
    print(f"  Simplified: {ts_sub.num_samples} samples, "
          f"{ts_sub.num_sites:,} sites")

    max_loci = int(ts_sub.sequence_length // window_size)
    nloci = min(nloci, max_loci)
    print(f"  Windows: {nloci} x {window_size} bp")

    # Collect variants by window
    print("  Extracting variant sites...")
    window_variants = defaultdict(list)
    for var in ts_sub.variants():
        w = int(var.site.position // window_size)
        if w >= nloci:
            break
        pos = int(var.site.position) - w * window_size
        window_variants[w].append(
            (pos, list(var.alleles), var.genotypes.copy()))

    total_vars = sum(len(v) for v in window_variants.values())
    n_with_vars = len(window_variants)
    print(f"  {total_vars:,} variant sites across {n_with_vars} windows "
          f"(mean {total_vars / nloci:.1f}/window)")

    # Write output files
    os.makedirs(bppdir, exist_ok=True)
    max_name_len = max(len(n) for n in bpp_names)
    name_width = max_name_len + 2

    # --- sequences.phy ---
    seq_path = os.path.join(bppdir, "sequences.phy")
    print(f"  Writing {seq_path} ...")
    with open(seq_path, "w") as f:
        for w in range(nloci):
            if w > 0:
                f.write("\n")
            f.write(f"{n_samples} {window_size}\n\n")
            variants = window_variants.get(w, [])
            for s in range(n_samples):
                # Invariant sites → 'A'; variant sites → actual allele
                seq = bytearray(b'A' * window_size)
                for pos, alleles, genos in variants:
                    allele = alleles[genos[s]]
                    if allele:
                        seq[pos] = ord(allele[0])
                f.write(f"{bpp_names[s]:<{name_width}s}{seq.decode()}\n")
            if (w + 1) % 1000 == 0:
                sys.stdout.write(f"\r    {w + 1}/{nloci} loci written")
                sys.stdout.flush()
    if nloci >= 1000:
        print()

    # --- imap.txt ---
    imap_path = os.path.join(bppdir, "imap.txt")
    with open(imap_path, "w") as f:
        for i in range(n_afr):
            f.write(f"afr{i+1}\tAFR\n")
        if not no_ghost:
            f.write("ghost1\tGHOST\n")
            f.write("ghost2\tGHOST\n")

    # --- infer.ctl ---
    n_ghost = 0 if no_ghost else 2
    ctl_path = os.path.join(bppdir, "infer.ctl")
    ctl_text = (
        f"seed = 123\n"
        f"seqfile = sequences.phy\n"
        f"Imapfile = imap.txt\n"
        f"jobname = bpp_introg\n"
        f"\n"
        f"speciesdelimitation = 0\n"
        f"speciestree = 0\n"
        f"\n"
        f"species&tree = 2 AFR GHOST\n"
        f"                 {n_afr}  {n_ghost}\n"
        f"                 ((AFR)H[&phi=0.97,&tau-parent=yes], "
        f"(GHOST, H[&tau-parent=no])S)R;\n"
        f"\n"
        f"phase = 0\n"
        f"usedata = 1\n"
        f"nloci = {nloci}\n"
        f"cleandata = 0\n"
        f"\n"
        f"thetaprior = gamma 2 2000\n"
        f"tauprior = gamma 20 93000\n"
        f"phiprior = 1 1\n"
        f"\n"
        f"finetune = 1\n"
        f"print = 1 0 0 0\n"
        f"burnin = 16000\n"
        f"sampfreq = 2\n"
        f"nsample = 40000\n"
        f"threads = {threads}\n"
        f"ancestry = 1\n"
    )
    with open(ctl_path, "w") as f:
        f.write(ctl_text)

    # --- sample_ids.json ---
    ids_path = os.path.join(bppdir, "sample_ids.json")
    with open(ids_path, "w") as f:
        json.dump(sample_ids, f, indent=2)

    seq_size = os.path.getsize(seq_path) / 1e6
    print(f"\n  Output in {bppdir}/:")
    print(f"    sequences.phy  ({seq_size:.1f} MB, {nloci} loci)")
    print(f"    imap.txt       ({n_samples} samples)")
    print(f"    infer.ctl      (threads={threads})")
    print(f"    sample_ids.json")
    return nloci


# ================================================================
#  2.  RUN BPP
# ================================================================

BPP_DEFAULT = "bpp"  # assumes the dev-branch binary is on $PATH (see INSTALL.md)


def run_bpp(bppdir, bpp_bin=BPP_DEFAULT):
    """Execute BPP inference."""
    ctl_path = os.path.join(bppdir, "infer.ctl")
    if not os.path.exists(ctl_path):
        print(f"  ERROR: {ctl_path} not found. Run --stage prepare first.")
        return False

    print(f"  $ {bpp_bin} --cfile infer.ctl  (cwd={bppdir})")
    result = subprocess.run([bpp_bin, "--cfile", "infer.ctl"], cwd=bppdir)
    if result.returncode != 0:
        print(f"  BPP failed (rc={result.returncode})")
        return False

    ancestry = os.path.join(
        bppdir, "bpp_introg.ancestry.per-sequence-per-locus.txt")
    if os.path.exists(ancestry):
        print(f"  Output: {ancestry}")
        return True
    print(f"  WARNING: {ancestry} not found")
    return False


# ================================================================
#  3.  PARSE ANCESTRY OUTPUT
# ================================================================

def parse_ancestry(bppdir):
    """
    Parse BPP dev-branch ancestry output → {original_node_id: np.array(nloci)}.

    Reads ancestry.per-sequence-per-locus.txt (CSV: Locus,Sequence,PP).
    Only keeps AFR sequences (skips GHOST).
    """
    ancestry_path = os.path.join(
        bppdir, "bpp_introg.ancestry.per-sequence-per-locus.txt")
    ids_path = os.path.join(bppdir, "sample_ids.json")

    with open(ids_path) as f:
        sample_ids = json.load(f)  # BPP name → original node ID

    scores = defaultdict(dict)
    max_locus = 0

    with open(ancestry_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            locus = int(parts[0])
            seq_name = parts[1]
            pp = float(parts[2])

            # Skip GHOST sequences
            if seq_name.startswith("GHOST^"):
                continue

            # "AFR^afr1" → "afr1"
            ind_name = seq_name.split('^')[1] if '^' in seq_name else seq_name
            if ind_name not in sample_ids:
                continue

            orig_id = sample_ids[ind_name]
            scores[orig_id][locus - 1] = pp  # BPP is 1-based
            max_locus = max(max_locus, locus)

    nloci = max_locus
    result = {}
    for orig_id, locus_scores in scores.items():
        arr = np.zeros(nloci)
        for idx, pp in locus_scores.items():
            arr[idx] = pp
        result[orig_id] = arr

    print(f"  Parsed {len(result)} AFR haplotypes x {nloci} loci")
    return result


# ================================================================
#  4.  COMPUTE AND SAVE ROC
# ================================================================

def compute_and_save_roc(simdir, bppdir, chain_label=None):
    """Compute ROC for BPP MSC-I and save.

    chain_label: if given, output is roc_bpp_{label}.npz and the method
    label in the print/plot is 'BPP MSC-I ({label})'. If None (default),
    output is roc_bpp.npz with label 'BPP MSC-I' — original behavior.
    """
    p = Params()
    scores = parse_ancestry(bppdir)

    with open(os.path.join(simdir, "true_tracts.json")) as f:
        tracts = {int(k): v for k, v in json.load(f).items()}

    labels = tracts_to_window_labels(
        tracts, p.seq_length, p.window_size, p.introg_label_threshold)

    al, asc = aggregate_for_roc(labels, scores)
    fpr, tpr, auroc = compute_roc(al, asc)

    method_label = f"BPP MSC-I ({chain_label})" if chain_label else "BPP MSC-I"
    suffix = f"_{chain_label}" if chain_label else ""

    print(f"  AUROC ({method_label}) = {auroc:.4f}")
    print(f"  ({int(al.sum()):,} positive / {len(al):,} total windows)")

    outpath = os.path.join(simdir, f"roc_bpp{suffix}.npz")
    np.savez(outpath, fpr=fpr, tpr=tpr, auroc=auroc)
    print(f"  Saved -> {outpath}")
    return fpr, tpr, auroc


# ================================================================
#  5.  PLOT COMBINED ROC
# ================================================================

def plot_combined_roc(simdir):
    """Load all ROC files and plot combined curves.

    BPP results are discovered by globbing roc_bpp*.npz so per-chain
    files (roc_bpp_r1.npz, roc_bpp_r2.npz, ...) each appear as a
    separate curve labeled 'BPP MSC-I (r1)', 'BPP MSC-I (r2)', etc.
    """
    roc_dict = {}
    for name, fname in [
        ("CR (true trees)",       "roc_cr_true.npz"),
        ("CR (SINGER posterior)", "roc_cr_singer.npz"),
        ("IBDmix",                "roc_ibdmix.npz"),
    ]:
        path = os.path.join(simdir, fname)
        if os.path.exists(path):
            d = np.load(path)
            roc_dict[name] = (d["fpr"], d["tpr"], float(d["auroc"]))

    for path in sorted(glob.glob(os.path.join(simdir, "roc_bpp*.npz"))):
        stem = os.path.basename(path)[len("roc_bpp"):-len(".npz")]
        label = f"BPP MSC-I ({stem[1:]})" if stem.startswith("_") else "BPP MSC-I"
        d = np.load(path)
        roc_dict[label] = (d["fpr"], d["tpr"], float(d["auroc"]))

    if roc_dict:
        plot_roc_curves(roc_dict,
                        os.path.join(simdir, "roc_comparison_bpp.png"))
        print()
        print("  +-----------------------------+---------+")
        print("  | Method                      |  AUROC  |")
        print("  +-----------------------------+---------+")
        for nm, (_, _, a) in roc_dict.items():
            print(f"  | {nm:<27s} | {a:.4f}  |")
        print("  +-----------------------------+---------+")
    else:
        print("  No ROC data found.")


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BPP MSC-I introgression detection ROC pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  prepare   Extract 10 kb window alignments from sim.trees
  run_bpp   Execute BPP inference (~9-10 days for 100 haplotypes, 5000 loci)
  roc       Parse BPP output and compute/plot ROC curves
  all       Run all stages sequentially
        """)
    parser.add_argument("--stage", default="all",
                        choices=["all", "prepare", "run_bpp", "roc"])
    parser.add_argument("--simdir", default="introg_sim",
                        help="Simulation directory (default: introg_sim)")
    parser.add_argument("--bppdir", default="bpp_introg",
                        help="BPP working directory (default: bpp_introg)")
    parser.add_argument("--n-afr", type=int, default=100,
                        help="Number of AFR haplotypes (default: 100)")
    parser.add_argument("--threads", type=int, default=4,
                        help="BPP thread count (default: 4; requires the dev "
                             "branch at commit 1a8e4ab or later, which fixes "
                             "a race condition in per-sequence ancestry)")
    parser.add_argument("--nloci", type=int, default=5000,
                        help="Number of 10 kb windows (default: 5000)")
    parser.add_argument("--bpp-bin", default=BPP_DEFAULT,
                        help=f"Path to bpp binary (default: {BPP_DEFAULT})")
    parser.add_argument("--no-ghost", action="store_true",
                        help="AFR-only analysis (no GHOST sequences)")
    parser.add_argument("--chains", nargs="+", default=None, metavar="NAME",
                        help="Chain subdirectory names under --bppdir "
                             "(e.g. --chains r1 r2). When given, run_bpp "
                             "executes BPP in each subdir and roc writes "
                             "a separate roc_bpp_{NAME}.npz per chain so "
                             "they plot as overlay curves. Omit for "
                             "single-chain (legacy) behavior.")
    args = parser.parse_args()

    run_all = args.stage == "all"

    if run_all or args.stage == "prepare":
        print("\n" + "=" * 60)
        print("  STAGE: prepare")
        print("=" * 60)
        prepare_bpp_data(args.simdir, args.bppdir, args.n_afr,
                         args.nloci, args.threads,
                         no_ghost=args.no_ghost)

    chains = args.chains or [None]  # None ⇒ single-chain (bppdir itself)

    if run_all or args.stage == "run_bpp":
        print("\n" + "=" * 60)
        print("  STAGE: run_bpp")
        print("=" * 60)
        for chain in chains:
            sub = os.path.join(args.bppdir, chain) if chain else args.bppdir
            if chain:
                print(f"\n  --- Chain: {chain} ({sub}) ---")
            run_bpp(sub, args.bpp_bin)

    if run_all or args.stage == "roc":
        print("\n" + "=" * 60)
        print("  STAGE: roc")
        print("=" * 60)
        for chain in chains:
            sub = os.path.join(args.bppdir, chain) if chain else args.bppdir
            if chain:
                print(f"\n  --- Chain: {chain} ({sub}) ---")
            compute_and_save_roc(args.simdir, sub, chain_label=chain)
        plot_combined_roc(args.simdir)

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
