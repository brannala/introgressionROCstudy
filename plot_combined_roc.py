#!/usr/bin/env python3
"""
Plot a combined ROC figure from all roc_*.npz files in a simdir.

Reads every fixed-name method (CR true, CR SINGER, IBDmix, CR Relate, S*),
plus per-chain BPP MSC-I curves discovered by globbing roc_bpp*.npz so r1/r2
appear as overlaid lines. Writes a single PNG; intended to replace the
per-script combined plots that only know about a subset of methods.

Usage:
  python plot_combined_roc.py --simdir sim_A
  python plot_combined_roc.py --simdir sim_B \
      --out sim_B/roc_comparison_all.pdf \
      --title "Ghost Archaic Introgression Detection (sim B)"
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "CR (true trees)":       {"color": "#d62728", "ls": "-",  "lw": 2.5},
    "CR (SINGER posterior)": {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
    "IBDmix":                {"color": "#2ca02c", "ls": "-",  "lw": 2.0},
    "BPP MSC-I":             {"color": "#ff7f0e", "ls": "-",  "lw": 2.0},
    "CR (Relate)":           {"color": "#8c564b", "ls": "-",  "lw": 2.0},
    "S*":                    {"color": "#e377c2", "ls": "-",  "lw": 2.0},
}


def load_method(simdir, fname, label):
    """Load one roc_*.npz if it exists; return (label, fpr, tpr, auroc) or None."""
    path = os.path.join(simdir, fname)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (label, d["fpr"], d["tpr"], float(d["auroc"]))


def load_all(simdir):
    """Return list of (label, fpr, tpr, auroc) sorted by AUROC descending.

    BPP r1/r2 (and any other equal-AUROC pairs) keep insertion order so
    the chain replicates sit adjacent in the legend.
    """
    items = []
    for fname, label in [
        ("roc_cr_true.npz",     "CR (true trees)"),
        ("roc_cr_singer.npz",   "CR (SINGER posterior)"),
        ("roc_cr_relate.npz",   "CR (Relate)"),
        ("roc_ibdmix.npz",      "IBDmix"),
        ("roc_sstar.npz",       "S*"),
    ]:
        row = load_method(simdir, fname, label)
        if row is not None:
            items.append(row)

    # BPP — discover all chains via glob so r1/r2 etc. each appear as a curve
    bpp_paths = sorted(glob.glob(os.path.join(simdir, "roc_bpp*.npz")))
    for path in bpp_paths:
        stem = os.path.basename(path)[len("roc_bpp"):-len(".npz")]
        label = f"BPP MSC-I ({stem[1:]})" if stem.startswith("_") else "BPP MSC-I"
        d = np.load(path)
        items.append((label, d["fpr"], d["tpr"], float(d["auroc"])))

    # Sort by AUROC descending; stable sort preserves BPP r1 before r2
    items.sort(key=lambda row: row[3], reverse=True)
    return items


def plot(items, outpath, title="Ghost Archaic Introgression Detection"):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, fpr, tpr, auroc in items:
        base_label = label.split(" (")[0] if label.startswith("BPP MSC-I (") else label
        style = STYLES.get(base_label, {"color": "gray", "ls": "-", "lw": 1.5})
        ax.plot(fpr, tpr, label=f"{label}  (AUC = {auroc:.3f})", **style)
    ax.plot([0, 1], [0, 1], "k:", lw=0.7, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simdir", default="introg_sim",
                    help="Directory containing roc_*.npz files")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: <simdir>/roc_comparison_all.png)")
    ap.add_argument("--title", default="Ghost Archaic Introgression Detection")
    args = ap.parse_args()

    items = load_all(args.simdir)
    if not items:
        raise SystemExit(f"No roc_*.npz files found in {args.simdir}")
    outpath = args.out or os.path.join(args.simdir, "roc_comparison_all.png")
    plot(items, outpath, title=args.title)
    print(f"Saved -> {outpath}")
    print()
    print(f"  {'Method':<28s}  AUROC")
    print(f"  {'-' * 28}  -----")
    for label, _, _, auroc in items:
        print(f"  {label:<28s}  {auroc:.4f}")


if __name__ == "__main__":
    main()
