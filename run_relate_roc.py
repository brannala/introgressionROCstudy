#!/usr/bin/env python3
"""
Relate-based coalescence ratio ROC curve for ghost archaic introgression.

Stages:
  install      Build Relate and relate_lib from source
  prepare      Convert VCF to Relate input format (.haps/.sample/.map)
  run_relate   Run Relate ARG inference
  cr           Convert Relate output to tskit, compute CR
  roc          Compute and plot ROC curve

Reuses functions from introgression_roc_pipeline.py.

Bruce Rannala, Feb 2026
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np

# Reuse pipeline functions
from introgression_roc_pipeline import (
    Params,
    compute_cr_for_haplotype,
    tracts_to_window_labels,
    aggregate_for_roc,
    compute_roc,
    plot_roc_curves,
)

# Tool paths. Override via environment variables, e.g.:
#   export RELATE_BIN=/path/to/Relate
#   export RELATE_FILE_FORMATS=/path/to/RelateFileFormats
#   export CONVERT_BIN=/path/to/Convert    # from relate_lib
# Defaults assume the binaries are on $PATH (see INSTALL.md).
RELATE_BIN = os.environ.get("RELATE_BIN", "Relate")
RELATE_FILE_FORMATS = os.environ.get("RELATE_FILE_FORMATS", "RelateFileFormats")
CONVERT_BIN = os.environ.get("CONVERT_BIN", "Convert")


# ================================================================
#  Stage 0: install
# ================================================================

def stage_install():
    """Build Relate and relate_lib from source."""
    repos = os.path.expanduser("~/repos")

    # --- Relate ---
    relate_dir = os.path.join(repos, "relate_build")
    if not os.path.isdir(relate_dir):
        print("  Cloning Relate...")
        subprocess.run(
            ["git", "clone", "https://github.com/MyersGroup/relate.git",
             "relate_build"],
            cwd=repos, check=True)
    build_dir = os.path.join(relate_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    if not os.path.isfile(RELATE_BIN):
        print("  Building Relate...")
        subprocess.run(
            ["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
            cwd=build_dir, check=True)
        subprocess.run(["make", "-j4"], cwd=build_dir, check=True)
    print(f"  Relate binary: {RELATE_BIN}")

    # --- relate_lib ---
    lib_dir = os.path.join(repos, "relate_lib")
    if not os.path.isdir(lib_dir):
        print("  Cloning relate_lib...")
        subprocess.run(
            ["git", "clone",
             "https://github.com/leospeidel/relate_lib.git"],
            cwd=repos, check=True)
    lib_build = os.path.join(lib_dir, "build")
    os.makedirs(lib_build, exist_ok=True)
    if not os.path.isfile(CONVERT_BIN):
        print("  Building relate_lib...")
        subprocess.run(
            ["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
            cwd=lib_build, check=True)
        subprocess.run(["make", "-j4"], cwd=lib_build, check=True)
    print(f"  Convert binary: {CONVERT_BIN}")


# ================================================================
#  Stage 1: prepare — VCF to Relate input
# ================================================================

def stage_prepare(sim_dir, out_dir, p):
    """Convert singer_input.vcf to Relate .haps/.sample/.map files."""
    os.makedirs(out_dir, exist_ok=True)
    vcf_path = os.path.join(sim_dir, "singer_input.vcf")

    # --- Parse VCF ---
    print("  Parsing VCF...")
    snp_rows = []  # list of (pos, ref, alt, genotypes_str)
    sample_names = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                sample_names = parts[9:]  # tsk_0, tsk_1, ...
                continue
            parts = line.strip().split("\t")
            pos = int(parts[1])
            snp_id = parts[2]
            ref = parts[3]
            alt = parts[4]
            # Skip multiallelic sites (Relate requires biallelic)
            if "," in alt:
                continue
            # Parse phased genotypes → haplotype columns
            haps = []
            for gt_field in parts[9:]:
                gt = gt_field.split(":")[0]  # just GT
                alleles = gt.replace("|", "/").split("/")
                haps.extend(alleles)
            snp_rows.append((pos, ref, alt, snp_id, haps))

    n_haps = len(sample_names) * 2
    n_snps = len(snp_rows)
    print(f"  {n_snps} SNPs, {n_haps} haplotypes, "
          f"{len(sample_names)} diploid samples")

    # --- Write .haps file ---
    haps_path = os.path.join(out_dir, "input.haps")
    print(f"  Writing {haps_path}...")
    with open(haps_path, "w") as f:
        for i, (pos, ref, alt, snp_id, haps) in enumerate(snp_rows):
            cols = ["1", f"snp_{i}", str(pos), ref, alt] + haps
            f.write(" ".join(cols) + "\n")

    # --- Write .sample file ---
    sample_path = os.path.join(out_dir, "input.sample")
    print(f"  Writing {sample_path}...")
    with open(sample_path, "w") as f:
        f.write("ID_1 ID_2 missing\n")
        f.write("0 0 0\n")
        for name in sample_names:
            f.write(f"{name} {name} 0\n")

    # --- Write .map file (uniform recombination) ---
    map_path = os.path.join(out_dir, "input.map")
    rate_cM_per_Mb = p.rho * 1e8  # 1.2e-8 * 1e8 = 1.2 cM/Mb
    total_cM = p.seq_length * p.rho * 100  # 50e6 * 1.2e-8 * 100 = 60 cM
    print(f"  Writing {map_path} (rate={rate_cM_per_Mb} cM/Mb, "
          f"total={total_cM} cM)...")
    with open(map_path, "w") as f:
        f.write("pos COMBINED_rate Genetic_Map\n")
        f.write(f"0 {rate_cM_per_Mb} 0\n")
        f.write(f"{int(p.seq_length)} {rate_cM_per_Mb} {total_cM}\n")

    print("  Prepare complete.")


# ================================================================
#  Stage 2: run_relate — Execute Relate
# ================================================================

def stage_run_relate(out_dir, p):
    """Run Relate ARG inference."""
    abs_out = os.path.abspath(out_dir)
    haps = os.path.join(abs_out, "input.haps")
    sample = os.path.join(abs_out, "input.sample")
    map_file = os.path.join(abs_out, "input.map")

    for f in [haps, sample, map_file]:
        if not os.path.isfile(f):
            print(f"  ERROR: Missing {f}. Run --stage prepare first.")
            return False

    # Clean up any stale temp directory from a previous run
    temp_dir = os.path.join(abs_out, "relate_arg")
    if os.path.isdir(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

    # Relate requires output in working directory, so cd into out_dir
    cmd = [
        RELATE_BIN, "--mode", "All",
        "-m", str(p.mu),
        "-N", str(2 * p.Ne_afr),
        "--haps", haps,
        "--sample", sample,
        "--map", map_file,
        "-o", "relate_arg",
    ]
    print(f"  $ cd {abs_out} && {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=abs_out)
    if r.returncode != 0:
        print(f"  Relate STDERR:\n{r.stderr}", file=sys.stderr)
        print(f"  Relate STDOUT:\n{r.stdout}")
        return False

    anc = os.path.join(abs_out, "relate_arg.anc")
    mut = os.path.join(abs_out, "relate_arg.mut")
    if os.path.isfile(anc) and os.path.isfile(mut):
        anc_sz = os.path.getsize(anc) / 1e6
        mut_sz = os.path.getsize(mut) / 1e6
        print(f"  Output: {anc} ({anc_sz:.1f} MB), {mut} ({mut_sz:.1f} MB)")
        return True
    else:
        print("  ERROR: Relate output files not found.")
        print(f"  STDOUT: {r.stdout[:500]}")
        return False


# ================================================================
#  Stage 3: cr — Convert to tskit + compute CR
# ================================================================

def stage_cr(sim_dir, out_dir, p):
    """Convert Relate output to tskit and compute coalescence ratios."""
    import tskit

    anc = os.path.join(out_dir, "relate_arg.anc")
    mut = os.path.join(out_dir, "relate_arg.mut")
    ts_prefix = os.path.join(out_dir, "relate_ts")
    ts_path = ts_prefix + ".trees"

    # --- Convert to tskit ---
    if not os.path.isfile(ts_path):
        print("  Converting Relate output to tskit...")
        abs_out = os.path.abspath(out_dir)
        cmd = [
            RELATE_FILE_FORMATS, "--mode", "ConvertToTreeSequence",
            "-i", "relate_arg",
            "-o", "relate_ts",
        ]
        print(f"  $ cd {abs_out} && {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=abs_out)
        if r.returncode != 0:
            print(f"  Convert STDERR:\n{r.stderr}", file=sys.stderr)
            print(f"  Convert STDOUT:\n{r.stdout}")
            return None
    else:
        print(f"  Using existing {ts_path}")

    # --- Load tree sequence ---
    ts = tskit.load(ts_path)
    print(f"  Relate tree sequence: {ts.num_samples} samples, "
          f"{ts.num_trees} trees, {ts.num_sites} sites")

    # --- Load node ID mapping ---
    with open(os.path.join(sim_dir, "node_id_map.json")) as f:
        id_map = json.load(f)  # str(orig) → simp
    rev_map = {v: int(k) for k, v in id_map.items()}

    # Relate sample nodes 0..99 correspond to haplotypes in .haps order,
    # which matches the simplified node IDs from the VCF.
    n_samples = ts.num_samples
    relate_samples = list(range(n_samples))  # 0..99
    print(f"  Relate samples: {relate_samples[:3]}...{relate_samples[-3:]}")

    # --- Compute CR for each haplotype ---
    print("  Computing coalescence ratios...")
    cr_result = {}
    for i, focal in enumerate(relate_samples):
        others = [n for n in relate_samples if n != focal]
        sys.stdout.write(f"\r  CR haplotype {i+1}/{n_samples}")
        sys.stdout.flush()
        cr = compute_cr_for_haplotype(ts, focal, others, p)
        orig_id = rev_map[focal]
        cr_result[orig_id] = cr
    print()

    # --- Save ---
    out_file = os.path.join(sim_dir, "cr_relate.npz")
    np.savez(out_file, **{str(k): v for k, v in cr_result.items()})
    print(f"  Saved CR data → {out_file}")

    # --- Quick sanity check ---
    all_cr = np.concatenate(list(cr_result.values()))
    finite = all_cr[np.isfinite(all_cr)]
    print(f"  CR stats: median={np.median(finite):.2f}, "
          f"mean={np.mean(finite):.2f}, "
          f"p95={np.percentile(finite, 95):.2f}")

    return cr_result


# ================================================================
#  Stage 4: roc — Compute and plot ROC
# ================================================================

def stage_roc(sim_dir, p):
    """Compute Relate ROC and plot combined ROC curves."""
    # --- Load CR data ---
    cr_file = os.path.join(sim_dir, "cr_relate.npz")
    if not os.path.isfile(cr_file):
        print(f"  ERROR: {cr_file} not found. Run --stage cr first.")
        return
    d = np.load(cr_file)
    cr_relate = {int(k): d[k] for k in d.files}

    # --- Load ground truth ---
    with open(os.path.join(sim_dir, "true_tracts.json")) as f:
        tracts = {int(k): v for k, v in json.load(f).items()}
    labels = tracts_to_window_labels(
        tracts, p.seq_length, p.window_size, p.introg_label_threshold)

    # --- Compute ROC ---
    al, asc = aggregate_for_roc(labels, cr_relate)
    fpr, tpr, auroc = compute_roc(al, asc)
    print(f"  AUROC (CR Relate vs truth) = {auroc:.4f}  "
          f"({int(al.sum()):,} pos / {len(al):,} windows)")
    np.savez(os.path.join(sim_dir, "roc_cr_relate.npz"),
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
        ("BPP MSC-I (no ghost)",  "roc_bpp_noghost.npz"),
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
        description="Relate-based coalescence ratio ROC curve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python %(prog)s --stage install       # build Relate from source
  python %(prog)s --stage prepare       # VCF → Relate input
  python %(prog)s --stage run_relate    # run Relate inference
  python %(prog)s --stage cr            # convert to tskit + compute CR
  python %(prog)s --stage roc           # plot ROC curves
  python %(prog)s --stage all           # all stages
        """)
    parser.add_argument("--sim-dir", default="introg_sim",
                        help="Directory with simulation outputs")
    parser.add_argument("--out-dir", default="relate_output",
                        help="Working directory for Relate files")
    parser.add_argument("--stage", default="all",
                        choices=["all", "install", "prepare",
                                 "run_relate", "cr", "roc"])
    args = parser.parse_args()

    p = Params()
    sim_dir = args.sim_dir
    out_dir = args.out_dir
    run_all = args.stage == "all"

    if run_all or args.stage == "install":
        print("\n" + "=" * 60)
        print("  STAGE: install")
        print("=" * 60)
        stage_install()

    if run_all or args.stage == "prepare":
        print("\n" + "=" * 60)
        print("  STAGE: prepare")
        print("=" * 60)
        stage_prepare(sim_dir, out_dir, p)

    if run_all or args.stage == "run_relate":
        print("\n" + "=" * 60)
        print("  STAGE: run_relate")
        print("=" * 60)
        ok = stage_run_relate(out_dir, p)
        if not ok and not run_all:
            sys.exit(1)

    if run_all or args.stage == "cr":
        print("\n" + "=" * 60)
        print("  STAGE: cr")
        print("=" * 60)
        stage_cr(sim_dir, out_dir, p)

    if run_all or args.stage == "roc":
        print("\n" + "=" * 60)
        print("  STAGE: roc")
        print("=" * 60)
        stage_roc(sim_dir, p)

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
