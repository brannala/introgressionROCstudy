#!/usr/bin/env python3
"""
Simulation-based evaluation of ARG-based introgression detection.

Corrected model matching SINGER (Deng, Nielsen & Song 2025, Nat Genet)
Figure 6: archaic introgression from an UNSAMPLED GHOST population
into AFRICANS, detected using coalescence ratio on inferred ARGs.

Demography:
    ANC ──┬── (τ = 500 kya) ──── GHOST  (archaic, unsampled)
          │                         │
          │               introgression pulse (60 kya, ~3%)
          │                         ↓
          └── (τ = 500 kya) ──── AFR    (sampled Africans)

The ghost population is unsampled EXCEPT for one individual used as
the archaic reference genome for IBDmix.  SINGER sees only AFR.

Four ROC curves:
  1. CR (true trees)       vs simulation truth   — oracle upper bound
  2. CR (SINGER posterior)  vs simulation truth   — true SINGER performance
  3. IBDmix LOD            vs simulation truth   — never evaluated in paper
  4. CR (SINGER posterior)  vs IBDmix labels      — reproduces paper's method

Pipeline stages:
  simulate   → msprime simulation + ground truth extraction
  cr_true    → coalescence ratio on true tree sequence
  run_ibdmix → IBDmix tract calling (needs IBDmix installed)
  run_singer → SINGER inference (needs SINGER installed)
  cr_singer  → CR from SINGER posterior samples
  roc        → plot all ROC curves

Dependencies:
  pip install msprime tskit numpy matplotlib

External:
  SINGER ≥ v0.1.8: https://github.com/popgenmethods/SINGER
  IBDmix ≥ v1.0.1: conda install -c bioconda ibdmix

Bruce Rannala, Feb 2026
"""

import msprime
import tskit
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import json
import subprocess
import shutil
import argparse
import multiprocessing
from collections import defaultdict


# ================================================================
#  PARAMETERS
# ================================================================

class Params:
    """Simulation and analysis parameters."""

    # -- Demographic model (SINGER Fig 6a) --
    gen_time       = 28          # years per generation
    Ne_afr         = 20_000      # African effective population size
    Ne_ghost       = 10_000      # ghost archaic Ne
    Ne_anc         = 20_000      # ancestral Ne

    # Times in years → converted to generations via properties
    t_split_yrs    = 500_000     # AFR–GHOST divergence
    t_introg_yrs   = 60_000      # introgression pulse (GHOST → AFR)
    introg_frac    = 0.03        # proportion of AFR lineages from GHOST

    # -- Genome --
    seq_length     = 50_000_000  # 50 Mb
    mu             = 1.2e-8      # per-bp per-gen mutation rate
    rho            = 1.2e-8      # per-bp per-gen recombination rate

    # -- Samples --
    n_afr_diploid  = 50          # 100 AFR haplotypes (SINGER input)
    n_ghost_diploid = 1          # 2 haplotypes (archaic ref for IBDmix)

    # -- Analysis --
    window_size    = 10_000      # 10 kb windows for CR computation
    introg_label_threshold = 0.5 # fraction of window → label = 1

    # -- SINGER --
    singer_bin       = "singer_master"
    convert_bin      = "convert_to_tskit"
    singer_n_samples = 100       # posterior samples
    singer_thin      = 20

    # -- IBDmix --
    ibdmix_generate_gt = "generate_gt"
    ibdmix_bin         = "ibdmix"
    ibdmix_lod_threshold = 3.0
    ibdmix_min_length    = 50_000  # minimum tract length (bp)

    @property
    def t_split(self):
        return self.t_split_yrs / self.gen_time

    @property
    def t_introg(self):
        return self.t_introg_yrs / self.gen_time


# ================================================================
#  1.  DEMOGRAPHIC MODEL & SIMULATION
# ================================================================

def build_demography(p: Params) -> msprime.Demography:
    """
    Two-population model with ghost archaic introgression into Africans.

        ANC ──┬── t_split ──── GHOST  (archaic)
              │                   │
              │         pulse at t_introg (3%)
              │                   ↓
              └── t_split ──── AFR    (modern Africans)

    Backward in time: at t_introg, fraction p of AFR lineages jump
    to GHOST.  At t_split, GHOST and AFR merge into ANC.
    """
    dem = msprime.Demography()
    dem.add_population(name="AFR",   initial_size=p.Ne_afr)
    dem.add_population(name="GHOST", initial_size=p.Ne_ghost)
    dem.add_population(name="ANC",   initial_size=p.Ne_anc)

    # Introgression: backward, AFR lineages → GHOST at t_introg
    # (forward: GHOST → AFR gene flow)
    dem.add_mass_migration(
        time=p.t_introg, source="AFR", dest="GHOST",
        proportion=p.introg_frac
    )

    # Population split: AFR and GHOST merge into ANC
    dem.add_population_split(
        time=p.t_split, derived=["AFR", "GHOST"], ancestral="ANC"
    )

    dem.sort_events()
    return dem


def simulate(p: Params, seed: int = 42) -> tskit.TreeSequence:
    """Run coalescent simulation with mutations."""
    dem = build_demography(p)
    samples = [
        msprime.SampleSet(p.n_afr_diploid,   population="AFR",   ploidy=2),
        msprime.SampleSet(p.n_ghost_diploid,  population="GHOST", ploidy=2),
    ]
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=dem,
        sequence_length=p.seq_length,
        recombination_rate=p.rho,
        record_migrations=True,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=p.mu, random_seed=seed + 1)
    return ts


def get_pop_map(ts: tskit.TreeSequence) -> dict:
    """Return dict: population_name → [sample node IDs]."""
    pop_map = {}
    for pop in ts.populations():
        md = pop.metadata
        if isinstance(md, bytes):
            md = json.loads(md) if md else {}
        name = md.get("name", str(pop.id))
        nodes = [n.id for n in ts.nodes()
                 if n.population == pop.id and n.is_sample()]
        if nodes:
            pop_map[name] = nodes
    return pop_map


# ================================================================
#  2.  GROUND TRUTH EXTRACTION
# ================================================================

def extract_true_introgression(ts: tskit.TreeSequence, p: Params):
    """
    Extract exact introgression tracts from migration records.

    Requires the simulation to have been run with record_migrations=True.
    A segment [l, r) is introgressed for AFR haplotype i if and only if
    the migrations table contains a record moving an ancestral lineage
    of i from AFR to GHOST (backward in time) at t_introg over an
    interval covering [l, r).

    The old TMRCA-based criterion (TMRCA with ghost ref < t_split) had
    no false positives but ~46% false negatives: two lineages in GHOST
    with Ne=10k over 15,714 generations fail to coalesce with probability
    exp(-15714/20000) ≈ 0.46.

    Returns dict: afr_node_id → [(left, right), ...]
    """
    if ts.num_migrations == 0:
        raise ValueError(
            "No migration records found. Re-run simulation with "
            "record_migrations=True."
        )

    pop_map = get_pop_map(ts)
    afr_nodes = set(pop_map["AFR"])

    # Get population IDs by name
    pop_id_by_name = {}
    for pop in ts.populations():
        md = pop.metadata
        if isinstance(md, bytes):
            md = json.loads(md) if md else {}
        name = md.get("name", str(pop.id))
        pop_id_by_name[name] = pop.id

    afr_pop = pop_id_by_name["AFR"]
    ghost_pop = pop_id_by_name["GHOST"]

    # Collect migration records: AFR → GHOST (backward) at t_introg
    mig_intervals = defaultdict(list)
    for mig in ts.migrations():
        if mig.source == afr_pop and mig.dest == ghost_pop:
            mig_intervals[mig.node].append((mig.left, mig.right))

    for node in mig_intervals:
        mig_intervals[node].sort()

    n_mig_nodes = len(mig_intervals)
    n_mig_segs = sum(len(v) for v in mig_intervals.values())
    print(f"  {n_mig_nodes} migrated lineages, {n_mig_segs} migration segments")

    tracts = {n: [] for n in afr_nodes}

    # For each tree, find AFR leaves descended from migrated nodes
    for tree in ts.trees():
        tl, tr = tree.interval
        for mig_node, intervals in mig_intervals.items():
            for ml, mr in intervals:
                if mr <= tl or ml >= tr:
                    continue
                ol = max(tl, ml)
                or_ = min(tr, mr)
                try:
                    for leaf in tree.leaves(mig_node):
                        if leaf in afr_nodes:
                            tracts[leaf].append((ol, or_))
                except ValueError:
                    pass  # node not in this tree

    for nid in tracts:
        tracts[nid] = _merge_intervals(sorted(tracts[nid]))
    return tracts


def _merge_intervals(ivs):
    """Merge sorted overlapping/adjacent intervals."""
    if not ivs:
        return []
    merged = [ivs[0]]
    for l, r in ivs[1:]:
        if l <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))
    return merged


# ================================================================
#  3.  COALESCENCE RATIO
# ================================================================

def compute_cr_for_haplotype(ts, focal, others, p: Params):
    """
    Windowed coalescence ratio for one focal leaf.

    CR(w) = (weighted count of pairwise TMRCAs ≥ t_split)
          / (weighted count with t_introg ≤ TMRCA < t_split)

    Introgressed windows show HIGH CR: coalescence is depleted in the
    [t_introg, t_split) interval (because the focal lineage jumped to
    GHOST) and enriched above t_split (eventual coalescence in ANC).

    Weights account for fractional tree overlap with each window.
    """
    W = p.window_size
    n_windows = int(np.ceil(ts.sequence_length / W))
    mass_above  = np.zeros(n_windows)
    mass_within = np.zeros(n_windows)

    t_split  = p.t_split
    t_introg = p.t_introg

    for tree in ts.trees():
        tl, tr = tree.interval
        w0 = int(tl // W)
        w1 = min(int(np.ceil(tr / W)), n_windows)

        tmrcas = np.array([
            ts.node(tree.mrca(focal, o)).time
            if tree.mrca(focal, o) != tskit.NULL else np.inf
            for o in others
        ])
        n_above  = np.sum(tmrcas >= t_split)
        n_within = np.sum((tmrcas >= t_introg) & (tmrcas < t_split))

        for w in range(w0, w1):
            wl = w * W
            wr = min((w + 1) * W, ts.sequence_length)
            frac = (min(tr, wr) - max(tl, wl)) / W
            mass_above[w]  += frac * n_above
            mass_within[w] += frac * n_within

    with np.errstate(divide="ignore", invalid="ignore"):
        cr = mass_above / mass_within
    bad = mass_within == 0
    cr[bad & (mass_above > 0)] = 1e6
    cr[bad & (mass_above == 0)] = 0.0
    return cr


def compute_cr_all_afr(ts, p: Params, pop_map=None):
    """Compute CR for every AFR haplotype. Returns dict: node_id → cr array."""
    if pop_map is None:
        pop_map = get_pop_map(ts)
    afr = pop_map["AFR"]
    result = {}
    for i, focal in enumerate(afr):
        others = [n for n in afr if n != focal]
        sys.stdout.write(f"\r  CR haplotype {i+1}/{len(afr)}")
        sys.stdout.flush()
        result[focal] = compute_cr_for_haplotype(ts, focal, others, p)
    print()
    return result


def compute_cr_posterior_average(trees_files, focal, others, p: Params):
    """
    Average CR across SINGER posterior ARG samples.

    Critical for smooth coalescence distributions (Supp Fig 16):
    single ARGs give degenerate point masses; posterior averaging
    over multiple topologies and branch lengths produces smooth CRs.
    """
    cr_sum   = None
    n_loaded = 0
    for f in trees_files:
        if not os.path.exists(f):
            continue
        ts_i = tskit.load(f)
        cr_i = compute_cr_for_haplotype(ts_i, focal, others, p)
        if cr_sum is None:
            cr_sum = np.zeros_like(cr_i)
        cr_sum += np.minimum(cr_i, 1e6)
        n_loaded += 1
    if n_loaded == 0:
        raise FileNotFoundError("No SINGER .trees files found")
    return cr_sum / n_loaded


# ================================================================
#  4.  WINDOW LABELS AND ROC
# ================================================================

def tracts_to_window_labels(tracts, seq_length, window_size, threshold=0.5):
    """Convert per-haplotype tracts → binary window labels."""
    nw = int(np.ceil(seq_length / window_size))
    labels = {}
    for nid, ivs in tracts.items():
        cov = np.zeros(nw)
        for l, r in ivs:
            w0 = int(l // window_size)
            w1 = min(int(np.ceil(r / window_size)), nw)
            for w in range(w0, w1):
                wl = w * window_size
                wr = min((w + 1) * window_size, seq_length)
                cov[w] += (min(r, wr) - max(l, wl)) / (wr - wl)
        labels[nid] = (cov >= threshold).astype(int)
    return labels


def aggregate_for_roc(labels_dict, scores_dict):
    """Pool per-haplotype labels and scores into flat arrays."""
    all_lab, all_scr = [], []
    for nid in labels_dict:
        if nid not in scores_dict:
            continue
        lab = labels_dict[nid]
        scr = scores_dict[nid]
        n = min(len(lab), len(scr))
        all_lab.append(lab[:n])
        all_scr.append(scr[:n])
    if not all_lab:
        return np.array([]), np.array([])
    return np.concatenate(all_lab), np.concatenate(all_scr)


def compute_roc(labels, scores):
    """Return fpr, tpr, auroc."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    labels, scores = labels[ok], scores[ok]
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return np.array([0, 1]), np.array([0, 1]), 0.5
    order = np.argsort(-scores)
    ls = labels[order]
    tp = np.cumsum(ls)
    fp = np.cumsum(1 - ls)
    tpr = np.concatenate([[0], tp / labels.sum()])
    fpr = np.concatenate([[0], fp / (len(labels) - labels.sum())])
    auroc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auroc


# ================================================================
#  5.  FILE OUTPUT
# ================================================================

def write_simulation_outputs(ts, p: Params, outdir: str):
    """
    Write all files needed for downstream analysis.

    Directory layout:
      {outdir}/
        sim.trees              full tree sequence (truth)
        afr_only.trees         AFR-only simplified (for SINGER comparison)
        singer_input.vcf       AFR-only VCF (SINGER input)
        modern.vcf             AFR-only VCF (IBDmix modern input)
        archaic.vcf            GHOST-only VCF (IBDmix archaic input)
        all.vcf                full VCF
        sample_map.json        node_id → {population, individual, ...}
        node_id_map.json       original_node_id → afr_simplified_node_id
        afr_samples.txt        AFR VCF sample names (for IBDmix)
        true_tracts.json       ground truth introgression tracts
    """
    os.makedirs(outdir, exist_ok=True)
    pop_map = get_pop_map(ts)

    # ---- Full tree sequence (with migration records for ground truth) ----
    ts.dump(os.path.join(outdir, "sim.trees"))

    # ---- Version without migrations (needed for simplify) ----
    tables = ts.dump_tables()
    tables.migrations.clear()
    ts_no_mig = tables.tree_sequence()

    # ---- Sample map ----
    sample_info = {}
    for pop_name, nodes in pop_map.items():
        for nid in nodes:
            ind = ts.node(nid).individual
            sample_info[nid] = {
                "population": pop_name,
                "node_id": nid,
                "individual": ind,
                "vcf_sample": f"tsk_{ind}",
                "haplotype_within_ind": nid - 2 * ind,
            }
    with open(os.path.join(outdir, "sample_map.json"), "w") as f:
        json.dump({str(k): v for k, v in sample_info.items()}, f, indent=2)

    # ---- AFR-only tree sequence and VCF (for SINGER) ----
    afr_nodes = sorted(pop_map["AFR"])
    ts_afr = ts_no_mig.simplify(samples=afr_nodes)
    ts_afr.dump(os.path.join(outdir, "afr_only.trees"))
    with open(os.path.join(outdir, "singer_input.vcf"), "w") as f:
        ts_afr.write_vcf(f)

    # Node ID mapping: original → AFR-simplified
    id_map = {}
    for new_id, old_id in enumerate(afr_nodes):
        id_map[old_id] = new_id
    with open(os.path.join(outdir, "node_id_map.json"), "w") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f, indent=2)

    # ---- Modern VCF for IBDmix (= AFR only) ----
    shutil.copy2(
        os.path.join(outdir, "singer_input.vcf"),
        os.path.join(outdir, "modern.vcf"))

    # ---- Archaic VCF for IBDmix (= GHOST only) ----
    ghost_nodes = pop_map["GHOST"]
    ts_ghost = ts_no_mig.simplify(samples=ghost_nodes)
    with open(os.path.join(outdir, "archaic.vcf"), "w") as f:
        ts_ghost.write_vcf(f)

    # ---- Full VCF ----
    with open(os.path.join(outdir, "all.vcf"), "w") as f:
        ts_no_mig.write_vcf(f)

    # ---- AFR sample names in the modern/singer VCF ----
    # In ts_afr (simplified), individuals are 0..n_afr_diploid-1
    # VCF sample names: tsk_0, tsk_1, ...
    with open(os.path.join(outdir, "afr_samples.txt"), "w") as f:
        for i in range(p.n_afr_diploid):
            f.write(f"tsk_{i}\n")

    # ---- Ground truth tracts ----
    tracts = extract_true_introgression(ts, p)
    tracts_json = {
        str(k): [(float(l), float(r)) for l, r in v]
        for k, v in tracts.items()
    }
    with open(os.path.join(outdir, "true_tracts.json"), "w") as f:
        json.dump(tracts_json, f)

    # ---- Summary ----
    afr_n = pop_map["AFR"]
    total_introg = sum(
        sum(r - l for l, r in tracts[n]) for n in afr_n)
    avg_frac = total_introg / (len(afr_n) * p.seq_length)
    n_tracts = sum(len(tracts[n]) for n in afr_n)
    avg_len  = total_introg / max(n_tracts, 1)

    print(f"  Tree sequence: {ts.num_sites:,} sites, "
          f"{ts.num_trees:,} trees, {ts.num_samples} samples")
    for name, nodes in pop_map.items():
        print(f"    {name}: {len(nodes)} haplotypes "
              f"(nodes {min(nodes)}–{max(nodes)})")
    print(f"  Ghost introgression into AFR: {avg_frac*100:.2f}% of genome")
    print(f"    {n_tracts} tracts across {len(afr_n)} haplotypes, "
          f"mean length {avg_len/1e3:.1f} kb")
    print(f"  Files → {outdir}/")

    return pop_map, tracts


# ================================================================
#  6.  IBDmix
# ================================================================

def run_ibdmix(outdir: str, p: Params):
    """
    Run IBDmix: detect segments in AFR that are IBD with GHOST.

    Pipeline:
      1) generate_gt: merge archaic + modern genotypes
      2) ibdmix: compute LOD scores, emit tracts above threshold
      3) filter by minimum tract length

    IBDmix output: ID  chrom  start  end  LOD  (tab-separated)
    """
    gt_file      = os.path.join(outdir, "ibdmix_gt.txt")
    raw_output   = os.path.join(outdir, "ibdmix_raw.txt")
    final_output = os.path.join(outdir, "ibdmix_tracts.txt")

    # Step 1: merge genotypes
    cmd_gt = [
        p.ibdmix_generate_gt,
        "--archaic", os.path.join(outdir, "archaic.vcf"),
        "--modern",  os.path.join(outdir, "modern.vcf"),
        "--output",  gt_file,
    ]
    print(f"  $ {' '.join(cmd_gt)}")
    r = subprocess.run(cmd_gt, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr}", file=sys.stderr)
        return None

    # Step 2: call tracts
    # All AFR samples are tested; there is no separate outgroup.
    # IBDmix uses allele frequencies within the modern panel itself.
    cmd_ibd = [
        p.ibdmix_bin,
        "--genotype", gt_file,
        "--output",   raw_output,
        "--LOD-threshold", str(p.ibdmix_lod_threshold),
    ]
    print(f"  $ {' '.join(cmd_ibd)}")
    r = subprocess.run(cmd_ibd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr}", file=sys.stderr)
        return None

    # Step 3: length filter
    n_raw, n_kept = 0, 0
    with open(raw_output) as fin, open(final_output, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5 or parts[0] == "ID":
                # IBDmix emits a header row "ID\tchrom\tstart\tend\tslod"
                # as the first line of ibdmix_raw.txt — skip it.
                continue
            n_raw += 1
            if int(parts[3]) - int(parts[2]) >= p.ibdmix_min_length:
                fout.write(line)
                n_kept += 1
    print(f"  {n_kept}/{n_raw} tracts ≥ {p.ibdmix_min_length/1e3:.0f} kb"
          f" → {final_output}")
    return final_output


def parse_ibdmix_output(filepath, p: Params):
    """
    Parse IBDmix output → per-original-node tracts and window scores.

    IBDmix reports per-diploid-individual.  We assign each tract to
    BOTH haplotypes of that individual (conservative; the paper takes
    the "larger ratio from two haplotypes" which is similar).

    Returns:
        tracts: dict orig_node_id → [(left, right), ...]
        scores: dict orig_node_id → np.array of per-window max LOD
    """
    outdir = os.path.dirname(filepath)
    with open(os.path.join(outdir, "node_id_map.json")) as f:
        id_map = json.load(f)  # str(orig_node) → simp_node
    rev_map = {v: int(k) for k, v in id_map.items()}

    nw = int(np.ceil(p.seq_length / p.window_size))
    tracts   = defaultdict(list)
    max_lod  = defaultdict(lambda: np.zeros(nw))

    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.startswith("ID") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            sample_name = parts[0]
            try:
                start, end = int(parts[2]), int(parts[3])
                lod        = float(parts[4])
            except ValueError:
                continue

            try:
                ind = int(sample_name.replace("tsk_", ""))
            except ValueError:
                continue

            # Both haplotypes of this individual
            for hap in (0, 1):
                simp_node = 2 * ind + hap
                if simp_node not in rev_map:
                    continue
                orig = rev_map[simp_node]
                tracts[orig].append((start, end))
                w0 = max(0, int(start // p.window_size))
                w1 = min(int(np.ceil(end / p.window_size)), nw)
                for w in range(w0, w1):
                    max_lod[orig][w] = max(max_lod[orig][w], lod)

    for nid in tracts:
        tracts[nid] = _merge_intervals(sorted(tracts[nid]))
    return dict(tracts), dict(max_lod)


# ================================================================
#  7.  SINGER
# ================================================================

def run_singer(outdir: str, p: Params):
    """
    Run SINGER on AFR-only data.

    SINGER sees only AFR haplotypes — the ghost archaic is unsampled.
    It infers the ARG and produces posterior samples as .trees files.
    """
    vcf_prefix = os.path.join(outdir, "singer_input")
    out_prefix = os.path.join(outdir, "singer_arg")
    ts_prefix  = os.path.join(outdir, "singer_ts")

    cmd = [
        p.singer_bin,
        "-m",      str(p.mu),
        "-Ne",     str(p.Ne_afr),
        "-ratio",  f"{p.rho / p.mu:.2f}",
        "-vcf",    vcf_prefix,
        "-output", out_prefix,
        "-start",  "0",
        "-end",    str(int(p.seq_length)),
        "-n",      str(p.singer_n_samples),
        "-thin",   str(p.singer_thin),
        "-polar",  "0.5",
    ]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  SINGER failed: {r.stderr}", file=sys.stderr)
        _print_manual_commands(outdir, p)
        return None

    cmd_conv = [
        p.convert_bin,
        "-input",  out_prefix,
        "-output", ts_prefix,
        "-start",  "0",
        "-end",    str(p.singer_n_samples - 1),
        "-step",   "1",
    ]
    print(f"  $ {' '.join(cmd_conv)}")
    r = subprocess.run(cmd_conv, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  convert_to_tskit failed: {r.stderr}", file=sys.stderr)
        return None

    found = sum(1 for i in range(p.singer_n_samples)
                if os.path.exists(f"{ts_prefix}_{i}.trees"))
    print(f"  {found}/{p.singer_n_samples} .trees files created")
    return ts_prefix


def _cr_one_posterior_sample(args):
    """
    Worker: load one posterior tree file, compute CR for all AFR
    haplotypes using tskit's C-level pair_coalescence_counts.

    Returns np.ndarray of shape (n_afr, n_windows).
    """
    tree_file, afr_nodes, t_split, t_introg, window_size = args
    ts = tskit.load(tree_file)
    n_afr = len(afr_nodes)

    time_windows = np.array([0.0, t_introg, t_split, np.inf])
    genome_windows = np.arange(0, ts.sequence_length + window_size,
                               window_size)
    n_gw = len(genome_windows) - 1
    cr_all = np.zeros((n_afr, n_gw))

    for i, focal in enumerate(afr_nodes):
        others = [n for n in afr_nodes if n != focal]
        counts = ts.pair_coalescence_counts(
            sample_sets=[[focal], others],
            time_windows=time_windows,
            windows=genome_windows,
            span_normalise=False,
        )
        # counts[:, 1] = pairs in [t_introg, t_split)  → n_within
        # counts[:, 2] = pairs in [t_split, inf)        → n_above
        n_within = counts[:, 1]
        n_above = counts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = n_above / n_within
        bad = n_within == 0
        cr[bad & (n_above > 0)] = 1e6
        cr[bad & (n_above == 0)] = 0.0
        cr_all[i] = np.minimum(cr, 1e6)

    return cr_all


def compute_cr_from_singer(outdir: str, p: Params):
    """
    Posterior-averaged CR from SINGER ARG samples.

    Parallelised across posterior samples: each worker loads one tree
    file and computes CR for ALL AFR haplotypes in a single pass.
    Results are averaged across posterior samples.

    Returns dict: original_node_id → cr_array
    """
    ts_prefix = os.path.join(outdir, "singer_ts")
    trees_files = [f"{ts_prefix}_{i}.trees"
                   for i in range(p.singer_n_samples)
                   if os.path.exists(f"{ts_prefix}_{i}.trees")]
    if not trees_files:
        print("  No SINGER .trees files found", file=sys.stderr)
        return None

    with open(os.path.join(outdir, "node_id_map.json")) as f:
        id_map = json.load(f)
    rev_map = {v: int(k) for k, v in id_map.items()}

    # AFR nodes in the simplified tree sequence (SINGER's space)
    with open(os.path.join(outdir, "sample_map.json")) as f:
        sample_info = json.load(f)
    afr_orig = sorted([int(nid) for nid, info in sample_info.items()
                       if info["population"] == "AFR"])
    afr_simp = sorted([id_map[str(n)] for n in afr_orig])

    n_files = len(trees_files)
    n_workers = min(4, n_files)
    print(f"  {n_files} posterior samples, "
          f"{len(afr_simp)} AFR haplotypes, {n_workers} workers")

    worker_args = [(f, afr_simp, p.t_split, p.t_introg, p.window_size)
                   for f in trees_files]

    cr_sum = None
    n_done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for cr_i in pool.imap_unordered(
                _cr_one_posterior_sample, worker_args):
            if cr_sum is None:
                cr_sum = np.zeros_like(cr_i)
            cr_sum += cr_i
            n_done += 1
            sys.stdout.write(
                f"\r  Posterior samples processed: {n_done}/{n_files}")
            sys.stdout.flush()
    print()

    cr_avg = cr_sum / n_files
    cr_result = {}
    for i, node_s in enumerate(afr_simp):
        cr_result[rev_map[node_s]] = cr_avg[i]
    return cr_result


def _print_manual_commands(outdir, p):
    """Print commands for manual execution."""
    print("\n" + "="*65)
    print("  Commands for manual execution:")
    print("="*65)
    print(f"""
# --- IBDmix ---
# Install: conda install -c bioconda ibdmix
generate_gt \\
    --archaic {outdir}/archaic.vcf \\
    --modern  {outdir}/modern.vcf \\
    --output  {outdir}/ibdmix_gt.txt

ibdmix \\
    --genotype {outdir}/ibdmix_gt.txt \\
    --output   {outdir}/ibdmix_raw.txt \\
    --LOD-threshold {p.ibdmix_lod_threshold}

# Filter by length (≥ {p.ibdmix_min_length/1e3:.0f} kb):
awk -F'\\t' '($4-$3) >= {p.ibdmix_min_length}' \\
    {outdir}/ibdmix_raw.txt > {outdir}/ibdmix_tracts.txt

# --- SINGER ---
# Download binary: https://github.com/popgenmethods/SINGER/releases
singer_master \\
    -m {p.mu} -Ne {p.Ne_afr} -ratio {p.rho/p.mu:.1f} \\
    -vcf {outdir}/singer_input \\
    -output {outdir}/singer_arg \\
    -start 0 -end {int(p.seq_length)} \\
    -n {p.singer_n_samples} -thin {p.singer_thin} -polar 0.5

convert_to_tskit \\
    -input {outdir}/singer_arg \\
    -output {outdir}/singer_ts \\
    -start 0 -end {p.singer_n_samples - 1} -step 1

# Then re-run: python {sys.argv[0]} --stage cr_singer --outdir {outdir}
""")


# ================================================================
#  8.  PLOTTING
# ================================================================

def plot_roc_curves(roc_dict, outfile):
    """Plot all ROC curves."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    styles = {
        "CR (true trees)":       {"color": "#d62728", "ls": "-",  "lw": 2.5},
        "CR (SINGER posterior)": {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
        "IBDmix":                {"color": "#2ca02c", "ls": "-",  "lw": 2.0},
        "CR-SINGER vs IBDmix":   {"color": "#9467bd", "ls": "--", "lw": 1.8},
        "BPP MSC-I":             {"color": "#ff7f0e", "ls": "-",  "lw": 2.0},
        "CR (Relate)":           {"color": "#8c564b", "ls": "-",  "lw": 2.0},
        "S*":                    {"color": "#e377c2", "ls": "-",  "lw": 2.0},
        "BPP MSC-I (no ghost)":  {"color": "#ff7f0e", "ls": "--", "lw": 2.0},
    }
    for name, (fpr, tpr, auroc) in roc_dict.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1.5})
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auroc:.3f})", **s)
    ax.plot([0, 1], [0, 1], "k:", lw=0.7, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Ghost Archaic Introgression Detection", fontsize=13)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"  Saved → {outfile}")


def plot_cr_heatmap(ts, focal_node, p: Params, tracts, outfile,
                    max_time_yrs=800_000):
    """
    Coalescence time distribution heatmap (Fig 6c).

    Each column = one 10 kb window.
    Y-axis = time bins (kya).
    Colour = density of pairwise TMRCAs of focal with all other AFR.
    White dashed lines = introgression window boundaries.
    Red shading = true introgressed tracts.
    """
    pop_map = get_pop_map(ts)
    afr_nodes = pop_map["AFR"]
    others = [n for n in afr_nodes if n != focal_node]

    W = p.window_size
    nw = int(np.ceil(ts.sequence_length / W))
    n_tbins = 100
    max_gens = max_time_yrs / p.gen_time
    tedges = np.linspace(0, max_gens, n_tbins + 1)
    heatmap = np.zeros((n_tbins, nw))

    for tree in ts.trees():
        tl, tr = tree.interval
        w0 = int(tl // W)
        w1 = min(int(np.ceil(tr / W)), nw)
        tmrcas = np.array([
            ts.node(tree.mrca(focal_node, o)).time
            if tree.mrca(focal_node, o) != tskit.NULL else np.inf
            for o in others
        ])
        for w in range(w0, w1):
            wl = w * W
            wr = min((w + 1) * W, ts.sequence_length)
            frac = (min(tr, wr) - max(tl, wl)) / W
            hist, _ = np.histogram(tmrcas, bins=tedges)
            heatmap[:, w] += frac * hist

    csums = heatmap.sum(axis=0, keepdims=True)
    csums[csums == 0] = 1
    heatmap /= csums

    fig, ax = plt.subplots(figsize=(14, 4))
    extent = [0, ts.sequence_length / 1e6, 0, max_time_yrs / 1e3]
    im = ax.imshow(heatmap, aspect="auto", origin="lower", extent=extent,
                   cmap="hot", interpolation="bilinear")
    ax.axhline(p.t_introg_yrs / 1e3, color="white", lw=1, ls="--", alpha=0.8)
    ax.axhline(p.t_split_yrs / 1e3,  color="white", lw=1, ls="--", alpha=0.8)
    if focal_node in tracts:
        for l, r in tracts[focal_node]:
            ax.axvspan(l / 1e6, r / 1e6, color="cyan", alpha=0.15)
    ax.set_xlabel("Genomic position (Mb)", fontsize=11)
    ax.set_ylabel("Coalescence time (kya)", fontsize=11)
    ax.set_title(f"Pairwise coalescence — AFR haplotype {focal_node}",
                 fontsize=12)
    plt.colorbar(im, ax=ax, label="Density", shrink=0.8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"  Saved heatmap → {outfile}")


# ================================================================
#  9.  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ghost archaic introgression detection: "
                    "simulation + ROC evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python %(prog)s --stage simulate            # simulate data
  python %(prog)s --stage cr_true             # CR on true trees (oracle)
  python %(prog)s --stage run_ibdmix          # needs IBDmix installed
  python %(prog)s --stage run_singer          # needs SINGER installed
  python %(prog)s --stage cr_singer           # after SINGER finishes
  python %(prog)s --stage roc                 # plot ROC curves
  python %(prog)s --stage all                 # run everything
        """)
    parser.add_argument("--outdir",     default="introg_sim")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--seq-length", type=float, default=None,
                        help="Sequence length in bp (default 50 Mb)")
    parser.add_argument("--stage",      default="all",
                        choices=["all", "simulate", "cr_true",
                                 "run_ibdmix", "run_singer",
                                 "cr_singer", "roc"])
    parser.add_argument("--singer-bin",    default=None)
    parser.add_argument("--convert-bin",   default=None)
    parser.add_argument("--ibdmix-bin",    default=None)
    parser.add_argument("--ibdmix-gt-bin", default=None)
    parser.add_argument("--n-samples",     type=int, default=None,
                        help="SINGER posterior samples")
    args = parser.parse_args()

    p = Params()
    if args.seq_length:    p.seq_length = int(args.seq_length)
    if args.singer_bin:    p.singer_bin = args.singer_bin
    if args.convert_bin:   p.convert_bin = args.convert_bin
    if args.ibdmix_bin:    p.ibdmix_bin = args.ibdmix_bin
    if args.ibdmix_gt_bin: p.ibdmix_generate_gt = args.ibdmix_gt_bin
    if args.n_samples:     p.singer_n_samples = args.n_samples

    outdir  = args.outdir
    run_all = args.stage == "all"

    # ────────────── SIMULATE ──────────────
    if run_all or args.stage == "simulate":
        print("\n" + "="*60)
        print("  STAGE: simulate")
        print("="*60)
        ts = simulate(p, seed=args.seed)
        write_simulation_outputs(ts, p, outdir)

    # ────────────── CR ON TRUE TREES ──────────────
    if run_all or args.stage == "cr_true":
        print("\n" + "="*60)
        print("  STAGE: cr_true  (oracle upper bound)")
        print("="*60)
        ts = tskit.load(os.path.join(outdir, "sim.trees"))
        pop_map = get_pop_map(ts)
        with open(os.path.join(outdir, "true_tracts.json")) as f:
            tracts = {int(k): v for k, v in json.load(f).items()}

        cr_true = compute_cr_all_afr(ts, p, pop_map)
        np.savez(os.path.join(outdir, "cr_true.npz"),
                 **{str(k): v for k, v in cr_true.items()})

        labels = tracts_to_window_labels(
            tracts, p.seq_length, p.window_size, p.introg_label_threshold)
        al, asc = aggregate_for_roc(labels, cr_true)
        fpr, tpr, auroc = compute_roc(al, asc)
        print(f"  AUROC = {auroc:.4f}  "
              f"({int(al.sum()):,} pos / {len(al):,} windows)")
        np.savez(os.path.join(outdir, "roc_cr_true.npz"),
                 fpr=fpr, tpr=tpr, auroc=auroc)

        # Heatmap for first AFR haplotype
        afr0 = pop_map["AFR"][0]
        plot_cr_heatmap(ts, afr0, p, tracts,
                        os.path.join(outdir, "heatmap_true.png"))

    # ────────────── IBDmix ──────────────
    if run_all or args.stage == "run_ibdmix":
        print("\n" + "="*60)
        print("  STAGE: run_ibdmix")
        print("="*60)
        if shutil.which(p.ibdmix_generate_gt) is None:
            print(f"  '{p.ibdmix_generate_gt}' not on PATH.")
            _print_manual_commands(outdir, p)
        else:
            out = run_ibdmix(outdir, p)
            if out:
                ibdmix_tracts, ibdmix_scores = parse_ibdmix_output(out, p)

                # ROC: IBDmix vs simulation truth
                with open(os.path.join(outdir, "true_tracts.json")) as f:
                    true_t = {int(k): v for k, v in json.load(f).items()}
                true_lab = tracts_to_window_labels(
                    true_t, p.seq_length, p.window_size,
                    p.introg_label_threshold)
                al, asc = aggregate_for_roc(true_lab, ibdmix_scores)
                fpr, tpr, auroc = compute_roc(al, asc)
                print(f"  AUROC (IBDmix vs truth) = {auroc:.4f}")
                np.savez(os.path.join(outdir, "roc_ibdmix.npz"),
                         fpr=fpr, tpr=tpr, auroc=auroc)

                # Save IBDmix labels for the proxy ROC later
                ibdmix_lab = tracts_to_window_labels(
                    ibdmix_tracts, p.seq_length, p.window_size,
                    p.introg_label_threshold)
                np.savez(os.path.join(outdir, "ibdmix_labels.npz"),
                         **{str(k): v for k, v in ibdmix_lab.items()})

    # ────────────── SINGER ──────────────
    if run_all or args.stage == "run_singer":
        print("\n" + "="*60)
        print("  STAGE: run_singer")
        print("="*60)
        if shutil.which(p.singer_bin) is None:
            print(f"  '{p.singer_bin}' not on PATH.")
            _print_manual_commands(outdir, p)
        else:
            run_singer(outdir, p)

    # ────────────── CR FROM SINGER POSTERIOR ──────────────
    if run_all or args.stage == "cr_singer":
        print("\n" + "="*60)
        print("  STAGE: cr_singer")
        print("="*60)
        ts_prefix = os.path.join(outdir, "singer_ts")
        found = sum(1 for i in range(p.singer_n_samples)
                    if os.path.exists(f"{ts_prefix}_{i}.trees"))
        if found == 0:
            print("  No .trees files found. Run --stage run_singer first.")
            _print_manual_commands(outdir, p)
        else:
            cr_singer = compute_cr_from_singer(outdir, p)
            if cr_singer:
                np.savez(os.path.join(outdir, "cr_singer.npz"),
                         **{str(k): v for k, v in cr_singer.items()})

                # ROC: CR-SINGER vs truth
                with open(os.path.join(outdir, "true_tracts.json")) as f:
                    true_t = {int(k): v for k, v in json.load(f).items()}
                true_lab = tracts_to_window_labels(
                    true_t, p.seq_length, p.window_size,
                    p.introg_label_threshold)
                al, asc = aggregate_for_roc(true_lab, cr_singer)
                fpr, tpr, auroc = compute_roc(al, asc)
                print(f"  AUROC (CR-SINGER vs truth) = {auroc:.4f}")
                np.savez(os.path.join(outdir, "roc_cr_singer.npz"),
                         fpr=fpr, tpr=tpr, auroc=auroc)

                # Proxy ROC: CR-SINGER vs IBDmix labels (paper method)
                ibd_f = os.path.join(outdir, "ibdmix_labels.npz")
                if os.path.exists(ibd_f):
                    d = np.load(ibd_f)
                    ibdmix_lab = {int(k): d[k] for k in d.files}
                    al, asc = aggregate_for_roc(ibdmix_lab, cr_singer)
                    fpr, tpr, auroc = compute_roc(al, asc)
                    print(f"  AUROC (CR-SINGER vs IBDmix) = {auroc:.4f}")
                    np.savez(os.path.join(
                        outdir, "roc_cr_singer_vs_ibdmix.npz"),
                        fpr=fpr, tpr=tpr, auroc=auroc)

    # ────────────── PLOT ──────────────
    if run_all or args.stage == "roc":
        print("\n" + "="*60)
        print("  STAGE: roc")
        print("="*60)
        roc_dict = {}
        for name, fname in [
            ("CR (true trees)",       "roc_cr_true.npz"),
            ("CR (SINGER posterior)", "roc_cr_singer.npz"),
            ("IBDmix",                "roc_ibdmix.npz"),
            ("CR-SINGER vs IBDmix",   "roc_cr_singer_vs_ibdmix.npz"),
        ]:
            path = os.path.join(outdir, fname)
            if os.path.exists(path):
                d = np.load(path)
                roc_dict[name] = (d["fpr"], d["tpr"], float(d["auroc"]))

        if roc_dict:
            plot_roc_curves(roc_dict,
                            os.path.join(outdir, "roc_comparison.png"))
            print()
            print("  ┌─────────────────────────────┬─────────┐")
            print("  │ Method                      │  AUROC  │")
            print("  ├─────────────────────────────┼─────────┤")
            for nm, (_, _, a) in roc_dict.items():
                print(f"  │ {nm:<27s} │ {a:.4f}  │")
            print("  └─────────────────────────────┴─────────┘")
        else:
            print("  No ROC data found. Run earlier stages first.")

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
