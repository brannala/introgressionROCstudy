
#!/usr/bin/env python3
"""
Simulate ghost archaic introgression with msprime and generate BPP input files.

Demography (backward in time):
    ANC ──┬── t_split (500 kya) ──── GHOST  (archaic, unsampled)
          │                            │
          │              introgression pulse (60 kya, 3%)
          │                            ↓
          └── t_split (500 kya) ──── AFR    (sampled Africans)

At t_introg, 3% of AFR lineages are moved to GHOST (backward-time convention,
same as ms -em).  Forward in time this is GHOST → AFR gene flow.

Outputs:
    outdir/sim.trees       — full tree sequence (with migration records)
    outdir/sequences.phy   — multi-locus phylip for BPP
    outdir/imap.txt        — sample-to-species mapping
    outdir/infer.ctl       — BPP control file (MSC-I model)
    outdir/sample_ids.json — BPP sample name → original node ID

Dependencies:
    pip install msprime tskit numpy

Bruce Rannala & Ziheng Yang, April 2026
"""

import argparse
import json
import os
import sys
import numpy as np
import msprime
import tskit
from collections import defaultdict


# ================================================================
#  PARAMETERS
# ================================================================

class Params:
    """Simulation and analysis parameters (defaults match SINGER Fig 6a)."""

    # Demographic model
    gen_time        = 28          # years per generation
    Ne_afr          = 20_000      # African effective population size
    Ne_ghost        = 10_000      # ghost archaic Ne
    Ne_anc          = 20_000      # ancestral Ne
    t_split_yrs     = 500_000     # AFR-GHOST divergence (years)
    t_introg_yrs    = 60_000      # introgression pulse (years)
    introg_frac     = 0.03        # proportion of AFR lineages from GHOST

    # Genome
    seq_length      = 50_000_000  # total simulated length (bp)
    mu              = 1.2e-8      # per-bp per-gen mutation rate
    rho             = 1.2e-8      # per-bp per-gen recombination rate

    # Samples
    n_afr_diploid   = 50          # diploid AFR individuals (100 haplotypes)
    n_ghost_diploid = 1           # diploid GHOST individuals (2 haplotypes)

    # BPP loci
    window_size     = 10_000      # locus length (bp)

    @property
    def t_split(self):
        return self.t_split_yrs / self.gen_time

    @property
    def t_introg(self):
        return self.t_introg_yrs / self.gen_time


# ================================================================
#  1.  MSPRIME SIMULATION
# ================================================================

def build_demography(p):
    """
    Two-population model with ghost archaic introgression into Africans.

    Backward in time: at t_introg, fraction introg_frac of AFR lineages
    jump to GHOST.  At t_split, GHOST and AFR merge into ANC.
    """
    dem = msprime.Demography()
    dem.add_population(name="AFR",   initial_size=p.Ne_afr)
    dem.add_population(name="GHOST", initial_size=p.Ne_ghost)
    dem.add_population(name="ANC",   initial_size=p.Ne_anc)

    # Introgression: backward, AFR lineages -> GHOST at t_introg
    # (forward: GHOST -> AFR gene flow)
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


def simulate(p, seed=42):
    """Run coalescent simulation with mutations.  Returns a tree sequence."""
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


# ================================================================
#  2.  PARTITION INTO BPP LOCI
# ================================================================

def _canonicalize_jc69(pat):
    """Canonicalize a site pattern under JC69 base exchangeability.

    Relabel so the first unique base becomes A, the second C, the third G,
    the fourth T.  E.g. TTGTTT -> AACAAA, CCGCCA -> AACAAC.
    """
    mapping = {}
    canonical = 'ACGT'
    next_idx = 0
    result = []
    for base in pat:
        if base not in mapping:
            mapping[base] = canonical[next_idx]
            next_idx += 1
        result.append(mapping[base])
    return ''.join(result)


def partition_to_bpp(ts, outdir, p, n_afr_haplotypes, nloci,
                     no_ghost=False, threads=1, seed=123,
                     window_sizes=None, site_patterns=False):
    """
    Chop a simulated tree sequence into fixed-size or variable-size loci
    and write BPP files.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Full simulation (from simulate()).
    outdir : str
        Output directory for BPP files.
    p : Params
        Simulation parameters.
    n_afr_haplotypes : int
        Number of AFR haplotypes to include (must be <= 2*n_afr_diploid).
    nloci : int
        Maximum number of loci to extract (ignored when window_sizes given).
    no_ghost : bool
        If True, exclude GHOST sequences (BPP runs without outgroup).
    threads : int
        BPP thread count for control file.
    seed : int
        Random seed for BPP control file.
    window_sizes : list of int, optional
        Per-locus window sizes in bp.  If None, use fixed p.window_size.
    site_patterns : bool
        If True, write compressed site-pattern format (P flag) instead of
        full sequences.
    """

    # Select sample nodes
    afr_nodes = list(range(n_afr_haplotypes))
    if no_ghost:
        keep_nodes = afr_nodes
    else:
        ghost_nodes = [2 * p.n_afr_diploid, 2 * p.n_afr_diploid + 1]
        keep_nodes = afr_nodes + ghost_nodes
    n_samples = len(keep_nodes)
    print(f"  Subsampling: {n_afr_haplotypes} AFR"
          f"{'' if no_ghost else ' + 2 GHOST'}"
          f" = {n_samples} haplotypes")

    # BPP sample names
    sample_ids = {}
    bpp_names = []
    for i, nid in enumerate(afr_nodes):
        name = f"afr{i+1}"
        sample_ids[name] = nid
        bpp_names.append(f"AFR^{name}")
    if not no_ghost:
        for i, nid in enumerate(ghost_nodes):
            name = f"ghost{i+1}"
            sample_ids[name] = nid
            bpp_names.append(f"GHOST^{name}")

    # Simplify (must clear migration records first)
    tables = ts.dump_tables()
    tables.migrations.clear()
    ts_sub = tables.tree_sequence().simplify(samples=keep_nodes)
    print(f"  Simplified: {ts_sub.num_samples} samples, "
          f"{ts_sub.num_sites:,} sites")

    # Build per-locus sizes and cumulative start positions
    if window_sizes is not None:
        nloci = len(window_sizes)
        cum_starts = np.concatenate([[0], np.cumsum(window_sizes)])
        total_bp = int(cum_starts[-1])
        if total_bp > ts_sub.sequence_length:
            raise ValueError(
                f"Window sizes sum to {total_bp} bp but sequence length "
                f"is {int(ts_sub.sequence_length)} bp")
        print(f"  Windows: {nloci} variable-size loci "
              f"(min {min(window_sizes)}, max {max(window_sizes)}, "
              f"mean {np.mean(window_sizes):.0f} bp, "
              f"total {total_bp:,} bp)")
    else:
        window_size = p.window_size
        max_loci = int(ts_sub.sequence_length // window_size)
        nloci = min(nloci, max_loci)
        window_sizes = [window_size] * nloci
        cum_starts = np.arange(nloci + 1) * window_size
        print(f"  Windows: {nloci} x {window_size} bp")

    # Collect variants by window
    print("  Extracting variant sites...")
    window_variants = defaultdict(list)
    for var in ts_sub.variants():
        pos = int(var.site.position)
        if pos >= cum_starts[-1]:
            break
        w = int(np.searchsorted(cum_starts, pos, side='right')) - 1
        pos_in_window = pos - int(cum_starts[w])
        window_variants[w].append(
            (pos_in_window, list(var.alleles), var.genotypes.copy()))

    total_vars = sum(len(v) for v in window_variants.values())
    n_with_vars = len(window_variants)
    print(f"  {total_vars:,} variant sites across {n_with_vars} windows "
          f"(mean {total_vars / nloci:.1f}/window)")

    # Write output files
    os.makedirs(outdir, exist_ok=True)
    max_name_len = max(len(n) for n in bpp_names)
    name_width = max_name_len + 2

    # --- sequences.phy ---
    seq_path = os.path.join(outdir, "sequences.phy")
    fmt_label = "site-pattern" if site_patterns else "full-sequence"
    print(f"  Writing {seq_path} ({fmt_label} format) ...")
    with open(seq_path, "w") as f:
        for w in range(nloci):
            if w > 0:
                f.write("\n")
            ws = window_sizes[w]
            variants = window_variants.get(w, [])
            if site_patterns:
                # Collect unique site patterns under JC69 equivalence.
                # Under JC69 all bases are exchangeable, so only the
                # partition of samples matters, not the base labels.
                patterns = {}   # canonical pattern -> count
                pattern_list = []  # insertion-ordered unique patterns

                # Invariant sites (all same base -> canonical all-A)
                invariant_count = ws - len(variants)
                invariant_pat = 'A' * n_samples
                if invariant_count > 0:
                    patterns[invariant_pat] = invariant_count
                    pattern_list.append(invariant_pat)

                # Variant sites
                for _, alleles, genos in variants:
                    col = []
                    for s in range(n_samples):
                        allele = alleles[genos[s]]
                        col.append(allele[0] if allele else 'A')
                    pat = _canonicalize_jc69(''.join(col))
                    if pat in patterns:
                        patterns[pat] += 1
                    else:
                        patterns[pat] = 1
                        pattern_list.append(pat)

                n_patterns = len(pattern_list)
                counts = [patterns[p] for p in pattern_list]

                f.write(f"{n_samples} {n_patterns} P\n\n")
                for s in range(n_samples):
                    pat_seq = ''.join(p[s] for p in pattern_list)
                    f.write(f"{bpp_names[s]:<{name_width}s}{pat_seq}\n")
                f.write(' '.join(str(c) for c in counts) + '\n')
            else:
                f.write(f"{n_samples} {ws}\n\n")
                for s in range(n_samples):
                    seq = bytearray(b'A' * ws)
                    for pos_in_window, alleles, genos in variants:
                        allele = alleles[genos[s]]
                        if allele:
                            seq[pos_in_window] = ord(allele[0])
                    f.write(f"{bpp_names[s]:<{name_width}s}{seq.decode()}\n")
            if (w + 1) % 1000 == 0:
                sys.stdout.write(f"\r    {w + 1}/{nloci} loci written")
                sys.stdout.flush()
    if nloci >= 1000:
        print()

    # --- imap.txt ---
    imap_path = os.path.join(outdir, "imap.txt")
    with open(imap_path, "w") as f:
        for i in range(n_afr_haplotypes):
            f.write(f"afr{i+1}\tAFR\n")
        if not no_ghost:
            f.write("ghost1\tGHOST\n")
            f.write("ghost2\tGHOST\n")

    # --- infer.ctl ---
    n_ghost_seqs = 0 if no_ghost else 2
    ctl_path = os.path.join(outdir, "infer.ctl")
    ctl_text = (
        f"seed = {seed}\n"
        f"seqfile = sequences.phy\n"
        f"Imapfile = imap.txt\n"
        f"jobname = bpp_introg\n"
        f"\n"
        f"speciesdelimitation = 0\n"
        f"speciestree = 0\n"
        f"\n"
        f"species&tree = 2 AFR GHOST\n"
        f"                 {n_afr_haplotypes}  {n_ghost_seqs}\n"
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
        f"burnin = 4000\n"
        f"sampfreq = 2\n"
        f"nsample = 10000\n"
        f"threads = {threads}\n"
        f"ancestry = 1\n"
    )
    with open(ctl_path, "w") as f:
        f.write(ctl_text)

    # --- sample_ids.json ---
    ids_path = os.path.join(outdir, "sample_ids.json")
    with open(ids_path, "w") as f:
        json.dump(sample_ids, f, indent=2)

    seq_size = os.path.getsize(seq_path) / 1e6
    total_bp = int(cum_starts[-1])
    print(f"\n  Output in {outdir}/:")
    print(f"    sequences.phy   ({seq_size:.1f} MB, {nloci} loci, "
          f"{total_bp:,} bp total)")
    print(f"    imap.txt        ({n_samples} samples)")
    print(f"    infer.ctl       (threads={threads})")
    print(f"    sample_ids.json")
    return nloci


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulate ghost introgression with msprime and "
                    "generate BPP input files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    defaults = Params()
    parser.add_argument("--outdir", default="bpp_sim",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for msprime simulation")
    parser.add_argument("--seq-length", type=float,
                        default=defaults.seq_length,
                        help="Total sequence length (bp)")
    parser.add_argument("--n-afr", type=int,
                        default=2 * defaults.n_afr_diploid,
                        help="Number of AFR haplotypes")
    parser.add_argument("--n-ghost-diploid", type=int,
                        default=defaults.n_ghost_diploid,
                        help="Number of GHOST diploid individuals")
    parser.add_argument("--nloci", type=int, default=5000,
                        help="Maximum number of BPP loci to extract")
    parser.add_argument("--window-size", type=int,
                        default=defaults.window_size,
                        help="Locus length in bp (fixed-size mode)")
    parser.add_argument("--window-file", type=str, default=None,
                        help="File with one window size (bp) per line; "
                             "overrides --window-size and --nloci")
    parser.add_argument("--window-loop", type=str, default=None,
                        help="File with list of block sizes; generate "
                             "separate BPP output for each (outdir/w<size>/)")
    parser.add_argument("--nloci-file", type=str, default=None,
                        help="File with per-block-size nloci (one per line, "
                             "matching --window-loop); overrides --nloci")
    parser.add_argument("--site-patterns", action="store_true",
                        help="Write site-pattern format (P flag) instead "
                             "of full sequences")
    parser.add_argument("--no-ghost", action="store_true",
                        help="Exclude GHOST sequences from BPP input")
    parser.add_argument("--introg-frac", type=float,
                        default=defaults.introg_frac,

                        help="Introgression fraction (GHOST -> AFR)")
    parser.add_argument("--threads", type=int, default=1,
                        help="BPP threads (written to control file)")
    parser.add_argument("--no-sim", action="store_true",
                        help="Skip simulation, load existing sim.trees")

    args = parser.parse_args()

    p = Params()
    p.seq_length = int(args.seq_length)
    p.n_afr_diploid = args.n_afr // 2
    p.n_ghost_diploid = args.n_ghost_diploid
    p.introg_frac = args.introg_frac
    p.window_size = args.window_size

    os.makedirs(args.outdir, exist_ok=True)
    trees_path = os.path.join(args.outdir, "sim.trees")

    # Step 1: Simulate
    if args.no_sim:
        print(f"Loading existing {trees_path}")
        ts = tskit.load(trees_path)
    else:
        print(f"Simulating: {p.seq_length/1e6:.0f} Mb, "
              f"{2*p.n_afr_diploid} AFR + {2*p.n_ghost_diploid} GHOST "
              f"haplotypes, seed={args.seed}")
        print(f"  introg_frac={p.introg_frac}, "
              f"t_introg={p.t_introg_yrs/1000:.0f} kya, "
              f"t_split={p.t_split_yrs/1000:.0f} kya")
        ts = simulate(p, seed=args.seed)
        ts.dump(trees_path)
        print(f"  {ts.num_trees:,} trees, {ts.num_sites:,} sites, "
              f"{ts.num_mutations:,} mutations")
        print(f"  Saved: {trees_path}")

    # Step 2: Partition into BPP loci
    if args.window_loop:
        # --- Window-loop mode: one BPP run per block size ---
        with open(args.window_loop) as f:
            block_sizes = [int(x) for x in f.read().split()]
        print(f"Window-loop: {len(block_sizes)} block sizes from "
              f"{args.window_loop}: {block_sizes}")

        if args.nloci_file:
            with open(args.nloci_file) as f:
                nloci_list = [int(x) for x in f.read().split()]
            if len(nloci_list) != len(block_sizes):
                raise ValueError(
                    f"--nloci-file has {len(nloci_list)} entries but "
                    f"--window-loop has {len(block_sizes)} block sizes")
        else:
            nloci_list = [args.nloci] * len(block_sizes)

        for i, (bsize, nl) in enumerate(zip(block_sizes, nloci_list)):
            subdir = os.path.join(args.outdir, f"w{bsize}")
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(block_sizes)}] block size = {bsize} bp, "
                  f"nloci = {nl}, output = {subdir}/")
            print(f"{'='*60}")
            p.window_size = bsize
            partition_to_bpp(
                ts, subdir, p,
                n_afr_haplotypes=args.n_afr,
                nloci=nl,
                no_ghost=args.no_ghost,
                threads=args.threads,
                seed=args.seed,
                site_patterns=args.site_patterns,
            )

        print(f"\nDone. Generated {len(block_sizes)} BPP runs in "
              f"{args.outdir}/w*/")
    else:
        # --- Single-run mode ---
        # Parse variable window sizes if provided
        window_sizes = None
        if args.window_file:
            with open(args.window_file) as f:
                window_sizes = [int(x) for x in f.read().split()]
            print(f"Read {len(window_sizes)} window sizes from "
                  f"{args.window_file}")

        print(f"\nPartitioning into BPP loci...")
        partition_to_bpp(
            ts, args.outdir, p,
            n_afr_haplotypes=args.n_afr,
            nloci=args.nloci,
            no_ghost=args.no_ghost,
            threads=args.threads,
            seed=args.seed,
            window_sizes=window_sizes,
            site_patterns=args.site_patterns,
        )

        print(f"\nDone. To run BPP:\n  bpp --cfile {args.outdir}/infer.ctl")


if __name__ == "__main__":
    main()



