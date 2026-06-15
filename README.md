# introgressionROCstudy

Scripts to reproduce the seven-method ROC comparison of introgression-detection
methods (CR on true trees, BPP MSC-I, CR from SINGER, CR from Relate, IBDmix,
S\*) against an exact simulated ground truth for ghost archaic introgression.

The pipeline simulates a two-population demographic model (modern AFR receiving
a 3 % pulse from an unsampled "ghost" archaic at 60 kya, with the two populations
diverging at 500 kya) with `msprime`, extracts the true introgression intervals
from the migration records, runs each detection method on the simulated VCFs,
and produces a combined ROC plot with the AUROC for each method.

Running the pipeline twice with different seeds (sim\_A: 42, sim\_B: 43) yields
two independent simulations and the cross-sim AUROC table that is the central
empirical claim of the study: every method replicates within 0.02 AUROC across
sims, and the method ranking is preserved.

## What you get out

Two PDF figures, one per sim, with the same seven-method ROC overlay:

```
sim_A/roc_comparison_all.pdf      CR true > BPP > CR SINGER > CR Relate > IBDmix > S*
sim_B/roc_comparison_all.pdf      same ranking, AUROCs within 0.02 of sim_A
```

Expected AUROCs and per-method commentary live in [`expected_AUROCs.md`](expected_AUROCs.md).

## Quick start

1. **Install external tools** (BPP dev, SINGER v0.1.8, Relate `b54ede2`,
   IBDmix v1.0.1) — see [`INSTALL.md`](INSTALL.md).
2. **Make a Python venv** and install [`requirements.txt`](requirements.txt):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. **Reproduce sim\_A and sim\_B** — see the step-by-step
   recipe in [`REPRODUCE.md`](REPRODUCE.md).

## Pipeline overview

| Script | Role |
|---|---|
| `introgression_roc_pipeline.py` | core: msprime simulation, ground-truth extraction, CR on true trees, IBDmix, CR on SINGER posterior, ROC utilities |
| `msprime_to_bpp.py` | converts a tree-sequence simulation into BPP MSC-I phylip + control file |
| `run_bpp_roc.py` | BPP MSC-I inference + per-chain ROC (supports `--chains r1 r2` for between-chain convergence overlay) |
| `run_relate_roc.py` | Relate inference + CR computation + ROC |
| `run_sstar_roc.py` | S\* inference + ROC |
| `singer_orchestrator.sh` | chunked SINGER orchestrator: 50 × 1 Mb segments in parallel + `convert_long_ARG.py` merge into 100 genome-wide tree sequences |
| `plot_combined_roc.py` | reads every `roc_*.npz` from a simdir and writes a single combined ROC figure |

## Demographic model and parameters

| Parameter | Value | Source |
|---|---|---|
| Mutation rate μ | 1.2 × 10⁻⁸ | Deng et al. 2025, supp. D.1 |
| Recombination rate r | 1.2 × 10⁻⁸ (r/μ = 1.0) | same |
| Ne (AFR ancestral) | 20 000 | same |
| AFR–GHOST split τ | 500 000 years (Skoglund & Mathieson 2018) | same |
| Introgression time | 60 000 years | same |
| Introgression fraction | 3 % | model setting |
| Sequence length | 50 Mb | this study |
| n_AFR haplotypes | 100 | this study |
| n_GHOST haplotypes | 2 (IBDmix reference) | this study |
| Window size | 10 kb | this study |

## License

GPL v3 — see [`LICENSE`](LICENSE).

## Citation

If you use this pipeline, please cite the accompanying manuscript (in
preparation) and the underlying methods: msprime
([Baumdicker et al. 2022](https://doi.org/10.1093/genetics/iyab229)),
SINGER ([Deng et al. 2025](https://doi.org/10.1038/s41588-025-02317-9)),
Relate ([Speidel et al. 2019](https://doi.org/10.1038/s41588-019-0484-x)),
BPP ([Flouri et al. 2020](https://doi.org/10.1093/molbev/msz296)),
IBDmix ([Chen et al. 2020](https://doi.org/10.1016/j.cell.2020.01.012)),
and S\* ([Wall 2000](https://doi.org/10.1093/genetics/154.4.1271)).
