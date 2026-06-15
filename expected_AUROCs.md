# Expected AUROCs

These are the AUROC values produced by the pipeline as committed at the
HEAD of this repo, on the two reference simulations (sim\_A: msprime seed 42,
sim\_B: msprime seed 43). Tool versions are pinned per
[`INSTALL.md`](INSTALL.md).

| Method | sim\_A | sim\_B | Δ |
|---|---|---|---|
| CR (true trees) — oracle | 0.977 | 0.975 | −0.002 |
| BPP MSC-I r₁ | 0.9411 | 0.9329 | −0.008 |
| BPP MSC-I r₂ | 0.9412 | 0.9314 | −0.010 |
| CR (SINGER posterior, 100 ARGs) | 0.890 | 0.881 | −0.009 |
| CR (Relate) | 0.821 | 0.803 | −0.018 |
| IBDmix LOD | 0.732 | 0.734 | +0.002 |
| S\* | 0.646 | 0.639 | −0.006 |

Every method replicates within 0.02 AUROC across the two independent
simulations. The method ranking is preserved: CR (true) > BPP > CR (SINGER)
> CR (Relate) > IBDmix > S\*.

## What's reproducible vs. what's stochastic

- **Exact reproducibility** (bit-for-bit): the msprime simulation, the
  ground-truth extraction from migration records, and CR on the true trees
  (the oracle). Given the same `--seed`, you get identical `sim.trees`,
  identical `true_tracts.json`, identical `roc_cr_true.npz`.
- **Stochastic with bounded variance** (within ±0.003 typically): SINGER
  posterior sampling. Each `singer_master` call uses internal randomness;
  running the chunked orchestrator twice at the same seed produces
  slightly different posterior ARGs and hence slightly different CR-SINGER
  AUROCs.
- **Stochastic per-chain, very tight between chains at convergence**
  (±0.0015 in our runs): BPP MSC-I. The MCMC converges on the introgression
  posterior much more reliably than on individual parameters like τ_R,
  which lets per-chain AUROC agree to ~4 decimal places even when
  parameter-level Gelman–Rubin diagnostics flag mixing concerns.
- **Deterministic given the SINGER and Relate output**: IBDmix (greedy LOD
  search) and S\* (windowed statistic). Re-running these on the same VCFs
  produces identical scores and identical AUROCs.

## Sanity-check snippet

```python
import numpy as np
for sim in ["sim_A", "sim_B"]:
    for name, fname in [
        ("CR true",         "roc_cr_true.npz"),
        ("BPP r1",          "roc_bpp_r1.npz"),
        ("BPP r2",          "roc_bpp_r2.npz"),
        ("CR SINGER",       "roc_cr_singer.npz"),
        ("CR Relate",       "roc_cr_relate.npz"),
        ("IBDmix",          "roc_ibdmix.npz"),
        ("S*",              "roc_sstar.npz"),
    ]:
        d = np.load(f"{sim}/{fname}")
        print(f"{sim:<6}  {name:<11}  AUROC = {float(d['auroc']):.4f}")
```
