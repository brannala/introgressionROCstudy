# Reproducing the ROC figures

This recipe produces the two figures (`sim_A/roc_comparison_all.pdf` and
`sim_B/roc_comparison_all.pdf`) and the cross-sim AUROC table from the
accompanying manuscript. The only inputs are the two msprime seeds — every
downstream method is deterministic given those seeds (BPP MCMC is seeded
explicitly per chain).

**Wall-clock note.** Most stages are minutes to a few hours; BPP MSC-I is
9–10 days **per chain** on the full 100-haplotype + 2-ghost dataset at the
documented MCMC settings (burn-in = 16 000, sampfreq = 2, nsample = 40 000).
Plan two chains × two sims = 4 BPP runs. Other stages can run in parallel
on a separate machine while BPP cranks.

```text
sim_A : msprime seed = 42
sim_B : msprime seed = 43
BPP   : chain seeds r1 = 11111, r2 = 22222
```

## Reproduce a single sim (procedure for sim\_A)

Replace `SIM=sim_A SEED=42` with `SIM=sim_B SEED=43` to do sim\_B.

```bash
SIM=sim_A
SEED=42
PY=.venv/bin/python
```

### 1. Simulate + ground truth + CR on true trees

```bash
$PY introgression_roc_pipeline.py --stage simulate --outdir $SIM --seed $SEED
$PY introgression_roc_pipeline.py --stage cr_true  --outdir $SIM
```

Outputs: `$SIM/sim.trees`, `$SIM/true_tracts.json`, `$SIM/cr_true.npz`,
`$SIM/roc_cr_true.npz`, the haplotype VCFs (`$SIM/{all,modern,archaic,singer_input}.vcf`),
and a node-id mapping (`$SIM/node_id_map.json`) used by every downstream
method.

### 2. IBDmix

```bash
$PY introgression_roc_pipeline.py --stage run_ibdmix --outdir $SIM
```

Outputs: `$SIM/ibdmix_raw.txt`, `$SIM/ibdmix_tracts.txt`, `$SIM/roc_ibdmix.npz`.

### 3. SINGER (chunked) + CR-SINGER

The pipeline's built-in `run_singer` runs SINGER on the whole 50 Mb at once,
which is far slower than the chunked workflow we use in the manuscript.
Use `singer_orchestrator.sh` instead:

```bash
bash singer_orchestrator.sh $SIM    # ~24h on 50 cores; produces 100 .trees
$PY introgression_roc_pipeline.py --stage cr_singer --outdir $SIM
```

Outputs: 100 × `$SIM/singer_arg_<i>.trees`, `$SIM/cr_singer.npz`,
`$SIM/roc_cr_singer.npz`.

### 4. Relate

```bash
$PY run_relate_roc.py --sim-dir $SIM --out-dir $SIM/relate_output --stage prepare
$PY run_relate_roc.py --sim-dir $SIM --out-dir $SIM/relate_output --stage run_relate
$PY run_relate_roc.py --sim-dir $SIM --out-dir $SIM/relate_output --stage cr
$PY run_relate_roc.py --sim-dir $SIM --out-dir $SIM/relate_output --stage roc
```

If Relate exits non-zero with `Could not delete directory relate_arg/chunk_0/paint/`
**after** writing `relate_arg.anc` and `relate_arg.mut`, that's a known harmless
cleanup error in commit `b54ede2`; the outputs are valid. Skip `prepare`+`run_relate`
and just run `cr`+`roc` on the existing files.

Outputs: `$SIM/cr_relate.npz`, `$SIM/roc_cr_relate.npz`.

### 5. S\*

```bash
$PY run_sstar_roc.py --sim-dir $SIM --out-dir $SIM/sstar_output --stage prepare
$PY run_sstar_roc.py --sim-dir $SIM --out-dir $SIM/sstar_output --stage run_sstar
$PY run_sstar_roc.py --sim-dir $SIM --out-dir $SIM/sstar_output --stage roc
```

Outputs: `$SIM/roc_sstar.npz`.

### 6. BPP MSC-I (two chains)

```bash
# Prepare phylip + control files
$PY run_bpp_roc.py --stage prepare \
    --simdir $SIM --bppdir $SIM/bpp --n-afr 100

# Duplicate into r1/, r2/ with distinct seeds 11111, 22222
for chain in r1 r2; do
    mkdir -p $SIM/bpp/$chain
    cp $SIM/bpp/{sequences.phy,imap.txt,sample_ids.json} $SIM/bpp/$chain/
    sed -e "s/^seed = .*/seed = $([ $chain = r1 ] && echo 11111 || echo 22222)/" \
        $SIM/bpp/infer.ctl > $SIM/bpp/$chain/infer.ctl
done

# Run both chains (sequential here; in practice run in parallel on
# separate nodes — each takes ~9-10 days).
for chain in r1 r2; do
    (cd $SIM/bpp/$chain && bpp --cfile infer.ctl --no-pin)
done

# Per-chain ROC
$PY run_bpp_roc.py --stage roc \
    --simdir $SIM --bppdir $SIM/bpp --chains r1 r2
```

Outputs: `$SIM/roc_bpp_r1.npz`, `$SIM/roc_bpp_r2.npz`.

### 7. Combined ROC figure

```bash
$PY plot_combined_roc.py --simdir $SIM \
    --out $SIM/roc_comparison_all.pdf \
    --title "Ghost Archaic Introgression Detection ($SIM)"
```

Output: `$SIM/roc_comparison_all.pdf` and a summary table on stdout.

## Verify against expected AUROCs

Compare your AUROC table to [`expected_AUROCs.md`](expected_AUROCs.md). All
methods should reproduce to within ~0.001 modulo:

- SINGER MCMC stochasticity (within ±0.003 across replicate runs at the
  same seed because the SINGER posterior uses internal randomness);
- BPP chain mixing (per-chain AUROC reproduces ~0.0001 between chains in
  our runs, but ±0.001 between full re-runs at the same seed because BPP
  does not save state between iterations of its starting tree).

If any method's AUROC drifts more than ~0.01 from the expected value, suspect
either (a) the wrong tool version (see [`INSTALL.md`](INSTALL.md)) or (b) a
parameter override that wasn't intended.
