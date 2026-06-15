# Installation

The pipeline drives five external tools (BPP, SINGER, Relate, IBDmix, sstar)
and a Python environment built on `msprime`/`tskit`. **The version pins here
are the exact versions used to produce the figures in the accompanying
manuscript** — newer releases may produce slightly different AUROCs.

## 1. Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The constraint `tskit>=1.0,<1.1` is deliberate: tskit's mutation-table
validation tightened in 1.0 and the upstream SINGER `convert_to_tskit`
wrapper does not call `tables.sort()` + `tables.compute_mutation_parents()`
before `tables.tree_sequence()`. If you must run on a newer tskit, patch
that converter (see the comment in `singer_orchestrator.sh`).

## 2. BPP (dev branch, commit `1a8e4ab` or later)

```bash
git clone https://github.com/bpp/bpp.git ~/bpp
cd ~/bpp
git checkout dev
# Confirm you have the per-sequence-ancestry race fix (Mar 2026 or later)
git log --oneline | head | grep 1a8e4ab
make
ln -sf ~/bpp/src/bpp ~/bin/bpp     # or add ~/bpp/src to $PATH
```

The dev branch fixes a race condition in per-sequence ancestry estimation
that affected the production (`master`) branch with `threads > 1`. The
ancestry output format is `Locus,Sequence,PP` (CSV); `run_bpp_roc.py`
parses this format directly.

## 3. SINGER v0.1.8

```bash
# Choose either the binary release or build from source.

# Option A: binary release (Linux x86_64)
curl -L -o singer-0.1.8.tar.gz \
    https://github.com/popgenmethods/SINGER/releases/download/v0.1.8/singer-0.1.8-beta-linux-x86_64.tar.gz
tar -xzf singer-0.1.8.tar.gz
# The directory contains the C++ `singer` binary plus the Python wrappers
# `singer_master`, `convert_to_tskit`, `convert_long_ARG.py`, `index_vcf.py`.

# Option B: clone + build
git clone https://github.com/popgenmethods/SINGER.git ~/SINGER
cd ~/SINGER/SINGER/SINGER
git checkout v0.1.8
make
```

Add the SINGER directory to `$PATH` so `singer_master`, `convert_to_tskit`,
`convert_long_ARG.py` are callable. **Do NOT** use SINGER `dev` HEAD past
v0.1.8 — there is an open numerical issue that produces incorrect coalescence
times. The pinned md5s are:

```
singer         c5f17dc5e4ee697d8a6a5295772dd1b3
singer_master  fcc840174cee9b9cbaa34f388ef7794d
```

## 4. Relate (commit `b54ede2`)

```bash
git clone https://github.com/MyersGroup/relate.git ~/relate
cd ~/relate
git checkout b54ede2
mkdir build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j4
```

You also need `relate_lib`'s `Convert` binary for the tskit conversion:

```bash
git clone https://github.com/leospeidel/relate_lib.git ~/relate_lib
cd ~/relate_lib
git checkout 9a7e703
mkdir build && cd build
cmake ..
make -j4
```

Then either add the binaries to `$PATH` or export:

```bash
export RELATE_BIN=~/relate/build/Relate
export RELATE_FILE_FORMATS=~/relate/build/RelateFileFormats
export CONVERT_BIN=~/relate_lib/build/Convert
```

`run_relate_roc.py` reads those environment variables and falls back to
PATH lookup.

## 5. IBDmix v1.0.1

```bash
conda install -c bioconda ibdmix=1.0.1
# binaries: ibdmix, generate_gt
```

## 6. S\* (`sstar` v1.1+)

Installed via `requirements.txt` (`sstar>=1.1`). No separate setup.

## Smoke test

```bash
.venv/bin/python introgression_roc_pipeline.py --stage simulate \
    --outdir /tmp/simtest --seed 42 --seq-length 1000000
.venv/bin/python introgression_roc_pipeline.py --stage cr_true \
    --outdir /tmp/simtest
ls /tmp/simtest/cr_true.npz
```

If `cr_true.npz` exists, msprime + the core pipeline are correctly installed.
