#!/usr/bin/env bash
# Chunked SINGER orchestrator (generic; no Slurm dependency).
#
# Splits the 50 Mb singer_input.vcf into 1 Mb segments, runs
# singer_master in parallel on each segment, then merges the per-segment
# ARGs into 100 genome-wide tree sequences via convert_long_ARG.py.
#
# Usage:
#   bash singer_orchestrator.sh <simdir>
#   NCORES=12 bash singer_orchestrator.sh sim_A
#
# Required input in <simdir>:
#   singer_input.vcf   (produced by introgression_roc_pipeline.py --stage simulate)
#
# Optional environment variables (defaults shown):
#   NCORES       cores to use in parallel (default: $(nproc) or 4)
#   SINGER_DIR   path to SINGER source/binary directory containing
#                singer, singer_master, convert_long_ARG.py, index_vcf.py
#                (default: looks them up on $PATH)
#   PY           Python interpreter that has tskit + numpy importable
#                (default: $(which python3))
#   NE           ancestral Ne passed to singer (default: 20000)
#   MU           per-bp per-gen mutation rate (default: 1.2e-8)
#   RATIO        r / mu (default: 1.00)
#   BLOCK        segment size in bp (default: 1000000)
#   N_SAMPLES    posterior samples per segment (default: 100)
#   THIN         singer thinning interval (default: 20)
#   POLAR        site flip probability (default: 0.5)
#
# Output:
#   <simdir>/singer_arg_<i>.trees    for i = 0..N_SAMPLES-1
#   <simdir>/singer_input.index
#
# The pipeline's `--stage cr_singer` step picks up the .trees files
# directly via the introgression_roc_pipeline.py wrapper.

set -uo pipefail

SIM=${1:?usage: bash singer_orchestrator.sh <simdir>}
SIM=$(cd "$SIM" && pwd)   # absolutise

NCORES=${NCORES:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}
PY=${PY:-$(command -v python3)}
NE=${NE:-20000}
MU=${MU:-1.2e-8}
RATIO=${RATIO:-1.00}
BLOCK=${BLOCK:-1000000}
N_SAMPLES=${N_SAMPLES:-100}
THIN=${THIN:-20}
POLAR=${POLAR:-0.5}

# Find SINGER bins
if [ -n "${SINGER_DIR:-}" ]; then
    SINGER_MASTER=$SINGER_DIR/singer_master
    CONVERT_LONG=$SINGER_DIR/convert_long_ARG.py
    INDEX_VCF=$SINGER_DIR/index_vcf.py
else
    SINGER_MASTER=$(command -v singer_master)
    CONVERT_LONG=$(command -v convert_long_ARG.py)
    INDEX_VCF=$(command -v index_vcf.py)
fi

for binname in SINGER_MASTER CONVERT_LONG INDEX_VCF; do
    if [ -z "${!binname:-}" ] || [ ! -e "${!binname}" ]; then
        echo "error: $binname not found (looked for $binname=${!binname:-unset})" >&2
        echo "Set SINGER_DIR to the directory containing singer_master, " \
             "convert_long_ARG.py, index_vcf.py — see INSTALL.md." >&2
        exit 2
    fi
done

VCF_PREFIX=$SIM/singer_input
OUT_PREFIX=$SIM/singer_arg

echo "=== START $(date) ==="
echo "Sim dir:        $SIM"
echo "Cores:          $NCORES"
echo "Python:         $PY"
echo "singer_master:  $SINGER_MASTER"

# ─── Step 1: Index VCF into 1 Mb segments ───
echo
echo "=== Step 1: Indexing VCF into ${BLOCK} bp segments ==="
"$PY" "$INDEX_VCF" "$VCF_PREFIX" "$BLOCK"
NSEGS=$(wc -l < "${VCF_PREFIX}.index")
echo "Total segments: ${NSEGS}"

# ─── Step 2: parallel singer_master per segment ───
echo
echo "=== Step 2: Running SINGER MCMC (${NSEGS} segments, ${NCORES} parallel) ==="
echo "Started at $(date)"

CMDFILE=$(mktemp)
SEG_IDX=0
while read -r BP REST; do
    NEXT=$((SEG_IDX + 1))
    END_BP=$((BP + BLOCK))
    echo "$PY $SINGER_MASTER -Ne ${NE} -m ${MU} -ratio ${RATIO} \
         -vcf ${VCF_PREFIX} -output ${OUT_PREFIX}_${SEG_IDX}_${NEXT} \
         -start ${BP} -end ${END_BP} \
         -n ${N_SAMPLES} -thin ${THIN} -polar ${POLAR}"
    SEG_IDX=$((SEG_IDX + 1))
done < "${VCF_PREFIX}.index" > "$CMDFILE"

echo "Built $(wc -l < "$CMDFILE") commands; first one:"
head -1 "$CMDFILE"

# xargs -P runs N_CORES jobs in parallel; each line is one command
cat "$CMDFILE" | xargs -P "${NCORES}" -I '{}' bash -c '{}'
rm -f "$CMDFILE"

echo "SINGER MCMC finished at $(date)"
echo "Sanity check — node files per iteration:"
for i in 0 1 50 99; do
    CNT=$(ls "${OUT_PREFIX}"_*_*_nodes_${i}.txt 2>/dev/null | wc -l)
    echo "  iteration ${i}: ${CNT}/${NSEGS} segments"
done

# ─── Step 3: convert_long_ARG.py per posterior sample ───
echo
echo "=== Step 3: Merging segments into ${N_SAMPLES} genome-wide .trees ==="
for i in $(seq 0 $((N_SAMPLES - 1))); do
    "$PY" "$CONVERT_LONG" \
        -vcf "$VCF_PREFIX" -output "$OUT_PREFIX" -iteration "$i" 2>/dev/null \
        && echo "  Converted iteration ${i}" \
        || echo "  FAILED iteration ${i}"
done

N_TREES=$(ls "${OUT_PREFIX}"_*.trees 2>/dev/null | wc -l)
echo "=== Done at $(date) ==="
echo "Trees files created: ${N_TREES}/${N_SAMPLES}"
