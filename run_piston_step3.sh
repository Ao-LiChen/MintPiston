#!/bin/bash
# run_piston_step3.sh
# Run piston.sif feature extraction.
# Usage:
#   bash run_piston_step3.sh              # use 32 parallel jobs (default)
#   bash run_piston_step3.sh 8            # use 8 parallel jobs
#   bash run_piston_step3.sh 32 600       # 32 jobs, 600s timeout per PPI

N_JOBS=${1:-32}
TIMEOUT=${2:-600}

PIPELINE_DIR="/home/chenaoli/piston/MintPiston"
PIPELINE_OUT="$PIPELINE_DIR/pipeline_out"
PPI_LIST="$PIPELINE_OUT/ppi_list.txt"
CONFIG="$PIPELINE_OUT/piston_config.py"
GRID_DIR="$PIPELINE_OUT/grid"
LOG_DIR="$PIPELINE_OUT/logs"
SIF="$PIPELINE_DIR/downloads/piston.sif"
PISTON_PY="$PIPELINE_DIR/piston-main/piston.py"

if [ ! -f "$PPI_LIST" ]; then
    echo "ERROR: PPI list not found: $PPI_LIST"
    echo "Run: python sabdab_pipeline.py --step 1 first"
    exit 1
fi

if [ ! -f "$SIF" ]; then
    echo "ERROR: piston.sif not found: $SIF"
    exit 1
fi

mkdir -p "$LOG_DIR" "$GRID_DIR"

# Build list of PPIs still to process
TODO=$(mktemp)
while IFS= read -r ppi; do
    [ -z "$ppi" ] && continue
    if [ ! -f "$GRID_DIR/${ppi}.npy" ]; then
        echo "$ppi" >> "$TODO"
    fi
done < "$PPI_LIST"

TOTAL=$(wc -l < "$PPI_LIST")
TODO_N=$(wc -l < "$TODO")
DONE_N=$(( TOTAL - TODO_N ))

echo "[piston step3] Total: $TOTAL  Done: $DONE_N  To process: $TODO_N"
echo "[piston step3] Parallel jobs: $N_JOBS  Timeout per PPI: ${TIMEOUT}s"

if [ "$TODO_N" -eq 0 ]; then
    echo "[piston step3] Nothing to do!"
    rm -f "$TODO"
    exit 0
fi

# Run one PPI with apptainer (with protonation; piston checks raw_pdb dir and skips download if file exists)
run_one() {
    local ppi="$1"
    local log_file="$LOG_DIR/${ppi}.log"

    # Skip if already done
    if [ -f "$GRID_DIR/${ppi}.npy" ]; then
        return 0
    fi

    timeout "$TIMEOUT" apptainer exec \
        --bind "$PIPELINE_DIR/piston-main:$PIPELINE_DIR/piston-main" \
        --bind "$PIPELINE_OUT:$PIPELINE_OUT" \
        "$SIF" \
        python3 "$PISTON_PY" \
        --config "$CONFIG" \
        prepare --ppi "$ppi" \
        > "$log_file" 2>&1

    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAILED $ppi (exit $rc)" >&2
        return 1
    fi
    echo "OK $ppi" >&2
}
export -f run_one
export TIMEOUT LOG_DIR GRID_DIR PIPELINE_DIR PIPELINE_OUT SIF PISTON_PY CONFIG

# Use GNU parallel or xargs for parallel execution
if command -v parallel &>/dev/null; then
    echo "[piston step3] Using GNU parallel"
    parallel -j "$N_JOBS" --bar run_one {} < "$TODO"
else
    echo "[piston step3] Using xargs (install GNU parallel for better output)"
    cat "$TODO" | xargs -P "$N_JOBS" -I{} bash -c 'run_one "$@"' _ {}
fi

rm -f "$TODO"

# Report
DONE_AFTER=$(ls "$GRID_DIR"/*.npy 2>/dev/null | wc -l)
FAILED=$(( TOTAL - DONE_AFTER ))
echo "[piston step3] Complete. Grid files: $DONE_AFTER / $TOTAL  (failed/missing: $FAILED)"
