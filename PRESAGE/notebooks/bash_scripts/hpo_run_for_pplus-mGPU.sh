#!/bin/bash
# Parallel GPU dispatch for presage HPO sweep
set -u

cleanup() {
    echo ""
    echo "Caught signal — killing all child jobs..."
    # Kill the whole process group (negative PID = process group)
    kill -TERM 0 2>/dev/null
    exec 3>&- 2>/dev/null
    exit 130
}
trap cleanup INT TERM


# ---- config ----
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1 | tr -d ' ')}"
LOG_DIR="${LOG_DIR:-./logs/K562GW}"
mkdir -p "$LOG_DIR"
echo "Dispatching across $NUM_GPUS GPU(s). Logs -> $LOG_DIR"

# ---- build job list ----
JOBS=()
for cline in K562; do
    for TF in 10 30 50; do
        for stype in random stratified; do
            for rs in 2; do
                JOBS+=("${cline}|${TF}|${stype}|${rs}")
            done
        done
    done
done
echo "Total jobs: ${#JOBS[@]}"

# ---- GPU semaphore via FIFO ----
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm "$FIFO"
for ((g=0; g<NUM_GPUS; g++)); do echo "$g" >&3; done

run_job() {
    local gpu_id=$1 cline=$2 TF=$3 stype=$4 rs=$5
    local name="${cline}_TF_${TF}_UF_10_rs_${rs}_${stype}"
    local log="${LOG_DIR}/pplus_${name}.log"

    echo "[GPU ${gpu_id}] START  ${name}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python presage_pbulk_train_final.py \
        --stage hpo \
        --variant pplus \
        --prepared_dir "../data/hpo_pplus_${cline}_${cline}_TF_${TF}_UF_10_rs_${rs}_${stype}" \
        --ds_config_file ../configs/dataset_base_config_hpo_singlecont_V4.json \
        --hpo_num_folds 5 \
        --n_optuna_trials 1 \
        --split_name "$name" \
        > "$log" 2>&1
    local status=$?
    echo "[GPU ${gpu_id}] DONE   ${name} (exit ${status})"

    # release GPU back to the pool
    echo "$gpu_id" >&3
}

# ---- dispatch ----
# Temporarily trim the job list for a smoke test
for job in "${JOBS[@]}"; do 
    IFS='|' read -r cline TF stype rs <<< "$job"
    read -r gpu_id <&3           # blocks until a GPU is free
    run_job "$gpu_id" "$cline" "$TF" "$stype" "$rs" &
done

wait
exec 3>&-
echo "All jobs complete."