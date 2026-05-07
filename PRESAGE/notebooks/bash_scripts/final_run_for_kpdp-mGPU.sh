#!/bin/bash
# Parallel GPU dispatch for presage final training sweep (kpdp variant, rs swept 1..3)
set -u
shopt -s nullglob

# ---- config ----
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1 | tr -d ' ')}"
LOG_DIR="${LOG_DIR:-./logs_final}"
mkdir -p "$LOG_DIR"
echo "Dispatching across $NUM_GPUS GPU(s). Logs -> $LOG_DIR"

# ---- cleanup on Ctrl+C ----
cleanup() {
    echo ""
    echo "Caught signal — killing all child jobs..."
    kill -TERM 0 2>/dev/null
    exec 3>&- 2>/dev/null
    exit 130
}
trap cleanup INT TERM

# ---- build job list ----
JOBS=()
for file in ./search_spaces_kpdp/K562GW/HPO_kpdp_*.json; do
    base="${file##*/}"
    IFS='_' read -r -a parts <<< "$base"
    if [[ ${#parts[@]} -lt 10 ]]; then
        echo "Skipping malformed: $file"
        continue
    fi
    cell="${parts[2]}"
    tf="${parts[4]}"
    mode="${parts[9]}"
    # rs from filename is intentionally ignored — we sweep it below
    for rs in 1 2 3; do
        JOBS+=("${file}"$'\t'"${cell}"$'\t'"${tf}"$'\t'"${rs}"$'\t'"${mode}")
    done
done
echo "Total jobs: ${#JOBS[@]}"

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "No matching JSON files found in ./search_spaces_kpdp/"
    exit 1
fi

# ---- GPU semaphore via FIFO ----
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm "$FIFO"
for ((g=0; g<NUM_GPUS; g++)); do echo "$g" >&3; done

run_job() {
    local gpu_id=$1 file=$2 cell=$3 tf=$4 rs=$5 mode=$6
    local name="${cell}_TF_${tf}_UF_10_rs_${rs}_${mode}"
    local log="${LOG_DIR}/kpdp_${name}.log"

    echo "[GPU ${gpu_id}] START  ${name}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python presage_pbulk_train_final.py \
        --stage final \
        --variant kpdp \
        --prepared_dir "../data/final_rs${rs}/hpo_kpdp_${cell}_${cell}_TF_${tf}_UF_10_rs_${rs}_${mode}" \
        --ds_config_file ../configs/dataset_base_config_hpo_V1-Large.json \
        --hpo_num_folds 1 \
        --split_name "$name" \
        --hpo_search_space "$file" \
        > "$log" 2>&1
    local status=$?
    echo "[GPU ${gpu_id}] DONE   ${name} (exit ${status})"

    echo "$gpu_id" >&3
}

# ---- dispatch ----
for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r file cell tf rs mode <<< "$job"
    read -r gpu_id <&3
    run_job "$gpu_id" "$file" "$cell" "$tf" "$rs" "$mode" &
done

wait
exec 3>&-
echo "All jobs complete."