#!/bin/bash
set -euo pipefail

# ============================================================================
# Non-kpdp variants (pplus, only) — uses per-cell-line h5ad (~5-6GB each)
#
# As such this script will run 36 jobs total (3 cell lines [D1 HCT HEK] × 3 TF × 2 split types × 2 variants)
# Each job uses ~15-20 GB working memory
# To customize, i.e. launch for 1 or 2 cell lines only see Usage or adjust CELL_LINES 
# To customize TF, variant, random, stratified -- adjust the last part (i.e. code block beginning with parallel ... ) 

# Usage:
#   ./data_prep_for_hpo_runs_non_kpdp.sh [MAX_JOBS] [CELL_LINE]
#   ./data_prep_for_hpo_runs_non_kpdp.sh 2            # all cell lines, 2 parallel
#   ./data_prep_for_hpo_runs_non_kpdp.sh 2 K562       # K562 only, 2 parallel 
#
# Recommendation:
#   64GB VM → MAX_JOBS=2 
#  128GB VM -> MAX_JOBS=3 
# ============================================================================

MAX_JOBS=${1:-2}
CELL_LINE=${2:-all}  # single cell line or "all"

if [[ "$CELL_LINE" == "all" ]]; then
    CELL_LINES=(K562) 
else
    CELL_LINES=("$CELL_LINE")
fi

run_job() {
    local cline="$1"
    local TF="$2"
    local stype="$3"
    local variant="$4"
    local seed="$5"

    local split="${cline}_TF_${TF}_UF_10_rs_${seed}_${stype}"  # rs_2 here = seed 2 splits for hpo
    local target="$cline"

    echo "[START] variant=$variant | target=$target | TF=$TF | split=$stype | seed=$seed" 

    python prepare_presage_data_memopt_scdata.py \
        --scdata_file "../data/merged_GW_screens_lognorm_single_cell.h5ad" \
        --split_col "$split" \
        --target_name "$target" \
        --target_key "cell_type" \
        --pert_key "perturbation" \
        --pert_group_key "perturbation_group" \
        --control_key "non-targeting" \
        --prior_type "all" \
        --n_nmf_embedding "128" \
        --output_dir "../data/hpo_${variant}_${target}_${split}" \
        --mode "target_fold_5" \
        --stage "hpo" \
        --variant "$variant" \
        --skip_degs

    echo "[DONE]  variant=$variant | target=$target | TF=$TF | split=$stype | seed=$seed"
}

export -f run_job

echo "============================================"
echo "Non-kpdp variants (pplus, only)"
echo "Cell lines: ${CELL_LINES[*]}"
echo "Running with ${MAX_JOBS} parallel jobs"
echo "============================================"

parallel --halt now,fail=1 \
         --jobs "$MAX_JOBS" \
         --colsep ' ' \
         --tagstring '[{1}|TF{2}|{3}|{4}|{5}]' \
         run_job {1} {2} {3} {4} {5}\
         ::: "${CELL_LINES[@]}" \
         ::: 10 50 30 \
         ::: random stratified \
         ::: only pplus \
         ::: 2

echo ""
echo "All non-kpdp jobs completed successfully."