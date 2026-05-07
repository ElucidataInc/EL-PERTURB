#!/bin/bash
set -euo pipefail

# ============================================================================
# kpdp variant — processed all cell-lines' h5ads (~5-6 GB each)
#
# As such this script will run 18 jobs total (3 cell lines [D1 HCT HEK] × 3 TF × 2 split types × 1 variant)
# Each job uses ~30GB working memory
# To customize, i.e. launch for 1 or 2 cell lines only see Usage or adjust CELL_LINES 
# To customize TF, variant, random, stratified -- adjust the last part (i.e. see code block beginning with parallel ... ) 

# Usage:
#   ./data_prep_for_hpo_runs_kpdp.sh [MAX_JOBS] [CELL_LINE]
#   ./data_prep_for_hpo_runs_kpdp.sh 1            # all cell lines, 2 parallel
#   ./data_prep_for_hpo_runs_kpdp.sh 1 K562       # K562 only, 2 parallel 
#
# Recommendation:
#   64GB VM → MAX_JOBS=1 
#  128GB VM -> MAX_JOBS=2 
# ============================================================================

MAX_JOBS=${1:-1}
CELL_LINE=${2:-all}

if [[ "$CELL_LINE" == "all" ]]; then
    CELL_LINES=(K562)
else
    CELL_LINES=("$CELL_LINE")
fi

run_job() {
    local cline="$1"
    local TF="$2"
    local stype="$3"

    local split="${cline}_TF_${TF}_UF_10_rs_2_${stype}"
    local target="$cline"

    echo "[START] kpdp | target=$target | TF=$TF | split=$stype"

    python prepare_presage_data_memopt_scdata.py \
        --scdata_file "../data/merged_GW_screens_lognorm_single_cell.h5ad" \
        --split_col "$split" \
        --target_name "$target" \
        --target_key "cell_type" \
        --pert_key "perturbation" \
        --pert_group_key "perturbation_group" \
        --control_key "non-targeting" \
        --prior_type "all" \
        --prior_by_group "cell_type" \
        --n_nmf_embedding "128" \
        --output_dir "../data/hpo_kpdp_${target}_${split}" \
        --mode "target_fold_5" \
        --stage "hpo" \
        --variant "kpdp" \
        --skip_degs

    echo "[DONE]  kpdp | target=$target | TF=$TF | split=$stype"
}

export -f run_job

echo "============================================"
echo "kpdp variant (DEGs skipped)"
echo "Cell lines: ${CELL_LINES[*]}"
echo "Running with ${MAX_JOBS} parallel jobs"
echo "============================================"

parallel --halt now,fail=1 \
         --jobs "$MAX_JOBS" \
         --colsep ' ' \
         --tagstring '[{1}|TF{2}|{3}|kpdp]' \
         run_job {1} {2} {3} \
         ::: "${CELL_LINES[@]}" \
         ::: 10 30 50 \
         ::: random stratified

echo ""
echo "All kpdp jobs completed successfully."