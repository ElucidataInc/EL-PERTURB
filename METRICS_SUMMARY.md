# Perturbation Prediction Metrics Framework

Summary of the performance evaluation framework implemented in `calc_performance_metrics_excl_pert_gene.py`. The framework assesses how well models predict perturbation-specific transcriptional responses by comparing predictions against ground truth DE profiles (Pseudobulk logFC).

---

## 1. Core Metrics

The metrics are categorized to provide a hierarchical view of model performance:
- **Transcriptome-level Sanity Check (MSE, Pearson)**: Assess the overall error in magnitude and direction of changes across the full transcriptome.
- **Biologically Significant Response (Pearson Top-K)**: Focus on the accuracy of the strongest, biologically significant responses with respect to the unperturbed control state.
- **Perturbation-Specific Response (WMSE, Pearson-Pert, Rank Scores, Matrix Distance)**: Assess how well the model predicts the unique differentiating signals that separate one perturbation from others, rather than just learning the shared transcriptional shift.

| Metric | Definition | Interpretation |
| :--- | :--- | :--- |
| **MSE** | Mean Squared Error across all genes. | Overall magnitude error (lower is better). |
| **WMSE** | Weighted MSE where gene weights are proportional to their significance (absolute t-scores) wrt the mean perturbed state. | Meaures accuracy on genes that deviate most from the general cell-line perturbed state, focusing on perturbation-specific signal intensity. |
| **Pearson** | Pearson correlation between predicted and ground truth LFC vectors. | Linear similarity of the predicted full transcriptomic profile. |
| **Pearson Top-K DEGs** | Pearson correlation restricted to the top 50 DEGs of the perturbation. | Accuracy on the most characteristic genes responding to the perturbation. |
| **Pearson Pert** | Pearson correlation of $(\text{Pred} - \text{CellMean})$ vs $(\text{GT} - \text{CellMean})$. | Isolates the model's ability to predict the specific directional shift that distinguishes this perturbation from the general cell-type mean across all training perturbations. |
| **Rank Score (L1/Cosine)** | Normalized retrieval rank: $(N - rank) / (N - 1)$ where $N$ is test set size. | Ability to distinguish the target perturbation from others. |
| **Matrix Distance** | Frobenius norm of $(\text{SimilarityMatrix}_{\text{pred}} - \text{SimilarityMatrix}_{\text{true}})$. | Preservation of global manifold structure between perturbations. |

---

## 2. Baselines & Comparisons

To provide context, every metric is compared against three naive baselines:

1.  **Cell-Mean (CM)**: The mean expression profile calculated over all perturbed cells in the target context training set. It represents the shared transcriptional background and sets the performance floor for a baseline that captures cell-type-specific but not perturbation-specific signals.
2.  **Perturb-Mean (PM)**: The average response to the same perturbation across *other* contexts/screens. Captures cross-context biological consistency.
3.  **Zero-Change**: A dummy predictor representing no response to the perturbation (vector of zeros).

---

## 3. Normalized Scoring System

Raw metrics can be difficult to interpret across different magnitudes. We use a **Scaled Score** system that normalizes model performance relative to the **Cell-Mean baseline**:

*   **Scale High (for Correlations/Ranks)**: $S = \frac{\text{Model} - \text{Baseline}}{1 - \text{Baseline}}$
*   **Scale Low (for MSE/Distance)**: $S = \frac{\text{Baseline} - \text{Model}}{\text{Baseline}}$
*   **Note**: For Pearson-Pert, the zero-change baseline is used instead of cell-mean baseline.

A score of **1.0** represents perfect prediction, **0.0** represents performance equal to the baseline, and negative values represent performance worse than the baseline.

---

## 4. Aggregated Scores

The pipeline distills performance into two primary meta-scores, separately for the test_seen and test_unseen samples as well as the full (Combined) test set:

### **Global Score (GS)**
The arithmetic mean of all 8 scaled metrics. It provides a rounded view of model quality across magnitude, correlation, retrieval, and manifold preservation.
> `GS = mean(s_MSE, s_WMSE, s_Pearson, s_Pearson_Pert, s_Pearson_TopK, s_Rank_L1, s_Rank_Cos, s_Mat_Dist)`

### **Specificity Score (SS)**
A focused score emphasizing the model's ability to capture **perturbation-specific** variance rather than just mean cell-type signals.
> `SS = mean(s_WMSE, s_Rank_L1, s_Rank_Cos, s_Mat_Dist, s_Pearson_Pert)`

---

## 5. Implementation Details

### Cis Effect Ignored
The perturbed gene is excluded as a var feature from the calculation of all the metrics per perturbation, 
to focus only on the trans effects of every perturbation.

### WMSE Weighting Logic
Weights are dynamically calculated per row based on the absolute t-scores found in the ground truth.
1.  Absolute t-scores are range-normalized to $[0, 1]$.
2.  Values are squared to emphasize peaks.
3.  Normalized such that the sum of weights per gene equals 1.0.

### Expected Inputs
*   **Prediction (.h5ad)**: Must contain `uns['training_params']` with `split_column` and `target_screen` keys.
*   **Ground Truth (.h5ad)**: Must contain a layer matching `t_scores_{split_suffix}` and an `obs` column for splitting.

### Saved Outputs
1.  **Annotated .h5ad**: The prediction file is updated with:
    *   `uns['performance_metrics_summary']`: A dataframe of aggregated results by group.
    *   `uns['performance_metrics_per_gene']`: Raw metrics for every evaluated sample.
2.  **Summary CSV**: A copy of the aggregated summary is saved to `./results/` for benchmarking.
