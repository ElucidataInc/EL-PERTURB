from functools import partial
from typing import Callable, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import os
import json
import anndata
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr, rankdata,  median_abs_deviation

from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    roc_auc_score,
    recall_score,
)

from functools import cache

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

import scanpy as sc
import pickle as pkl


def sphering_transform(cov, reg_param=1e-6, reg_trace=False, gpu=False, rotate=True):
    xp = cp if gpu else np

    # shrink
    if reg_trace:
        cov = (1 - reg_param) * cov + reg_param * xp.trace(cov) / cov.shape[0] * xp.eye(
            cov.shape[0]
        )
    else:
        cov = cov + reg_param * xp.eye(cov.shape[0])
    s, V = xp.linalg.eigh(cov)

    D = xp.diag(1.0 / xp.sqrt(s))
    Dinv = xp.diag(xp.sqrt(s))
    W = V @ D
    Winv = Dinv @ V.T
    if rotate:
        W = W @ V.T
        Winv = V @ Winv

    return W, Winv

def empirical_covariance(X, gpu=False, ddof=1.0):
    xp = cp if gpu else np
    # Center data
    n_obs, _ = X.shape
    loc = xp.mean(X, axis=0)
    X = X - loc
    cov = (X.T @ X) / (n_obs - ddof)
    return loc, cov


def oas_covariance(X, gpu=False, ddof=1.0):
    xp = cp if gpu else np
    X = xp.asarray(X)

    # Calculate covariance matrix
    n_obs, n_feat = X.shape
    loc = xp.mean(X, axis=0)
    X = X - loc
    cov = (X.T @ X) / (n_obs - ddof)

    # Calculate sufficient statistics
    tr = xp.trace(cov)
    tr_sq = tr**2
    frob_sq = xp.sum(cov**2)

    # Calculate OAS statistics
    num = (1 - 2 / n_feat) * frob_sq + tr_sq
    denom = (n_obs + 1 - 2 / n_feat) * (frob_sq - tr_sq / n_feat)
    shrinkage = 1.0 if denom == 0 else min(num / denom, 1.0)
    cov_ret = (1.0 - shrinkage) * cov + shrinkage * tr / n_feat

    return loc, cov_ret


class SpheringTransform(object):
    def __init__(
        self,
        controls,
        reg_param=1e-6,
        reg_trace=False,
        rotate=True,
        gpu=False,
        oas=False,
    ):
        self.gpu = gpu

        if self.gpu:
            controls = cp.asarray(controls)
        if oas:
            self.mu, cov = oas_covariance(controls, gpu=gpu)
        else:
            self.mu, cov = empirical_covariance(controls, gpu=gpu)
        self.W, self.Winv = sphering_transform(
            cov, reg_param=reg_param, reg_trace=reg_trace, gpu=gpu, rotate=rotate
        )

    def normalize(self, X):
        xp = cp if self.gpu else np
        if self.gpu:
            X = cp.asarray(X)

        ret = (X - self.mu) @ self.W
        return ret

    def recolor(self, X):
        xp = cp if self.gpu else np
        if self.gpu:
            X = cp.asarray(X)

        ret = X @ self.Winv + self.mu
        return ret


def run_sphering_transform(
    adata: anndata.AnnData,
    reg_param: float = 1e-3,
    reg_trace: bool = True,
    gpu: bool = False,
    oas: bool = True,
    query: str = "gene_symbol == 'NTC'",
    embedding_key: str = "X_pca",
    out_key: str = "X_pca_sphered",
) -> anndata.AnnData:
    if query is not None:
        idx = adata.obs.query(query).index
    else:
        idx = adata.index

    control_features = adata[idx, :].obsm[embedding_key]
    if gpu:
        control_features = cp.asarray(control_features)

    normalizer = SpheringTransform(
        control_features, reg_param=reg_param, reg_trace=reg_trace, gpu=gpu, oas=oas
    )

    # Apply the normalization transformation
    features = adata.obsm[embedding_key]
    if gpu:
        features = cp.asarray(features)
    features_sphered = normalizer.normalize(adata.obsm[embedding_key])
    if gpu:
        features_sphered = features_sphered.get()

    adata.obsm[out_key] = features_sphered

    return adata


class EvaluationSuite:
    """
    Wraps the functions in the eval suite. This would be called by the Evaluator class to specify which eval functions to do
    """

    def __init__(
        self,
        results: Iterable[pd.DataFrame],
        unfiltered_results: Iterable[pd.DataFrame],
        eval_functions: Iterable[str],
        train_perturbation_labels: np.array,
        control_cells: np.array,
        pert_mean: pd.DataFrame,
        kvals: Tuple[int] = (5, 10, 20, 50, 100, 200, 1000),
        geneset_file: str = "None",
        perturbation_cluster_file: str = "None",
        only_embedding: bool = False,
        both_embedding: bool = False,
    ):
        # this is if you pass an embedding directly to the eval
        # it enables geometric eval between your embedding space and the perturbseq data
        self.only_embedding = only_embedding
        self.both_embedding = both_embedding

        self.results = results
        self.unfiltered_results = unfiltered_results

        self.ctrl_cells = control_cells
        self.pert_mean = pert_mean

        self.train_perturbation_labels = train_perturbation_labels
        self.kvals = kvals
        self.eval_functions = eval_functions
        self.geneset_file = geneset_file
        self.perturbation_cluster_file = perturbation_cluster_file
        self.evals = {}
        self.each_perturbation_eval = {}

        self.topk_eval_function_suite = {
            "mse_topk_de": self.mse_topk_de,
            "avg_mse_topk_de": self.avg_mse_topk_de,
            "avg_normalized_mse_topk_de": self.avg_normalized_mse_topk_de,
            "avg_normalized_mse_topk_unionde": self.avg_normalized_mse_topk_unionde,
            "pn_mse_topk_de": self.norm_normalized_mse_topk_de,
            "norm_error_unionde": self.norm_error_unionde,
            "norm_rel_error_unionde": self.norm_rel_error_unionde,
            "pn_mse_topk_unionde": self.norm_normalized_mse_topk_unionde,
            "avg_pearson_topk_de": self.avg_pearson_topk_de,
            "avg_pearson_topk_unionde": self.avg_pearson_topk_unionde,
            "avg_cossim_topk_de": self.avg_cossim_topk_de,
            "avg_cossim_topk_unionde": self.avg_cossim_topk_unionde,
            "avg_cossim_topk_unionde_ntcmean": self.avg_cossim_topk_unionde_ntcmean,
            "avg_cossim_topk_de_ntcmean": self.avg_cossim_topk_de_ntcmean,
        }

        self.nontopk_eval_function_suite = {
            "geom/similarity_knn": self.similarity_knn,
            "phenocopy/phenocopy_virtual_screen": self.phenocopy_virtual_screen,
            # "virtual_screen":self.virtual_screen,
        }
        self.done_geometric_preprocess = False

    def __call__(self, train: bool):
        self.train = train

        # if not train and not self.only_embedding:
        if not self.only_embedding:
            self.results = [
                r
                for r in self.results
                if r.index.name not in self.train_perturbation_labels
            ]

            self.train_unfiltered_results = [
                r
                for r in self.unfiltered_results
                if r.index.name in self.train_perturbation_labels
            ]

            self.unfiltered_results = [
                r
                for r in self.unfiltered_results
                if r.index.name not in self.train_perturbation_labels
            ]
        #else:
        #    self.train_unfiltered_results = self.unfiltered_results[0].loc[np.isin(self.unfiltered_results.index,self.train_perturbation_labels),:]
        #    self.unfiltered_results[0] = self.unfiltered_results[0].loc[np.isin(self.unfiltered_results.index,self.train_perturbation_labels,invert=True),:]
        #    self.unfiltered_results[1] = self.unfiltered_results[1].loc[np.isin(self.unfiltered_results.index,self.train_perturbation_labels,invert=True),:]

        self.evals_df = pd.DataFrame(
            columns=["metric_category", "geneset", "k", "metric", "val"]
        )

        if len(self.results) > 0:
            if not self.only_embedding:

                for k in self.kvals:
                    print(k)
                    self._take_topk(k)

                    self._take_topk_union_degs(k)
                    self._take_topk_overlapping_degs(k)
                    for eval in self.eval_functions:
                        if eval in self.topk_eval_function_suite:
                            
                            metric = eval.replace("topk", f"top{k}")

                            val = self.topk_eval_function_suite[eval]()
                            self.evals[metric] = val
                            self.evals_df.loc[self.evals_df.shape[0], :] = [
                                "univariate",
                                "na",
                                k,
                                eval,
                                val,
                            ]

            k_geom = np.array([1, 2, 5, 10, 20, 50, 100])
            # this is for the deviation degs
            k_geom = k_geom[k_geom < np.max(self.kvals)]

            if len(self.unfiltered_results) > 1:

                for eval in self.eval_functions:
                    if eval in self.nontopk_eval_function_suite:
                        # print(eval)
                        eval_split = eval.split("/")
                        if len(eval_split) == 2:
                            category = eval_split[0]
                        # else:
                        #    category = 'virtual_screen'
                        for k in k_geom:
                            if not self.done_geometric_preprocess:
                                self.perturb_geometric_preprocess()
                                self.done_geometric_preprocess = True
                            self.current_k = k

                            if eval == "geom/phenocopy_virtual_screen":
                                if k != k_geom[0]:
                                    continue
                                geom_metric = self.nontopk_eval_function_suite[eval]()
                                for m in geom_metric:
                                    self.evals.update({f"{m}": geom_metric[m]})
                                    val = geom_metric[m]
                                    geneset = "na"
                                    self.evals_df.loc[self.evals_df.shape[0], :] = [
                                        category,
                                        geneset,
                                        0,
                                        m,
                                        val,
                                    ]

                            else:

                                geom_metric = self.nontopk_eval_function_suite[eval](k)

                                for m in geom_metric:
                                    self.evals.update({f"{m}_{k}": geom_metric[m]})

                                    val = geom_metric[m]

                                    geneset = "na"

                                    self.evals_df.loc[self.evals_df.shape[0], :] = [
                                        category,
                                        geneset,
                                        k,
                                        m,
                                        val,
                                    ]

                # virtual_screen
                k_virtual_screen_mad = [1, 2, 3, 4, 5]
                eval = "virtual_screen"
                category = "virtual_screen"

                for k in k_virtual_screen_mad:
                    virtual_metric = self.virtual_screen(k)

                    for m in virtual_metric:
                        self.evals.update({f"{m}_{k}": virtual_metric[m]})

                        val = virtual_metric[m]

                        if "positive" in m or "negative" in m:
                            geneset = "_".join(m.split("_")[:-2])
                            m = "_".join(m.split("_")[-2:])
                        elif "spearman" in m:
                            geneset = "_".join(m.split("_")[:-1])
                            m = "spearman"

                        self.evals_df.loc[self.evals_df.shape[0], :] = [
                            category,
                            geneset,
                            k,
                            m,
                            val,
                        ]

    def _take_topk(self, k):
        if k is None:
            k = self.results[0].shape[1]
        self.current_k = k

        min_n_genes = self.results[0].shape[1]
        for result in self.results:
            if result.shape[1] < min_n_genes:
                min_n_genes = result.shape[1]
        if min_n_genes < k:
            k = min_n_genes

        (
            self.ranked_truth,
            self.ranked_pred,
            self.perts_order,
            self.perts_order_union,
            self.feature_genes,
        ) = ([], [], [], [], [])
        for result in self.results:
            temp_result_tgt = result.loc["tgt"]
            # temp_result_tgt = temp_result_tgt[temp_result_tgt.abs() > 0.1]

            temp_result_pred = result.loc["pred"]
            # temp_result_pred = temp_result_pred.loc[temp_result_tgt.index]

            # if temp_result_tgt.shape[0] <= 1:
            #    continue

            self.ranked_truth.append(temp_result_tgt.values[:k])
            self.ranked_pred.append(temp_result_pred.values[:k])
            self.feature_genes.append(temp_result_tgt.index[:k])
            self.perts_order.append(result.index.name)

        self.ranked_truth = np.vstack(self.ranked_truth)
        self.ranked_pred = np.vstack(self.ranked_pred)

    def _take_topk_union_degs(self, k):
        deg_union = []
        for result in self.results:
            temp = result.loc["tgt"]
            # temp = temp[temp.abs() > 0.1]
            deg_union += list(temp.index[:k])
        deg_union = np.unique(deg_union)

        self.ranked_truth_union, self.ranked_pred_union = [], []
        for result in self.results:
            #print(result)
            # drop duplicate columns by name, keep the first one
            result = result.loc[:, ~result.columns.duplicated()]

            #found = [g for g in deg_union if g in result.columns]
            self.ranked_truth_union.append(result.loc["tgt"].reindex(deg_union).to_numpy())
            self.ranked_pred_union.append(result.loc["tgt"].reindex(deg_union).to_numpy())
            self.perts_order_union.append(result.index.name)
        self.deg_union = deg_union

        self.ranked_truth_union = np.vstack(self.ranked_truth_union)
        self.ranked_pred_union = np.vstack(self.ranked_pred_union)

    def _take_topk_overlapping_degs(self, k):
        deg_overlap = {}
        for result in self.results:
            tgt = result.loc["tgt"]
            pred = result.loc["pred"]

            deg = list(tgt.index[:k])
            for i, g in enumerate(deg):
                if g not in deg_overlap:
                    deg_overlap[g] = {}
                    deg_overlap[g]["tgt"] = []
                    deg_overlap[g]["pred"] = []
                deg_overlap[g]["tgt"].append(tgt.iloc[i])
                deg_overlap[g]["pred"].append(pred.iloc[i])
        self.ranked_truth_and_pred_overlap = deg_overlap

        new_ranked_truth_and_pred = {}
        for g in self.ranked_truth_and_pred_overlap:
            if len(self.ranked_truth_and_pred_overlap[g]["tgt"]) >= 20:
                new_ranked_truth_and_pred[g] = self.ranked_truth_and_pred_overlap[g]

        self.ranked_truth_and_pred_overlap = new_ranked_truth_and_pred

    def mse_topk_de(self) -> float:
        return mean_squared_error(self.ranked_truth, self.ranked_pred)

    def avg_mse_topk_de(self) -> float:
        return [
            mean_squared_error(tgt, pred)
            for tgt, pred in zip(self.ranked_truth, self.ranked_pred)
        ]

    def avg_normalized_mse_topk_de(self) -> float:
        counter = 0
        mse, norm = 0.0, 0.0
        for tgt, pred in zip(self.ranked_truth, self.ranked_pred):
            # saving single perturbations
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            mse_added = mean_squared_error(_tgt, _pred)

            p = self.perts_order[counter]
            if p not in self.each_perturbation_eval:
                self.each_perturbation_eval[p] = {}
            self.each_perturbation_eval[p][f"mse_{self.current_k}"] = mse_added

            mse += mse_added
            norm += mean_squared_error(_tgt, np.zeros_like(_tgt))
            counter += 1
        return mse / norm

    def avg_normalized_mse_topk_unionde(self) -> float:
        counter = 0
        mse, norm = 0.0, 0.0
        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):
            # saving single perturbations
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            mse_added = mean_squared_error(_tgt, _pred)

            mse += mse_added
            norm += mean_squared_error(_tgt, np.zeros_like(_tgt))
            p = self.perts_order_union[counter]
            if p not in self.each_perturbation_eval:
                self.each_perturbation_eval[p] = {}
            self.each_perturbation_eval[p][f"mse_union_{self.current_k}"] = mse_added
            counter += 1
        return mse / norm

    def norm_normalized_mse_topk_de(self) -> float:
        counter = 0
        all_norms = []
        all_mses = []
        for tgt, pred in zip(self.ranked_truth, self.ranked_pred):
            # saving single perturbations
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            mse_added = mean_squared_error(_tgt, _pred)

            p = self.perts_order[counter]
            if p not in self.each_perturbation_eval:
                self.each_perturbation_eval[p] = {}

            norm = mean_squared_error(tgt, np.zeros_like(tgt))
            self.each_perturbation_eval[p][f"norm_{self.current_k}"] = norm

            all_norms.append(norm)
            all_mses.append(mse_added)

            counter += 1

        all_mses = np.array(all_mses)
        all_norms = np.array(all_norms)

        q = np.quantile(all_norms, 0.1)
        new_mses = all_mses / (all_norms + q)

        # get error in norms
        for i in range(new_mses.shape[0]):
            p = self.perts_order[i]
            self.each_perturbation_eval[p][f"pnmse_{self.current_k}"] = new_mses[i]

        return np.mean(new_mses)

    def norm_error_unionde(self) -> float:
        all_errors = []
        counter = 0
        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            
            tgt_norm = np.linalg.norm(_tgt)
            pred_norm = np.linalg.norm(_pred)

            abs_error = abs(tgt_norm - pred_norm)
            all_errors.append(abs_error)

            p = self.perts_order_union[counter]
            self.each_perturbation_eval[p][
                f"norm_abserror_unionde_{self.current_k}"
            ] = abs_error

            counter += 1
        return np.mean(all_errors)

    def norm_rel_error_unionde(self) -> float:
        all_errors = []
        counter = 0
        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            
            tgt_norm = np.linalg.norm(_tgt)
            pred_norm = np.linalg.norm(_pred)

            abs_error = abs(tgt_norm - pred_norm)
            if pred_norm > 0:
                abs_error /= tgt_norm
            all_errors.append(abs_error)

            p = self.perts_order_union[counter]
            self.each_perturbation_eval[p][
                f"norm_relabserror_unionde_{self.current_k}"
            ] = abs_error

            counter += 1
        return np.mean(all_errors)

    def norm_normalized_mse_topk_unionde(self) -> float:
        counter = 0
        all_norms = []
        all_mses = []
        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):
            # saving single perturbations
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            mse_added = mean_squared_error(_tgt, _pred)

            p = self.perts_order_union[counter]
            if p not in self.each_perturbation_eval:
                self.each_perturbation_eval[p] = {}

            norm = mean_squared_error(_tgt, np.zeros_like(_tgt))
            self.each_perturbation_eval[p][f"norm_union_{self.current_k}"] = norm

            all_norms.append(norm)
            all_mses.append(mse_added)

            counter += 1

        all_mses = np.array(all_mses)
        all_norms = np.array(all_norms)

        q = np.quantile(all_norms, 0.1)
        new_mses = all_mses / (all_norms + q)

        # get error in norms
        for i in range(new_mses.shape[0]):
            p = self.perts_order_union[i]
            self.each_perturbation_eval[p][f"pnmse_union_{self.current_k}"] = new_mses[
                i
            ]

        return np.mean(new_mses)

    def avg_pearson_topk_de(self) -> float:
        counter = 0
        mse, norm = 0.0, 0.0
        all_cor = []
        for tgt, pred in zip(self.ranked_truth, self.ranked_pred):
            # saving single perturbations
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            all_cor.append(pearsonr(_tgt, _pred)[0])

            p = self.perts_order[counter]
            self.each_perturbation_eval[p][f"pearson_cor_{self.current_k}"] = all_cor[
                -1
            ]
            counter += 1

        return np.mean(all_cor)

    def avg_pearson_topk_unionde(self) -> float:
        counter = 0
        mse, norm = 0.0, 0.0
        all_cor = []
        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):
            # saving single perturbations
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            all_cor.append(pearsonr(_tgt, _pred)[0])

            p = self.perts_order_union[counter]
            self.each_perturbation_eval[p][f"pearson_cor_union_{self.current_k}"] = (
                all_cor[-1]
            )
            counter += 1

        return np.mean(all_cor)

    def avg_cossim_topk_de_ntcmean(self) -> float:
        all_cos = []
        counter = 0

        for tgt, pred in zip(self.ranked_truth, self.ranked_pred):
            p = self.perts_order[counter]
            
            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            single_cos = 1 - cosine(_tgt, _pred)

            self.each_perturbation_eval[p][
                f"cossim_ntcmean_{self.current_k}"
            ] = single_cos
            all_cos.append(single_cos)

            counter += 1

        return np.mean(all_cos)

    def avg_cossim_topk_unionde_ntcmean(self) -> float:
        all_cos = []
        counter = 0

        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):

            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            single_cos = 1 - cosine(_tgt, _pred)

            p = self.perts_order_union[counter]
            self.each_perturbation_eval[p][
                f"cossim_ntcmean_unionde_{self.current_k}"
            ] = single_cos
            all_cos.append(single_cos)

            counter += 1

        return np.mean(all_cos)

    def avg_cossim_topk_de(self) -> float:
        all_cos = []
        counter = 0

        mean_vec = self.pert_mean.mean(
            0
        )  # loc[self.pert_mean.index.isin(self.train_perturbation_labels),:].mean(0)

        for tgt, pred in zip(self.ranked_truth, self.ranked_pred):
            p = self.perts_order[counter]
            deg = self.feature_genes[counter]

            mean_vec_deg = mean_vec[deg].values

            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]

            tgt_temp = _tgt
            pred_temp = _pred

            temp = 1 - cosine(_tgt, _pred)
            
            _tgt = _tgt - mean_vec_deg
            _pred = _pred - mean_vec_deg

            single_cos = 1 - cosine(_tgt, _pred)

            self.each_perturbation_eval[p][f"cossim_{self.current_k}"] = single_cos
            all_cos.append(single_cos)

            counter += 1

        return np.mean(all_cos)

    def avg_cossim_topk_unionde(self) -> float:
        all_cos = []
        counter = 0
        mean_vec = self.pert_mean.mean(
            0
        )  # .loc[self.pert_mean.index.isin(self.train_perturbation_labels),:].mean(0)

        for tgt, pred in zip(self.ranked_truth_union, self.ranked_pred_union):
            mean_vec_deg = mean_vec.loc[self.deg_union].values
            
            tgt = tgt - mean_vec_deg
            pred = pred - mean_vec_deg

            mask = ~np.isnan(tgt)
            _tgt, _pred = tgt[mask], pred[mask]
            single_cos = 1 - cosine(_tgt, _pred)

            p = self.perts_order_union[counter]
            self.each_perturbation_eval[p][
                f"cossim_unionde_{self.current_k}"
            ] = single_cos
            all_cos.append(single_cos)

            counter += 1

        return np.mean(all_cos)

    def perturb_geometric_preprocess(
        self,
        use_pca: bool = True,
        n_components: int = 100,
    ) -> np.ndarray:

        # ctrl_cells = self.ctrl_cells

        if not self.only_embedding:

            deg_order = self.unfiltered_results[0].columns.values
            self.perts_order = np.array([r.index.name for r in self.unfiltered_results])
            tgt = np.array(
                [
                    r.loc[:, deg_order].loc["tgt", :].values
                    for r in self.unfiltered_results
                ]
            )
            pred = np.array(
                [
                    r.loc[:, deg_order].loc["pred", :].values
                    for r in self.unfiltered_results
                ]
            )

            train_pred = np.array(
                [
                    r.loc[:, deg_order].loc["pred", :].values
                    for r in self.train_unfiltered_results
                ]
            )
            train_tgt = np.array(
                [
                    r.loc[:, deg_order].loc["tgt", :].values
                    for r in self.train_unfiltered_results
                ]
            )

            if train_pred.shape[0] != 0:
                pred = np.concatenate([pred, train_pred], axis=0)
                tgt = np.concatenate([tgt, train_tgt], axis=0)

            self.train_pert_labels = [
                r.index.name for r in self.train_unfiltered_results
            ]
        else:
            # self.perts_order = np.array([r.index.name for r in self.unfiltered_results])

            pred, tgt = self.unfiltered_results
            self.perts_order = pred.index.values.ravel()

            pred = pred.values
            tgt = tgt.values

        # this will be used for virtual_screens
        self.tgt_pseudobulk = tgt  # .values
        self.pred_pseudobulk = pred  # .values

        if use_pca:
            n_components = min(n_components, *tgt.shape)

            if not self.only_embedding:

                # For geometric eval, define the embedding space as the sphered
                # principal components of the ground truth train and test data
                pca = PCA(n_components=n_components, random_state=0)
                tgt = pca.fit_transform(tgt)
                normalizer = SpheringTransform(
                    tgt, reg_param=0.5, reg_trace=True, gpu=False, oas=False
                )
                tgt = normalizer.normalize(tgt)
                pc_ntc = normalizer.normalize(
                    pca.transform(np.zeros((1, train_tgt.shape[1])))
                )

                pred_pca = PCA(n_components=n_components, random_state=0)
                pred = pred_pca.fit_transform(pred)
                pred_normalizer = SpheringTransform(
                    pred, reg_param=0.5, reg_trace=True, gpu=False, oas=False
                )
                pred = pred_normalizer.normalize(pred)

                self.pca = pca
                self.sphering_transform = normalizer

            else:
                if not self.both_embedding:
                    pca = PCA(n_components=n_components, random_state=0)

                    tgt = pca.fit_transform(tgt)
                    normalizer = SpheringTransform(
                        tgt, reg_param=0.5, reg_trace=True, gpu=False, oas=False
                    )
                    tgt = normalizer.normalize(tgt)
                    self.pca = pca
                    self.sphering_transform = normalizer

                # tgt = pca.transform(tgt)

        self.tgt_geom_preped = tgt
        self.pred_geom_preped = pred
        return tgt, pred

    def phenocopy_virtual_screen(
        self,
        metric="cosine",
        mads: [int] = list(np.arange(1,10,)),
    ) -> float:

        # working here
        if not self.only_embedding:
            nperturbations_tested = len(self.unfiltered_results)
        else:
            nperturbations_tested = self.unfiltered_results[0].shape[0] - len(self.train_perturbation_labels)

        tgt, pred = self.tgt_geom_preped, self.pred_geom_preped

        tgt_cos_mat = cosine_similarity(tgt)
        pred_cos_mat = cosine_similarity(pred)

        self.all_neighbors = {}
        eval_metric = {}
        recall_ks = [5, 10, 20, 50, 100]
        # training points start at n_perturbations_tested
        for m in mads:
            self.all_neighbors[m] = {}
            all_auc = []
            all_auc_global = []
            all_map = []
            nsignificant = []
            nsignificant_global = []
            all_cos_mad = []
            all_med = []
            

            all_cos = tgt_cos_mat[
                np.arange(nperturbations_tested, tgt_cos_mat.shape[0]), :]
            all_cos = all_cos.ravel()
            
            all_median = np.median(all_cos)
            all_mad = median_abs_deviation(all_cos)


            for i, tp in enumerate(
                np.arange(nperturbations_tested, tgt_cos_mat.shape[0])
            ):
                p = self.train_perturbation_labels[i]

                tgt_cos_vec = tgt_cos_mat[tp, :]
                pred_cos_vec = pred_cos_mat[tp, :]

                median_cos = np.median(tgt_cos_vec)

                cos_mad = median_abs_deviation(tgt_cos_vec)
                all_cos_mad.append(cos_mad*m)
                all_med.append(median_cos)

                binary_vec = tgt_cos_vec > median_cos + m * cos_mad
                nsignificant.append(np.sum(binary_vec))
                binary_vec_global = tgt_cos_vec > all_median + m * all_mad
                nsignificant_global.append(np.sum(binary_vec_global))
                # TODO: add caching of the neighbors for each perturbation
                # this would go in self.all_neighbors

                # only keep the test perturbations in query set
                binary_vec = binary_vec[:nperturbations_tested]
                binary_vec_global = binary_vec_global[:nperturbations_tested]
                pred_cos_vec = pred_cos_vec[:nperturbations_tested]
                if len(np.unique(binary_vec)) > 1:
                    # partial AUROC
                    auc = roc_auc_score(binary_vec, pred_cos_vec)
                    all_auc.append(auc)



                    if p not in self.each_perturbation_eval:
                        self.each_perturbation_eval[p] = {}

                    self.each_perturbation_eval[p][f"phenocopy_auroc_local_mad_{m}"] = (
                        all_auc[-1]
                    )
                    if len(np.unique(binary_vec_global)) > 1:
                        auc_global = roc_auc_score(
                            binary_vec_global, pred_cos_vec
                        )
                        all_auc_global.append(auc_global)

                        self.each_perturbation_eval[p][
                            f"phenocopy_auroc_global_mad_{m}"
                        ] = all_auc_global[-1]
                    else:
                        pass
                        #print(p,' no significant neighbors')


            eval_metric[f"phenocopy_auroc_local_mad_{m}"] = np.mean(all_auc)
            if len(all_auc_global) > 0:
                eval_metric[f"phenocopy_auroc_global_mad_{m}"] = np.mean(all_auc_global)
            
        
        for k in recall_ks:
            all_recall = []
            for i, tp in enumerate(
                np.arange(nperturbations_tested, tgt_cos_mat.shape[0])
            ):
                p = self.train_perturbation_labels[i]

                tgt_cos_vec = tgt_cos_mat[tp, :]
                pred_cos_vec = pred_cos_mat[tp, :]

                tgt_cos_vec = tgt_cos_vec[:nperturbations_tested]
                pred_cos_vec = pred_cos_vec[:nperturbations_tested]

                # Calculate recall using tgt_cos_vec and pred_cos_vec
                tgt_topk = np.argsort(tgt_cos_vec)[-k:]
                pred_topk = np.argsort(pred_cos_vec)[-k:]

                recall = len(set(tgt_topk).intersection(set(pred_topk))) / len(tgt_topk)
                all_recall.append(recall)

                self.each_perturbation_eval[p][f"phenocopy_recall_{k}"] = all_recall[-1]
            eval_metric[f"phenocopy_recall_{k}"] = np.mean(all_recall)
        
        return eval_metric

    def similarity_knn(
        self,
        n_neighbors=10,
        metric="cosine",
    ) -> float:

        if not self.only_embedding:
            nperturbations_tested = len(self.unfiltered_results)
        else:
            nperturbations_tested = self.unfiltered_results[0].shape[0] - len(self.train_perturbation_labels)

        tgt, pred = self.tgt_geom_preped, self.pred_geom_preped

        n_neighbors = min(n_neighbors, tgt.shape[0] - 2)

        tgt_knn = kneighbors_graph(
            tgt,
            metric=metric,
            n_neighbors=(n_neighbors + 1),
            mode="connectivity",
        ).toarray()

        pred_knn = kneighbors_graph(
            pred,
            metric=metric,
            n_neighbors=(n_neighbors + 1),
            mode="connectivity",
        ).toarray()

        tgt_knn = tgt_knn[:nperturbations_tested, :]
        pred_knn = pred_knn[:nperturbations_tested, :]

        all_ap_score = []
        all_auroc = []
        all_recall = []
        all_partial_auroc = []
        nperturbations_tested = 0
        for i in range(tgt_knn.shape[0]):
            tgtk = tgt_knn[i, :].astype(int)
            predk = pred_knn[i, :].astype(int)

            # remove self perturbation
            tgtk = np.delete(tgtk, i)
            predk = np.delete(predk, i)

            if len(np.unique(tgtk)) > 1:
                all_partial_auroc.append(roc_auc_score(tgtk, predk, max_fpr=0.05))
                all_auroc.append(roc_auc_score(tgtk, predk))
                all_ap_score.append(average_precision_score(tgtk, predk))
                all_recall.append(recall_score(tgtk, predk))

                nperturbations_tested += 1

                p = self.perts_order[i]
                if p not in self.each_perturbation_eval:
                    self.each_perturbation_eval[p] = {}

                self.each_perturbation_eval[p][f"geom_avgp_{self.current_k}"] = (
                    all_ap_score[-1]
                )
                self.each_perturbation_eval[p][f"geom_recall_{self.current_k}"] = (
                    all_recall[-1]
                )
                self.each_perturbation_eval[p][
                    f"geom_partial_auroc_{self.current_k}"
                ] = all_partial_auroc[-1]

        if len(all_ap_score) == 0:
            all_ap_score = [0]
            all_auroc = [0]
            all_recall = [0]
            all_partial_auroc = [0]

        eval_metric = {
            "geom_similarity_knn_auroc": np.mean(all_auroc),
            "geom_similarity_knn_avgp": np.mean(all_ap_score),
            "geom_similarity_knn_recall": np.mean(all_recall),
            "geom_similarity_partial_auroc": np.mean(all_partial_auroc),
            "geom_similarity_knn_nperturbations_test": nperturbations_tested,
        }

        return eval_metric

    def virtual_screen(
        self,
        k=10,
        n_pcs=50,
        n_neighbors=10,
        resolution=0.5,
    ):

        if os.path.isfile(self.geneset_file):
            with open(self.geneset_file, "r") as f:
                genesets = json.load(f)
        else:
            return {}

        # results[0].columns is so that all perturbations have genes in the same order

        all_unfiltered_results = self.unfiltered_results + self.train_unfiltered_results
        pseudobulk_gt = np.array(
            [
                r.loc["tgt", all_unfiltered_results[0].columns].values.ravel()
                for r in all_unfiltered_results
            ]
        )

        pert_names = np.array([r.index.name for r in all_unfiltered_results])

        pseudobulk_pred = np.array(
            [
                r.loc["pred", all_unfiltered_results[0].columns].values.ravel()
                for r in all_unfiltered_results
            ]
        )

        pseudobulk_gt = pd.DataFrame(
            pseudobulk_gt, columns=all_unfiltered_results[0].columns, index=pert_names
        )
        adata_gt = sc.AnnData(pseudobulk_gt)

        pseudobulk_pred = pd.DataFrame(
            pseudobulk_pred,
            columns=all_unfiltered_results[0].columns,
            index=pert_names,
        )
        adata_pred = sc.AnnData(pseudobulk_pred)

        test_pert_names = np.array([r.index.name for r in self.unfiltered_results])

        # if there are fewer perturbations in this test set than k
        if len(test_pert_names) <= k:
            return {}

        if not hasattr(self, "ground_truth_virtual_screen_perts"):
            self.ground_truth_virtual_screen_perts = {}
        self.ground_truth_virtual_screen_perts[k] = {}
        scores = {}
        for group, genes in genesets.items():
            genes = np.array(genes)[np.isin(genes, adata_pred.var.index)]
            sc.tl.score_genes(adata_gt, genes, score_name=f"geneset_{group}")
            sc.tl.score_genes(adata_pred, genes, score_name=f"geneset_{group}")
            vals, pred_scores, gt_scores, pos_neg_perturbations = (
                get_virtual_screen_scores(
                    adata_pred, adata_gt, k, f"geneset_{group}", test_pert_names
                )
            )

            self.ground_truth_virtual_screen_perts[k][group] = pos_neg_perturbations

            self.each_perturbation_eval[f"virtual_screen_scores_{group}"] = pred_scores
            self.each_perturbation_eval[f"gt_virtual_screen_scores_{group}"] = gt_scores

            for s, val in vals.items():
                scores["genesets_" + group + "_" + s] = val

        return scores


class Evaluator:
    """Orchestrate calls to eval functions.

    Args:
        var_names: expected gene names for the columns in predictions and targets
        degs: a dictionary mapping perturbation to list of strings identifying
            differentially expressed genes for that perturbation, in descending
            order of significance. See other notes below.
        kvals: values of k at which to evaluate topk metrics.
        include_version: if True, include version info along with metric values.

    Example:
        evaluator = Evaluator(var_names, degs)
        eval_dict = evaluator(pred_keys, tgts, preds)

    Note:
        The `degs` argument is expected to correspond to others in the following ways:
            1) All elements of the `pert_keys` argument to `__call__` must be found in `degs.keys()`.
            2) The elements of the lists in `degs.values()` are expected to match the elements
            of `var_names`, though `degs.values()` is allowed to contain extra genes which may have
            been subsequently removed from modeling.
    """

    def __init__(
        self,
        var_names: Iterable[str],
        degs: Mapping[str, Iterable[str]],
        ctrl_cells: sc.AnnData,
        train_splits=np.array,
        single_cells: sc.AnnData = None,
        kvals: Tuple[int] = (5, 10, 20, 50, 100, 200, 1000, 2000),
        only_normalized_mse: bool = False,
        include_version: bool = True,
        geneset_file: str = "None",
        perturbation_cluster_file: str = "None",
        ncells_per_perturbation_file: str = "None",
        dataset: str = "None",
        seed: str = "None",
    ):

        self.var_names = var_names

        self.degs = degs

        self.ctrl_cells = ctrl_cells.X
        self.single_cells = single_cells

        self.train_perturbation_labels = train_splits

        self.only_normalized_mse = only_normalized_mse

        self.kvals = kvals
        self.kvals_cached = kvals

        self.include_version = include_version

        self.geneset_file = geneset_file
        self.perturbation_cluster_file = perturbation_cluster_file
        self.ncells_per_perturbation_file = ncells_per_perturbation_file
        self.dataset = dataset
        self.seed = seed

        feature_genes = self.single_cells.var.index
        grouped_adata = self.single_cells.obs.groupby("perturbation_group")
        perts = [pert_obs[0] for pert_obs in grouped_adata]

        self.pert_mean = np.array(
            [
                np.mean(self.single_cells[pert_obs[1].index].X, axis=0)
                for pert_obs in grouped_adata
            ]
        )
        self.pert_mean = pd.DataFrame(
            self.pert_mean, index=perts, columns=feature_genes
        )

        self.pert_mean = self.pert_mean.loc[
            self.pert_mean.index.isin(self.train_perturbation_labels), :
        ]

    def _package_result(self, pert, tgt_vector, pred_vector):
        # Package predictions for one perturbation into a two-row dataframe
        # with columns sorted by ground-truth DE for the perturbation

        result = pd.DataFrame(
            data=np.vstack([tgt_vector, pred_vector]),
            index=pd.Index(["tgt", "pred"], name=pert),
            columns=self.var_names,
        )

        # filter degs in case original DEG computation included
        # genes that have since been removed
        # also filter results without degs
        degs = [x for x in self.degs[result.index.name] if x in result]

        return result[degs], result

    def __call__(
        self,
        pert_keys: Iterable[str],
        tgts: NDArray,
        preds: NDArray,
        test: bool,
        only_embedding: bool = False,
        both_embedding: bool = False,
    ) -> Mapping[str, float]:
        self.only_embedding = only_embedding
        self.both_embedding = both_embedding

        if test:
            self.kvals = self.kvals_cached
        else:
            self.kvals = (20, 200)

        training_datapoints = np.array(self.train_perturbation_labels)

        all_evals = {}
        self.all_single_evals = {}
        all_eval_dfs = []
        evals = self.evaluate(pert_keys, tgts, preds, test)
        if hasattr(self.eval_suite, "ground_truth_virtual_screen_perts"):
            self.ground_truth_virtual_screen_perts = (
                self.eval_suite.ground_truth_virtual_screen_perts
            )

        all_evals.update(evals)
        # single perturbation evals
        self.all_single_evals.update(self.each_perturbation_eval)
        self.eval_suite.evals_df["cluster"] = "all"
        # all_eval_dfs.append(self.eval_suite.evals_df)

        temp_eval_suite_df = None
        if test:
            p_clusts = {"all": pert_keys}
            # stratification by perturbation effect magnitutde
            if (self.ncells_per_perturbation_file != "None") and (
                os.path.isfile(self.ncells_per_perturbation_file)
            ):
                self.compute_norms(pert_keys, preds, tgts)
                p_clusts["perturbations_with_effect"] = self.perturbations_with_effect
                norm_auroc, norm_avgp = self.evaluate_norm_prediction(pert_keys)
                self.eval_suite.evals_df.loc[self.eval_suite.evals_df.shape[0], :] = [
                    "effect_prediction",
                    "na",
                    "na",
                    "norm_auroc",
                    norm_auroc,
                    "all",
                ]
                self.eval_suite.evals_df.loc[self.eval_suite.evals_df.shape[0], :] = [
                    "effect_prediction",
                    "na",
                    "na",
                    "norm_avgp",
                    norm_avgp,
                    "all",
                ]
                evals["effect_prediction_norm_auroc"] = norm_auroc
                evals["effect_prediction_norm_avgp"] = norm_avgp

                # overwrite these
                all_eval_dfs = [self.eval_suite.evals_df]
                all_evals = evals

            # stratification by clusters
            if os.path.isfile(self.perturbation_cluster_file):
                with open(self.perturbation_cluster_file, "r") as f:
                    perturbation_clusts = json.load(f)

                for clust, perts in perturbation_clusts.items():
                    p_clusts[clust] = perts
                    perts = np.array(perts)
                    print(clust, len(perts))

                    perts = perts[np.isin(perts, p_clusts["perturbations_with_effect"])]
                    p_clusts["perturbations_with_effect;" + clust] = list(perts)

            temp_eval_suite_df = self.eval_suite.evals_df[
                self.eval_suite.evals_df.metric_category.isin(
                    ["virtual_screen", "effect_prediction", "phenocopy"]
                )
            ]
            self.eval_suite.evals_df = pd.DataFrame(
                columns=self.eval_suite.evals_df.columns
            )

            for clust, perts in p_clusts.items():
                print(clust)
                # used to subset train
                saved_perts = perts.copy()

                # ensure the perturbations from a cluster are not in the training data
                perts = np.array(perts)[
                    np.isin(perts, training_datapoints, invert=True)
                ]
                
                subset_single_evals = {}
                for p in perts:
                    if p in self.all_single_evals:
                        subset_single_evals[p] = self.all_single_evals[p]
                subset_evals = pd.DataFrame(subset_single_evals).T

                subset_single_evals_train = {}

                for p in training_datapoints[np.isin(training_datapoints, saved_perts)]:
                    if p in self.all_single_evals:
                        subset_single_evals_train[p] = self.all_single_evals[p]
                subset_evals_train = pd.DataFrame(subset_single_evals_train).T

                subset_evals = subset_evals.mean(0)

                subset_evals_train = subset_evals_train.mean(0)

                print(subset_evals_train)
                

                # print(subset_evals)
                for m, v in subset_evals.items():
                    k = m.split("_")[-1]
                    met = "_".join(m.split("_")[:-1])
                    self.eval_suite.evals_df.loc[
                        self.eval_suite.evals_df.shape[0], :
                    ] = ["cluster", "na", k, met, v, clust]

                    
                    if "mse" in met:
                        if met == "mse":
                            norm = subset_evals.loc[f"norm_{k}"]
                        elif met == "mse_unionde":
                            norm = subset_evals.loc[f"norm_unionde_{k}"]
                        self.eval_suite.evals_df.loc[
                            self.eval_suite.evals_df.shape[0], :
                        ] = ["cluster", "na", k, f"normalized_{met}", v / (norm), clust]
                    

                for m, v in subset_evals_train.items():
                    print(m,v)
                    k = m.split("_")[-1]
                    met = "_".join(m.split("_")[:-1])
                    self.eval_suite.evals_df.loc[
                        self.eval_suite.evals_df.shape[0], :
                    ] = ["train_cluster", "na", k, met, v, clust]

        if temp_eval_suite_df is not None:
            self.eval_suite.evals_df = pd.concat(
                [self.eval_suite.evals_df, temp_eval_suite_df]
            )

        self.eval_dfs = self.eval_suite.evals_df

        del self.results
        del self.unfiltered_results

        return all_evals

    def evaluate_norm_prediction(self, pert_keys: Iterable[str]):
        norm_pred = self.norm_pred[
            np.isin(pert_keys, self.train_perturbation_labels, invert=True)
        ]
        test_p_with_effect = self.perturbations_with_effect[
            np.isin(
                self.perturbations_with_effect,
                self.train_perturbation_labels,
                invert=True,
            )
        ]
        test_pert_keys = pert_keys[
            np.isin(pert_keys, self.train_perturbation_labels, invert=True)
        ]

        bool_vec = np.isin(test_pert_keys, test_p_with_effect)

        norm_rank = rankdata(norm_pred, method="max")
        norm_rank = norm_rank / len(norm_rank)

        if len(np.unique(bool_vec)) == 2:
            auroc = roc_auc_score(bool_vec, norm_rank)
            avgp = average_precision_score(bool_vec, norm_rank)
        else:
            auroc = np.nan
            avgp = np.nan

        return auroc, avgp

    def compute_norms(self, pert_keys: Iterable[str], preds: NDArray, tgts: NDArray):
        with open(self.ncells_per_perturbation_file, "r") as f_in:
            ncells_per_pert = json.load(f_in)

        # first ncells to get all perturbations for null
        ncells = np.array([ncells_per_pert[p] for p in ncells_per_pert])

        single_cells = self.single_cells.X

        covariance_matrix = np.cov(single_cells.T)

        norm_tgt = compute_squared_l2_norm(
            tgts, covariance_matrix, ncells, unbiased=False
        )
        norm_pred = compute_squared_l2_norm(
            preds, covariance_matrix, ncells, unbiased=False
        )

        self.norm_tgt = norm_tgt
        self.norm_pred = norm_pred

        # get null distributions for each n
        # null_distribution = get_null_distribution_each_n(self.ctrl_cells,covariance_matrix,ncells,False)
        os.makedirs("cache", exist_ok=True)
        cache_file = f"./cache/{self.dataset}_{self.seed}_null.pkl"
        if not os.path.isfile(cache_file):
            single_cells = sc.AnnData(single_cells)
            single_cells.obs = self.single_cells.obs

            null_distribution = get_null_distribution_each_n(
                single_cells,
                covariance_matrix,
                ncells,
                False,
            )
            with open(cache_file, "wb") as f_out:
                pkl.dump(null_distribution, f_out)
        else:
            with open(cache_file, "rb") as f_in:
                null_distribution = pkl.load(f_in)

        # second ncells so we only have the perturbation in this set
        ncells = np.array([ncells_per_pert[p] for p in pert_keys])

        # import matplotlib.pyplot as plt
        significant = []
        # bf_threshold = 0.05 / len(ncells)
        self.null_dist = null_distribution
        self.ncells = ncells

        pval_threshold = 5e-2
        pvals = []
        for i, n in enumerate(ncells):
            null_vec = null_distribution[n]

            bool_vec = self.norm_tgt[i] < null_vec

            signif = np.mean(bool_vec)

            if signif < pval_threshold:
                significant.append(True)
            else:
                significant.append(False)

        significant = np.array(significant)
        self.norm_tgt = np.array(self.norm_tgt)

        self.perturbations_with_effect = pert_keys[significant]

    def evaluate(
        self, pert_keys: Iterable[str], tgts: NDArray, preds: NDArray, test: bool
    ) -> Mapping[str, float]:
        # ensure the number of perturbations is the same
        assert len(pert_keys) == tgts.shape[0] == preds.shape[0]

        if not self.only_embedding:
            assert tgts.shape[1] == preds.shape[1]

            # Package each perturbation into a separate two-row dataframe
            if hasattr(self, "results"):
                results = self.results
                unfiltered_results = self.unfiltered_results
            else:
                results = [
                    self._package_result(x, y, z)[0]
                    for x, y, z in zip(pert_keys, tgts, preds)
                    if x in self.degs
                ]  # remove perturbations that don't have degs computed
                unfiltered_results = [
                    self._package_result(x, y, z)[1]
                    for x, y, z in zip(pert_keys, tgts, preds)
                    if x in self.degs
                ]
                self.results = results
                self.unfiltered_results = unfiltered_results

        else:
            preds = pd.DataFrame(preds, index=pert_keys)
            tgts = pd.DataFrame(tgts, index=pert_keys)
            self.results = (preds, tgts)
            self.unfiltered_results = (preds, tgts)
            results = (preds, tgts)
            unfiltered_results = (preds, tgts)

        if not self.only_embedding:
            test_functions = [
                "avg_normalized_mse_topk_de",
                "pn_mse_topk_de",
                "avg_normalized_mse_topk_unionde",
                "pn_mse_topk_unionde",
                "avg_cossim_topk_de",
                "avg_cossim_topk_unionde",
                "avg_cossim_topk_unionde_ntcmean",
                "avg_cossim_topk_de_ntcmean",
                "avg_pearson_topk_de",
                "avg_pearson_topk_unionde",
                "geom/similarity_knn",
                "phenocopy/phenocopy_virtual_screen",
                "virtual_screen",
            ]

            val_functions = [
                "avg_normalized_mse_topk_de",
                "avg_normalized_mse_topk_unionde",
                "avg_cossim_topk_de",
                "avg_cossim_topk_unionde",
            ]
        else:
            val_functions = []
            test_functions = ["geom/similarity_knn",
                "phenocopy/phenocopy_virtual_screen",
            ]

        evals = {}
        # metrics for each perturbation separately
        self.each_perturbation_eval = {}
        if test:
            self.eval_suite = EvaluationSuite(
                results,
                unfiltered_results,
                test_functions,
                self.train_perturbation_labels,
                self.ctrl_cells,
                self.pert_mean,
                self.kvals,
                self.geneset_file,
                self.perturbation_cluster_file,
                self.only_embedding,
                self.both_embedding,
            )
            self.eval_suite(train=False)
            evals = self.eval_suite.evals
            self.each_perturbation_eval.update(self.eval_suite.each_perturbation_eval)

        else:
            self.eval_suite = EvaluationSuite(
                results,
                unfiltered_results,
                val_functions,
                self.train_perturbation_labels,
                self.ctrl_cells,
                self.pert_mean,
                self.kvals,
                self.perturbation_cluster_file,
                self.only_embedding,
            )

            self.eval_suite(train=True)
            evals = self.eval_suite.evals

        return evals


def get_null_distribution_each_n(mat, covariance_matrix, ncells, unbiased, nrep=1000):
    temp_adatas = []

    for tup in mat.obs.groupby("gene"):
        # subset perturbations with a lot of cells to rebalance
        if tup[1].shape[0] > 100:
            locs = np.random.choice(tup[1].index.values.ravel(), 100, replace=False)
            temp_adatas.append(mat[locs, :])
        else:
            temp_adatas.append(mat[tup[1].index, :])

    mat = sc.concat(temp_adatas).X

    uniq_ncells = np.unique(ncells)
    null_vecs = {}
    counter = 0
    for n in uniq_ncells:
        print(n, counter / len(uniq_ncells))
        counter += 1

        subsampled_mat = np.zeros((nrep, mat.shape[1]))
        for i in range(nrep):
            locs = np.random.choice(np.arange(mat.shape[0]), n, replace=False)
            temp_mat = mat[locs, :]
            subsampled_mat[i, :] = np.mean(temp_mat, axis=0)
        ncells = np.array([n for _ in range(nrep)])
        squared_l2_norm = compute_squared_l2_norm(
            subsampled_mat, covariance_matrix, ncells, unbiased=unbiased
        )

        null_vecs[n] = squared_l2_norm

    return null_vecs


def compute_squared_l2_norm(mat, covariance_matrix, ncells, unbiased):
    """
    Calculate the unbiased L2 norm of the mean of samples.

    Parameters:
    samples (numpy.ndarray): A 2D array where each column represents a sample.
    covariance_matrix (numpy.ndarray): The covariance matrix of the samples.

    Returns:
    float: The unbiased L2 norm of the mean of the samples.
    """

    # Biased L2 norm of the mean
    biased_l2_norm = np.linalg.norm(mat, axis=1)

    if not unbiased:
        return biased_l2_norm

    # Trace of the covariance matrix
    trace_covariance = np.trace(covariance_matrix)

    # Unbiased L2 norm of the mean
    unbiased_l2_norm_squared = biased_l2_norm**2 - (1 / ncells) * trace_covariance

    # Ensure the result is non-negative before taking the square root
    # unbiased_l2_norm_squared = np.array([max(i, 0) for i in unbiased_l2_norm_squared])

    # unbiased_l2_norm = np.sqrt(unbiased_l2_norm_squared)

    return np.sqrt(unbiased_l2_norm_squared)  # unbiased_l2_norm


def get_virtual_screen_scores(
    adata_pred: sc.AnnData,
    adata_gt: sc.AnnData,
    k: int,
    group: str,
    test_pert_names: list,
):

    gt_scores = adata_gt.obs.loc[:, group].sort_values(ascending=False)
    pred_scores = adata_pred.obs.loc[:, group]  # .sort_values(ascending=False)
    pred_scores = pred_scores.loc[gt_scores.index]

    train_gt_scores = gt_scores.loc[~gt_scores.index.isin(test_pert_names)]
    gt_scores = gt_scores[test_pert_names]
    pred_scores = pred_scores[test_pert_names]

    cor = spearmanr(pred_scores.values.ravel(), gt_scores.values.ravel())

    scores = {}
    scores["spearmanr"] = cor[0]

    mad = median_abs_deviation(train_gt_scores)
    median_score = np.median(train_gt_scores)

    # perturbations with best geneset score
    binary_vec = gt_scores > median_score + k * mad  # np.zeros(gt_scores.shape[0])
    positive_perturbations = binary_vec[binary_vec].index.values.ravel()

    # binary_vec[:k] = 1
    if len(np.unique(binary_vec)) != 1:
        scores["positive_auroc"] = roc_auc_score(binary_vec, pred_scores)
        scores["positive_avgp"] = average_precision_score(binary_vec, pred_scores)

    # perturbations with lowest geneset score
    pred_scores = pred_scores.loc[gt_scores.index]
    binary_vec = gt_scores < median_score - k * mad
    negative_perturbations = binary_vec[binary_vec].index.values.ravel()
    pred_scores *= -1

    # binary_vec[-1*k:] = 1
    # pred_scores = pred_scores * -1
    if len(np.unique(binary_vec)) != 1:
        scores["negative_auroc"] = roc_auc_score(binary_vec, pred_scores)
        scores["negative_avgp"] = average_precision_score(binary_vec, pred_scores)


    return (
        scores,
        pred_scores,
        gt_scores,
        (positive_perturbations, negative_perturbations),
    )


def resource_path(filename: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources", filename
    )


@cache
def load_replogle_gene_sets() -> pd.DataFrame:
    """
    Return a DataFrame with gene_name as index and
    one column, gene_set, which contains an integer assignment
    of each gene to a gene set.
    """
    with open(resource_path("replogle_gene_set_markers.json")) as f:
        gene_sets = json.load(f)
    return (
        pd.Series(
            {gene: set_id for set_id, genes in gene_sets.items() for gene in genes},
            name="gene_name",
        )
        .astype(int)
        .to_frame("gene_set")
    )
