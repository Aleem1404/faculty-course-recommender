from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
import scipy.stats as stats


def interpret_kappa(kappa: float) -> str:
    """Interprets Cohen's Kappa following the Landis & Koch (1977) benchmark."""
    if kappa < 0.0:
        return "Poor (Less than chance)"
    elif kappa <= 0.20:
        return "Slight Agreement"
    elif kappa <= 0.40:
        return "Fair Agreement"
    elif kappa <= 0.60:
        return "Moderate Agreement"
    elif kappa <= 0.80:
        return "Substantial Agreement"
    else:
        return "Almost Perfect Agreement"


def cohens_kappa(
    rater1: Sequence[int],
    rater2: Sequence[int],
    categories: Sequence[int] | None = None,
    weights: str = "unweighted",
) -> dict[str, Any]:
    """Computes Cohen's Kappa coefficient between two raters.
    
    Supports 'unweighted', 'linear', and 'quadratic' weights.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater sequences must be of identical length.")
    if len(rater1) == 0:
        return {
            "kappa": 0.0,
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "interpretation": "Insufficient Data",
            "weights": weights,
        }

    r1 = np.asarray(rater1, dtype=int)
    r2 = np.asarray(rater2, dtype=int)

    if categories is None:
        unique_cats = np.unique(np.concatenate([r1, r2]))
        unique_cats.sort()
    else:
        unique_cats = np.asarray(sorted(set(categories)), dtype=int)

    k = len(unique_cats)
    if k <= 1:
        return {
            "kappa": 1.0,
            "observed_agreement": 1.0,
            "expected_agreement": 1.0,
            "interpretation": "Almost Perfect Agreement",
            "weights": weights,
        }

    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    n = len(r1)

    # Build confusion matrix
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(r1, r2):
        if a in cat_to_idx and b in cat_to_idx:
            cm[cat_to_idx[a], cat_to_idx[b]] += 1.0

    cm /= float(n)
    p_row = cm.sum(axis=1)
    p_col = cm.sum(axis=0)
    expected = np.outer(p_row, p_col)

    # Weight matrix
    weight_mat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                weight_mat[i, j] = abs(i - j) / float(k - 1)
            elif weights == "quadratic":
                weight_mat[i, j] = ((i - j) ** 2) / float((k - 1) ** 2)
            else:  # unweighted
                weight_mat[i, j] = 0.0 if i == j else 1.0

    po_disagreement = float(np.sum(weight_mat * cm))
    pe_disagreement = float(np.sum(weight_mat * expected))

    if pe_disagreement == 0:
        kappa = 1.0
    else:
        kappa = 1.0 - (po_disagreement / pe_disagreement)

    observed_agreement = float(np.trace(cm))
    expected_agreement = float(np.sum(p_row * p_col))

    return {
        "kappa": round(float(kappa), 6),
        "observed_agreement": round(float(observed_agreement), 6),
        "expected_agreement": round(float(expected_agreement), 6),
        "interpretation": interpret_kappa(kappa),
        "weights": weights,
        "sample_size": int(n),
        "category_count": int(k),
    }


def fleiss_kappa(
    rating_matrix: np.ndarray | Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Computes Fleiss' Kappa for inter-rater agreement with arbitrary fixed raters.
    
    Parameters
    ----------
    rating_matrix : Array-like of shape (N, k) where N is subjects and k is categories.
                    rating_matrix[i, j] is the count of raters who assigned subject i to category j.
    """
    mat = np.asarray(rating_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError("rating_matrix must be a non-empty 2D array (subjects x categories).")

    n_subjects, n_categories = mat.shape
    n_raters = mat.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        raise ValueError("Each subject must be evaluated by the exact same number of raters.")

    m = n_raters[0]
    if m <= 1:
        raise ValueError("Fleiss Kappa requires at least 2 raters per subject.")

    # P_j = overall proportion of all assignments to category j
    p_j = mat.sum(axis=0) / (n_subjects * m)
    p_e = float(np.sum(p_j**2))

    # P_i = degree of agreement for subject i
    p_i = (np.sum(mat**2, axis=1) - m) / (m * (m - 1))
    p_o = float(np.mean(p_i))

    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "kappa": round(float(kappa), 6),
        "observed_agreement": round(float(p_o), 6),
        "expected_agreement": round(float(p_e), 6),
        "interpretation": interpret_kappa(kappa),
        "subjects": int(n_subjects),
        "categories": int(n_categories),
        "raters_per_subject": int(m),
    }


def bootstrap_confidence_interval(
    values: Sequence[float],
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Computes non-parametric bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_err": 0.0,
            "confidence_level": confidence_level,
        }

    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_indices = rng.integers(0, n, size=(num_bootstraps, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    std_err = float(np.std(boot_means))

    return {
        "mean": round(float(np.mean(arr)), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "std_err": round(std_err, 6),
        "confidence_level": float(confidence_level),
    }


def paired_significance_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Computes paired statistical significance tests (Wilcoxon and Paired t-test)
    between two ranking models across the same evaluation queries.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Score sequences must have identical length for paired test.")
    if len(a) < 2:
        return {
            "mean_a": float(np.mean(a)) if len(a) > 0 else 0.0,
            "mean_b": float(np.mean(b)) if len(b) > 0 else 0.0,
            "mean_diff": 0.0,
            "is_significant": False,
            "p_value": 1.0,
            "test_used": "none",
        }

    diff = b - a
    mean_diff = float(np.mean(diff))
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    # Effect size: Cohen's d for paired samples
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    # Normality test on paired differences
    is_normal = False
    shapiro_p = 0.0
    if len(diff) >= 3 and np.std(diff) > 1e-12:
        try:
            stat, shapiro_p = stats.shapiro(diff)
            is_normal = bool(shapiro_p > 0.05)
        except Exception:
            is_normal = False

    # Paired Student t-test
    ttest_p = 1.0
    ttest_stat = 0.0
    if np.std(diff) > 1e-12:
        try:
            t_res = stats.ttest_rel(b, a)
            ttest_stat = float(t_res.statistic)
            ttest_p = float(t_res.pvalue)
        except Exception:
            pass

    # Wilcoxon signed-rank test (non-parametric)
    wilcoxon_p = 1.0
    wilcoxon_stat = 0.0
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) >= 2:
        try:
            w_res = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except Exception:
            pass
    elif len(non_zero_diff) == 0:
        wilcoxon_p = 1.0

    # Choose primary test based on normality & sample size
    primary_test = "paired_ttest" if is_normal else "wilcoxon"
    primary_p = ttest_p if is_normal else wilcoxon_p
    is_significant = bool(primary_p < alpha)

    return {
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
        "mean_difference": round(mean_diff, 6),
        "percentage_gain": round(float(((mean_b - mean_a) / max(1e-6, mean_a)) * 100.0), 2),
        "shapiro_p_value": round(float(shapiro_p), 6),
        "diff_is_normal": bool(is_normal),
        "ttest_statistic": round(ttest_stat, 6),
        "ttest_p_value": round(ttest_p, 6),
        "wilcoxon_statistic": round(wilcoxon_stat, 6),
        "wilcoxon_p_value": round(wilcoxon_p, 6),
        "primary_test": primary_test,
        "primary_p_value": round(primary_p, 6),
        "cohens_d": round(cohens_d, 6),
        "is_significant": bool(is_significant),
        "significance_level": float(alpha),
    }
