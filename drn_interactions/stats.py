import numpy as np
import sklearn.metrics
from scipy.stats import mannwhitneyu
import pandas as pd


def p_adjust(pvalues: np.ndarray, method: str = "Benjamini-Hochberg"):
    """
    Adjust p values for multiple comparisons using various methods
    Args:
        pvalues: A numpy array of pvalues from various comparisons
        method: The p value correction method. Set of availible methods
                comprise {'Bonferroni', 'Bonferroni-Holm', 'Benjamini-Hochberg'}
    Returns:
        A numpy array of adjusted pvalues
    """

    n = pvalues.shape[0]
    new_pvalues = np.empty(n)

    if method == "Bonferroni":
        new_pvalues = n * pvalues

    elif method == "Bonferroni-Holm":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            new_pvalues[i] = (n - rank) * pvalue

    elif method == "Benjamini-Hochberg":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n / rank) * pvalue)
        for i in range(0, int(n) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]

    return new_pvalues


def auc(arr, to_1=True):
    if to_1:
        x = np.linspace(0, 1, len(arr))
    else:
        x = np.arange(len(arr))
    return sklearn.metrics.auc(x, arr)


def prop_of_total(arr):
    return arr / np.sum(arr)


def mannwhitneyu_plusplus(x, y, names=("x", "y"), compare_f=None):
    if compare_f is None:
        compare_f = mannwhitneyu
    out = {}
    out[f"n_{names[0]}"] = len(x)
    out[f"n_{names[1]}"] = len(y)
    out[f"Mean_{names[0]}"] = np.mean(x)
    out[f"Mean_{names[1]}"] = np.mean(y)
    out["Diff"] = np.mean(y) - np.mean(x)
    out["U"], out["p"] = compare_f(x, y)
    return pd.Series(out)


def se_mean(arr):
    return np.std(arr) / np.sqrt(len(arr))


def se_prop(arr: np.ndarray):
    p = arr.mean()
    return np.sqrt((p * (1 - p)) / len(arr))
