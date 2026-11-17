"""Small statistics in numpy only: AUC, L2 logistic regression, PCA, k-means and NMI.

Frames are strongly autocorrelated, so none of these come with p-values. They describe how well
a signal separates labels, not whether the separation is significant.
"""

import numpy as np
import pandas as pd


def auc(positive, negative):
    """P(a positive value > a negative value), ties count half. NaN when either side is empty."""
    positive = positive[~np.isnan(positive)]
    negative = negative[~np.isnan(negative)]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([positive, negative])).rank().to_numpy()
    n_pos, n_neg = len(positive), len(negative)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class Standardizer:
    """z-score with the training mean and std; NaN becomes 0 (the training mean) afterwards."""

    def fit(self, x):
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[~(self.std > 0)] = 1.0
        self.mean = np.nan_to_num(self.mean)
        return self

    def transform(self, x):
        return np.nan_to_num((x - self.mean) / self.std)


def logistic(x, y, l2=1.0, iterations=25):
    """Weights (intercept first) of an L2 logistic regression fitted by Newton steps. x is
    already standardised, y is 0/1. The intercept is not penalised."""
    xb = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(xb.shape[1])
    penalty = np.full(xb.shape[1], l2)
    penalty[0] = 0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(xb @ w, -30, 30)))
        grad = xb.T @ (p - y) + penalty * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def predict_proba(w, x):
    return 1 / (1 + np.exp(-np.clip(w[0] + x @ w[1:], -30, 30)))


def pca(x, n_components):
    """(components, explained variance ratio) of standardised x, by SVD."""
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    variance = s ** 2
    return vt[:n_components], variance[:n_components] / variance.sum()


def kmeans(x, k, seed, iterations=100):
    """Cluster index per row, k-means++ start."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min([np.sum((x - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    centres = np.array(centres)
    assign = np.zeros(len(x), dtype=np.int64)
    for _ in range(iterations):
        d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, assign) and _ > 0:
            break
        assign = new
        for j in range(k):
            if np.any(assign == j):
                centres[j] = x[assign == j].mean(axis=0)
    return assign


def nmi(a, b):
    """Normalised mutual information (arithmetic mean of the entropies) between two labelings."""
    table = pd.crosstab(a, b).to_numpy().astype(float)
    p = table / table.sum()
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    nz = p > 0
    mi = np.sum(p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz]))
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    return 2 * mi / (ha + hb) if ha + hb > 0 else 0.0
