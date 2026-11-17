"""Forward selection of base features by out-of-fold log-loss.

A base feature enters with all its windows at once. At each step the feature whose addition gives
the lowest log-loss, summed over the folds (each predicted by a model fitted on the others), is
kept. Training and test frames are subsampled by `stride` (frames are autocorrelated, so this
loses little and is `stride` times faster).
"""

import numpy as np

from .model import EPS, Logistic


def fold_masks(n, folds):
    masks = []
    for start, end in folds:
        mask = np.zeros(n, dtype=bool)
        mask[start:end + 1] = True
        masks.append(mask)
    return masks


def log_loss(columns, blocks, rows, target, folds, stride, l2):
    x = np.hstack([blocks[name] for name in columns])
    masks = fold_masks(len(target), folds)
    inside = np.any(masks, axis=0)
    total, count = 0.0, 0
    for held in masks:
        train = np.flatnonzero(rows & inside & ~held)[::stride]
        test = np.flatnonzero(rows & held)[::stride]
        p = np.clip(Logistic(l2).fit(x[train], target[train]).predict(x[test]), EPS, 1 - EPS)
        y = target[test]
        total -= np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        count += len(test)
    return total / count


def forward(blocks, rows, target, folds, steps, stride, l2):
    """[(feature, log-loss after adding it)] in the order chosen."""
    chosen = []
    for _ in range(steps):
        loss, name = min((log_loss([c for c, _ in chosen] + [name], blocks, rows, target, folds,
                                   stride, l2), name)
                         for name in sorted(blocks) if name not in dict(chosen))
        chosen.append((name, loss))
    return chosen
