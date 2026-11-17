"""Two L2 logistic regressions and a 3-state Viterbi path over their log-probabilities.

  owner  P(H | alive), fitted on frames labelled H or A
  dead   P(D), fitted on every graded frame

Per frame, log P(H) = log(1 - p_dead) + log p_home, log P(A) = log(1 - p_dead) + log(1 - p_home)
and log P(D) = log p_dead. The path over H, A and D maximises the sum of those log-probabilities
minus `team` per change H<->A and `dead` per stoppage. A stoppage is entered once and left once,
so only the sum of the two costs matters; each is `dead` / 2. Frames are autocorrelated, so the
per-frame log-probabilities overcount the evidence; the costs absorb that scale, and only their
ratio to the log-probabilities matters.
"""

import numpy as np

from .features import fg02_stats

H, A, D = 0, 1, 2
EPS = 1e-4  # probabilities are clipped to [EPS, 1 - EPS] before the log


class Logistic:
    """Standardise, then an L2 logistic regression (fg02 `stats.logistic`, Newton steps)."""

    def __init__(self, l2):
        self.l2 = l2

    def fit(self, x, y):
        self.scaler = fg02_stats.Standardizer().fit(x)
        self.weights = fg02_stats.logistic(self.scaler.transform(x), y.astype(np.float64), self.l2)
        return self

    def predict(self, x):
        return fg02_stats.predict_proba(self.weights, self.scaler.transform(x))

    def to_dict(self, columns):
        return {"l2": self.l2, "columns": list(columns), "intercept": float(self.weights[0]),
                "weights": self.weights[1:].tolist(), "mean": self.scaler.mean.tolist(),
                "std": self.scaler.std.tolist()}

    @classmethod
    def from_dict(cls, data):
        model = cls(data["l2"])
        model.scaler = fg02_stats.Standardizer()
        model.scaler.mean = np.array(data["mean"])
        model.scaler.std = np.array(data["std"])
        model.weights = np.concatenate(([data["intercept"]], data["weights"]))
        return model


def log_probs(p_home, p_dead):
    """(n, 3) log-probabilities of H, A and D."""
    p_home = np.clip(p_home, EPS, 1 - EPS)
    p_dead = np.clip(p_dead, EPS, 1 - EPS)
    alive = np.log(1 - p_dead)
    return np.column_stack([alive + np.log(p_home), alive + np.log(1 - p_home), np.log(p_dead)])


def viterbi(logp, team, dead):
    """State per frame (H, A or D) of the best path through `logp`."""
    half = dead / 2
    cost = ((0.0, team, half), (team, 0.0, half), (half, half, 0.0))
    rows = logp.tolist()
    n = len(rows)
    back = bytearray(3 * n)  # [3t + s] = state at t - 1 on the best path into state s at t
    score = list(rows[0])
    for t in range(1, n):
        new = []
        for to in range(3):
            best, arg = score[0] - cost[0][to], 0
            for came in (1, 2):
                value = score[came] - cost[came][to]
                if value > best:
                    best, arg = value, came
            back[3 * t + to] = arg
            new.append(best + rows[t][to])
        score = new
    path = np.empty(n, dtype=np.int8)
    state = max(range(3), key=score.__getitem__)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path
