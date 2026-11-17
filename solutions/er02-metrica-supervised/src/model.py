"""Pseudo-labels, logistic regression and the validation that needs no human label.

Pseudo-label (algorithmic, as RULES.md allows): on a frame where the tracked ball is within
`max_dist` of a player, and every player of the other team is at least `margin` further away than
that player, the label is that player's team. No person decides any of them.

The ball-free cues are fitted to predict these labels. Because the cues never see the ball, the
out-of-fold accuracy is also their accuracy on frames where the ball is missing, provided the
pseudo-labelled frames are representative (they are biased towards clear, close control).

The ball cue is NOT a training feature: it would read the label off the same ball. Its weight and
decay are chosen on a hide-the-ball simulation instead: for each pseudo-labelled frame, pretend
the ball was last seen some time earlier, freeze it there, and ask which combination of the frozen
ball and the ball-free logit best recovers the pseudo-label.
"""

import numpy as np

from .cues import frozen_ball, nearest_team_code
from .tracking import HOME


def pseudo_labels(players, ball, team, max_dist, margin):
    """(y, keep): y = 1 for H and 0 for A on frames where one player clearly holds the ball."""
    dist = np.sqrt(np.sum((players - ball[:, None, :]) ** 2, axis=2))
    dist = np.where(np.isnan(dist), np.inf, dist)
    home = team == HOME
    near_home = np.min(dist[:, home], axis=1)
    near_away = np.min(dist[:, ~home], axis=1)
    y = (near_home < near_away).astype(np.int8)
    nearest = np.minimum(near_home, near_away)
    other = np.maximum(near_home, near_away)
    with np.errstate(invalid="ignore"):   # inf - inf where the ball or a team is missing
        keep = np.isfinite(nearest) & (nearest <= max_dist) & (other - nearest >= margin)
    return y, keep


def fit_logistic(X, y, l2, iterations=50):
    """L2-regularised logistic regression by IRLS. The bias is not penalised. Returns (w, b)."""
    n, k = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(k + 1)
    penalty = np.full(k + 1, l2)
    penalty[-1] = 0.0
    for _ in range(iterations):
        z = A @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-9)
        grad = A.T @ (p - y) + penalty * beta
        hess = (A * w[:, None]).T @ A + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[:-1], float(beta[-1])


def _runs(mask):
    """[(start, end_exclusive)] of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def blocked_folds(n, graded, blocks_per_half):
    """Fold id per frame: each graded half is cut into `blocks_per_half` contiguous blocks. -1
    outside the graded frames."""
    fold = np.full(n, -1, dtype=np.int16)
    k = 0
    for start, end in _runs(graded):
        edges = np.linspace(start, end, blocks_per_half + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            fold[a:b] = k
            k += 1
    return fold


def _log_loss(logit, y):
    return float(np.mean(np.logaddexp(0.0, -logit) * y + np.logaddexp(0.0, logit) * (1 - y)))


def cross_validate(X, y, keep, folds, l2, purge):
    """Blocked cross-validation. Training frames within `purge` frames of the held-out block are
    dropped, because the smoothing window lets neighbouring frames share information.

    Returns accuracy and log-loss on the held-out pseudo-labels, per-fold accuracy, and the
    out-of-fold logit on every frame with complete features (NaN elsewhere)."""
    n = len(y)
    complete = np.all(~np.isnan(X), axis=1)
    oof = np.full(n, np.nan)
    fold_accuracy = []
    for f in range(int(folds.max()) + 1):
        held = folds == f
        idx = np.flatnonzero(held)
        lo, hi = max(0, idx[0] - purge), min(n, idx[-1] + purge + 1)
        near = np.zeros(n, dtype=bool)
        near[lo:hi] = True
        train = keep & ~near
        w, b = fit_logistic(X[train], y[train], l2)
        target = held & complete
        oof[target] = X[target] @ w + b
        test = held & keep
        if test.any():
            fold_accuracy.append(float(np.mean((oof[test] > 0) == (y[test] == 1))))
    scored = keep & ~np.isnan(oof)
    return {"accuracy": float(np.mean((oof[scored] > 0) == (y[scored] == 1))),
            "log_loss": _log_loss(oof[scored], y[scored]),
            "fold_accuracy": fold_accuracy, "oof_logit": oof}


def hide_ball_test(players, ball, team, y, keep, oof_logit, graded, seconds, ball_weights, taus,
                   fps, seed=0):
    """Agreement with the pseudo-labels when the ball is hidden for d seconds, for the frozen ball
    alone (solution 01, g0), the ball-free cues alone and their combination. Also picks the ball
    weight and decay (tau, frames) with the lowest log-loss when the hiding time follows the real
    distribution of ball-gap ages on graded frames."""
    frozen, gap = frozen_ball(ball)
    frames = np.flatnonzero(keep & ~np.isnan(oof_logit))
    yy = y[frames].astype(float)
    ball_free = oof_logit[frames]

    def frozen_code(delay):
        """Nearest-team code at each frame for the ball as last seen `delay` frames earlier,
        and the true age of that sighting."""
        src = np.maximum(frames - delay, 0)
        code = nearest_team_code(players[frames], frozen[src], team)
        return code, delay + gap[src]

    # Real distribution of "frames since the ball was last tracked" on graded frames without ball.
    missing = graded & np.isnan(ball[:, 0])
    ages = gap[missing]
    rng = np.random.default_rng(seed)
    delay = rng.choice(ages, size=len(frames)).astype(int)
    code, age = frozen_code(delay)
    ok = ~np.isnan(code)

    best = None
    grid = []
    for w in ball_weights:
        for tau in taus:
            logit = ball_free + np.where(ok, w * code * np.exp(-age / tau), 0.0)
            loss = _log_loss(logit, yy)
            grid.append({"ball_weight": w, "ball_tau": tau, "log_loss": loss,
                         "accuracy": float(np.mean((logit > 0) == (yy == 1)))})
            if best is None or loss < best["log_loss"]:
                best = grid[-1]

    table = []
    for s in seconds:
        d = int(round(s * fps))
        code, age = frozen_code(np.full(len(frames), d))
        ok_s = ~np.isnan(code)
        combined = ball_free + np.where(
            ok_s, best["ball_weight"] * code * np.exp(-age / best["ball_tau"]), 0.0)
        table.append({
            "seconds": s, "n": int(ok_s.sum()),
            "frozen_ball": float(np.mean((code[ok_s] > 0) == (yy[ok_s] == 1))),
            "ball_free": float(np.mean((ball_free[ok_s] > 0) == (yy[ok_s] == 1))),
            "combined": float(np.mean((combined[ok_s] > 0) == (yy[ok_s] == 1)))})
    return {"ball_weight": best["ball_weight"], "ball_tau": best["ball_tau"],
            "realistic_gaps": best, "grid": grid, "table": table}


def combined_logit(X, weights, bias, ball_code, ball_gap, ball_weight, ball_tau):
    """Ball-free logit (missing cues count as their mean, 0) plus the decaying frozen-ball term."""
    base = np.nan_to_num(X, nan=0.0) @ weights + bias
    ball = np.where(np.isnan(ball_code), 0.0,
                    ball_weight * np.nan_to_num(ball_code) * np.exp(-ball_gap / ball_tau))
    return base + ball
