"""Dead-ball detector from tracking only, trained on the D labels of the Metrica games.

When the ball dies, players stop running, the ball stops (or leaves the tracked area) and usually
lies near a line (throw-in, corner, goal kick). Features per frame, each at a short and a long
smoothing window, standardised per match over its graded frames:

  speed        mean speed of the tracked players (m/s)
  ball_seen    1 where the ball is tracked
  ball_speed   ball speed where tracked in both frames of the lag (m/s)
  ball_gap     log(1 + frames since the ball was last tracked)
  line_dist    distance of the last known ball position to the nearest pitch line (m)
  spread       log of the spread (std of x and y) of all tracked players
"""

import warnings

import numpy as np

from .cues import _moving_average, frozen_ball

FEATURES = ("speed", "ball_seen", "ball_speed", "ball_gap", "line_dist", "spread")


def raw_features(players, ball, pitch, lag, fps):
    n = len(players)
    dt = lag / float(fps)
    step = np.full_like(players, np.nan)
    step[lag:] = players[lag:] - players[:-lag]
    out = {}
    with warnings.catch_warnings():   # all-NaN frames give NaN, which is what we want
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.sqrt(np.sum(step ** 2, axis=2)) / dt
        out["speed"] = np.nanmean(speed, axis=1)
        out["ball_seen"] = (~np.isnan(ball[:, 0])).astype(np.float64)
        bstep = np.full_like(ball, np.nan)
        bstep[lag:] = ball[lag:] - ball[:-lag]
        out["ball_speed"] = np.sqrt(np.sum(bstep ** 2, axis=1)) / dt
        frozen, gap = frozen_ball(ball)
        out["ball_gap"] = np.log1p(gap.astype(np.float64))
        length, width = pitch
        out["line_dist"] = np.clip(np.minimum.reduce([frozen[:, 0], length - frozen[:, 0],
                                                      frozen[:, 1], width - frozen[:, 1]]), 0, None)
        sx = np.nanstd(players[:, :, 0], axis=1)
        sy = np.nanstd(players[:, :, 1], axis=1)
        out["spread"] = np.log(np.maximum(sx * sy, 1e-3))
    return {k: v.astype(np.float64) for k, v in out.items()}


def feature_matrix(raw, graded, windows):
    """(n, len(FEATURES) * len(windows)) smoothed, standardised; NaN replaced by 0 (the mean)."""
    cols = []
    for name in FEATURES:
        for w in windows:
            v = _moving_average(raw[name], w)
            ref = v[graded & ~np.isnan(v)]
            mean, std = float(ref.mean()), float(ref.std()) or 1.0
            cols.append(np.nan_to_num((v - mean) / std, nan=0.0))
    return np.stack(cols, axis=1)


def column_names(windows):
    return ["{}_w{}".format(f, w) for f in FEATURES for w in windows]
