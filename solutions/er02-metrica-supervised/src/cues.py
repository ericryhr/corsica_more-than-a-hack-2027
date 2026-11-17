"""Per-frame possession cues. Each one is a signed scalar: positive leans H, negative leans A.

Ball-free cues (they never look at the ball, so they still work in the 41% of frames without it):

  motion   mean x-velocity of all tracked players, signed by the home attack direction. A team on
           the ball moves the whole block towards the goal it attacks.
  lanes    blocked-lane fraction of H minus that of A. A lane is a segment between two teammates
           closer than `lane_max_len`; it is blocked when an opponent is within `lane_radius` of it.
           The defending team closes passing lanes, so the team whose lanes are blocked is on the ball.
  marking  mean distance from each player to the nearest opponent standing goal-side of them, A minus
           H. Defenders mark attackers from the goal side, so a small distance means that team is
           being defended.
  shape    log area of the 10-90 percentile box of H minus that of A. The team on the ball spreads.
  height   how far each team's block sits along its own attack direction, H minus A.
  tagging  Smart Tagging phases: +1 where a phase gives the ball to H, -1 to A, 0 with no phase.

Ball cue (used at prediction time with a confidence that decays with the time since the ball was
last tracked; it is not a training feature because the pseudo-labels come from the ball):

  ball_code  team of the player closest to the ball frozen at its last known position (+1 / -1).
  ball_gap   frames since the ball was last tracked (0 where it is tracked).

Ball-free cues are computed every `cue_stride` frames and held between them.
"""

import warnings

import numpy as np

from .tracking import AWAY, DEFENDING_CODES, FPS, HOME, ATTACKING_CODES

BALL_FREE = ("motion", "lanes", "marking", "shape", "height", "tagging")


def _hold(values, stride, n_frames):
    """Repeat strided values so every frame takes the value of the sample at or before it."""
    return np.repeat(values, stride)[:n_frames]


def motion(players, sign, frames, lag, min_players):
    """Mean x-velocity (m/s) of the players visible at frames-lag and frames+lag, times the home sign."""
    n = len(players)
    before = players[np.clip(frames - lag, 0, n - 1), :, 0]
    after = players[np.clip(frames + lag, 0, n - 1), :, 0]
    vx = (after - before) * FPS / (2.0 * lag)
    count = np.sum(~np.isnan(vx), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(vx, axis=1)
    mean[count < min_players] = np.nan
    return mean * sign[frames]


def _blocked_fraction(own, opp, max_len, min_len, radius):
    """Fraction of lanes between `own` players blocked by an `opp` player. own (c, n, 2), opp (c, m, 2)."""
    i, j = np.triu_indices(own.shape[1], k=1)
    a, b = own[:, i], own[:, j]                       # (c, p, 2)
    d = b - a
    length2 = np.sum(d * d, axis=2)                   # (c, p)
    valid = (length2 <= max_len ** 2) & (length2 >= min_len ** 2)   # NaN compares False
    rel = opp[:, None, :, :] - a[:, :, None, :]        # (c, p, m, 2)
    t = np.sum(rel * d[:, :, None, :], axis=3) / np.where(valid, length2, 1.0)[:, :, None]
    t = np.clip(t, 0.0, 1.0)
    gap = rel - t[..., None] * d[:, :, None, :]
    dist2 = np.sum(gap * gap, axis=3)                 # (c, p, m), NaN where opp missing
    blocked = np.any(dist2 <= radius ** 2, axis=2) & valid
    lanes = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(lanes > 0, blocked.sum(axis=1) / np.maximum(lanes, 1), np.nan)


def lanes(players, team, frames, max_len, min_len, radius, chunk=256):
    """Blocked-lane fraction of H minus that of A."""
    home, away = np.flatnonzero(team == HOME), np.flatnonzero(team == AWAY)
    out = np.full(len(frames), np.nan, dtype=np.float32)
    for k in range(0, len(frames), chunk):
        f = frames[k:k + chunk]
        p = players[f]
        h, a = p[:, home], p[:, away]
        out[k:k + chunk] = (_blocked_fraction(h, a, max_len, min_len, radius)
                            - _blocked_fraction(a, h, max_len, min_len, radius))
    return out


def _goal_side_distance(own, opp, own_sign, cap):
    """Mean over `own` players of the distance to the nearest opponent goal-side of them.
    own (c, n, 2), opp (c, m, 2), own_sign (c,) = +1 if `own` attacks towards +x."""
    rel = opp[:, None, :, :] - own[:, :, None, :]                  # (c, n, m, 2)
    ahead = rel[..., 0] * own_sign[:, None, None] > 0              # NaN compares False
    dist = np.sqrt(np.sum(rel * rel, axis=3))
    dist = np.where(ahead, dist, np.inf)
    nearest = np.minimum(np.min(dist, axis=2), cap)                # (c, n)
    nearest = np.where(np.isnan(own[..., 0]), np.nan, nearest)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(nearest, axis=1)


def marking(players, team, sign, frames, cap, chunk=1024):
    """Goal-side marking distance of A minus that of H (positive: H is the one being marked)."""
    home, away = np.flatnonzero(team == HOME), np.flatnonzero(team == AWAY)
    out = np.full(len(frames), np.nan, dtype=np.float32)
    for k in range(0, len(frames), chunk):
        f = frames[k:k + chunk]
        p, s = players[f], sign[f].astype(np.float32)
        h, a = p[:, home], p[:, away]
        out[k:k + chunk] = (_goal_side_distance(a, h, -s, cap) - _goal_side_distance(h, a, s, cap))
    return out


def shape(players, team, frames, min_players):
    """log(area of H's 10-90 percentile box) - log(area of A's)."""
    def log_area(slots):
        p = players[frames][:, slots]
        count = np.sum(~np.isnan(p[..., 0]), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanpercentile(p, 10, axis=1)
            hi = np.nanpercentile(p, 90, axis=1)
        area = np.prod(np.maximum(hi - lo, 1.0), axis=1)
        return np.where(count >= min_players, np.log(area), np.nan)
    return log_area(np.flatnonzero(team == HOME)) - log_area(np.flatnonzero(team == AWAY))


def height(players, team, sign, frames, pitch_length):
    """Mean advance of H's players along H's attack direction minus that of A along A's."""
    p = players[frames, :, 0]
    s = sign[frames].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xh = np.nanmean(p[:, team == HOME], axis=1)
        xa = np.nanmean(p[:, team == AWAY], axis=1)
    adv_h = np.where(s > 0, xh, pitch_length - xh)
    adv_a = np.where(s > 0, pitch_length - xa, xa)
    return adv_h - adv_a


def tagging(tags, n_frames):
    """+1 / -1 / 0 per frame from the Smart Tagging phases that name the team on the ball."""
    out = np.zeros(n_frames, dtype=np.float32)
    for x in tags:
        if x["side"] is None:
            continue
        if x["code"] in ATTACKING_CODES:
            value = 1.0 if x["side"] == HOME else -1.0
        elif x["code"] in DEFENDING_CODES:
            value = -1.0 if x["side"] == HOME else 1.0
        else:
            continue
        out[x["start"]:x["end"]] += value
    return np.clip(out, -1.0, 1.0)


def nearest_team_code(players, ball, team):
    """+1 if the player closest to `ball` is home, -1 if away, NaN if the ball or all players are
    missing. players (n, s, 2), ball (n, 2)."""
    dist = np.sum((players - ball[:, None, :]) ** 2, axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    code = np.where(team[np.argmin(dist, axis=1)] == HOME, 1.0, -1.0)
    code[np.isinf(np.min(dist, axis=1))] = np.nan
    return code


def frozen_ball(ball):
    """(ball frozen at its last known position, frames since it was last known). NaN before the
    first known ball; the gap is 0 where the ball is tracked."""
    n = len(ball)
    known = ~np.isnan(ball[:, 0])
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(known, index, -1))
    frozen = np.where((last >= 0)[:, None], ball[np.maximum(last, 0)], np.nan)
    gap = np.where(last >= 0, index - last, n).astype(np.float32)
    return frozen, gap


def compute_all(players, ball, team, sign, tags, pitch, config):
    """Every cue as a per-frame array (NaN where undefined)."""
    n = len(players)
    stride = config["cue_stride"]
    frames = np.arange(0, n, stride)
    minp = config["min_players"]
    out = {
        "motion": motion(players, sign, frames, config["motion_lag"], minp),
        "lanes": lanes(players, team, frames, config["lane_max_len"], config["lane_min_len"],
                       config["lane_radius"]),
        "marking": marking(players, team, sign, frames, config["marking_cap"]),
        "shape": shape(players, team, frames, minp),
        "height": height(players, team, sign, frames, pitch[0]),
    }
    out = {k: _hold(v, stride, n).astype(np.float32) for k, v in out.items()}
    out["tagging"] = tagging(tags, n)
    frozen, gap = frozen_ball(ball)
    out["ball_code"] = nearest_team_code(players, frozen, team).astype(np.float32)
    out["ball_gap"] = gap
    return out


def _moving_average(values, window):
    """Centred moving average over +/- window frames, ignoring NaN; NaN where all are NaN."""
    if window <= 0:
        return values
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    n = len(values)
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    c = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, (total[hi] - total[lo]) / np.maximum(c, 1), np.nan)


def feature_matrix(raw, names, graded, smooth_frames):
    """(n, k) matrix of smoothed cues standardised over the graded frames, and the statistics."""
    columns, stats = [], {}
    for name in names:
        values = _moving_average(raw[name].astype(np.float64), smooth_frames)
        ref = values[graded & ~np.isnan(values)]
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        stats[name] = {"mean": mean, "std": std}
        columns.append((values - mean) / std)
    return np.stack(columns, axis=1), stats
