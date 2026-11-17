"""The adapted methods, each a function of a match dict and its parameters that returns H/A/D
letters per frame. None of them reads match["labels"].

  baseline  00-baseline: team of the player nearest the ball, majority vote over +/- window.
  drift     01-ball-drift: the missing ball moved with the players' mean step, then as baseline.
  rules     fg01-smart-tagging's rule with only its ball part: ball evidence (weaker beyond
            ball_radius), dead on long ball gaps and on a ball outside the pitch, 2-state Viterbi.
            With `tags` (BAR-ATH only) the Smart Tagging enters with fixed prior constants.
  netfuse   fg04-passing-network: `rules` plus weight x clipped log-odds of a logistic owner model
            on the passing-network features.
  linear    fg03-linear-viterbi: owner and dead logistic models on windowed geometry features,
            3-state Viterbi. With network features in the owner model it is fg05.
"""

import numpy as np

from shared.challenge import pm

from .data import graded
from .peers import peer

LETTERS = np.array(["H", "A", "D"])
# An empty Smart Tagging table, for the sample games: every tag term of fg01's rule is then zero.
NO_TAGS = {"start": np.zeros(0, np.int64), "end": np.zeros(0, np.int64), "code": np.zeros(0, np.int8),
           "team": np.zeros(0, np.int8), "owner": np.zeros(0, np.int8), "kickoff": np.zeros(0, bool)}


def score(match, labels):
    """(spellF1, macroF1) against the match's labels."""
    result = pm.evaluate(pm.label_codes(match["labels"]), pm.label_codes(labels), bootstrap_n=0)
    return result["spell_f1_mean"], result["macro_f1"]


def nearest(match):
    """(codes, distance): team of the slot nearest the ball (-1 where the ball or every slot is
    missing) and its distance in metres. Distances do not depend on the orientation."""
    players, ball, team = match["players"], match["ball"], match["team"]
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    slot = d.argmin(axis=1)
    distance = d[np.arange(len(d)), slot]
    codes = team[slot].astype(np.int8)
    missing = np.isinf(distance)
    codes[missing] = -1
    return codes, np.where(missing, np.nan, distance).astype(np.float32)


def baseline(match, window):
    codes, _ = nearest(match)
    return peer("baseline", "baseline").smooth(codes, window)


def drift(match, gain, drift_frames, max_gap, window, min_players=3):
    module = peer("drift", "drift")
    step = module.mean_step(match["players"], min_players)
    ball = module.impute_ball(match["ball"], step, match["pitch"], gain, drift_frames, max_gap)
    return module.smooth(module.nearest_codes(match["players"], ball, match["team"]), window)


def rules(match, params, tags=None, extra=None, weight=0.0, clip=3.0):
    fuse = peer("fg04", "fuse")
    codes, distance = nearest(match)
    evidence, dead = fuse.parts(codes, distance, match["ball"], NO_TAGS if tags is None else tags,
                                match["pitch"], **params)
    return fuse.label(evidence, dead, params["switch_cost"], extra, weight, clip)


# --- trained models -----------------------------------------------------------------------------

def design(values, names, windows):
    return peer("fg03", "features").design(values, names, windows)[0]


def targets(labels):
    """{task: (rows, 0/1 target)} as fg03 fits them."""
    alive = np.isin(labels, ["H", "A"])
    return {"owner": (alive, labels == "H"), "dead": (labels != "X", labels == "D")}


def fit_linear(pool, spec):
    """fg03's owner and dead logistic models fitted on a pool of (values, labels), one row in
    `stride` frames (frames are autocorrelated, so this loses little)."""
    model = peer("fg03", "model")
    out = {}
    for task in ("owner", "dead"):
        xs, ys = [], []
        for values, labels in pool:
            rows, target = targets(labels)[task]
            keep = np.flatnonzero(rows)[::spec["stride"]]
            xs.append(design(values, spec[task], spec["windows"])[keep])
            ys.append(target[keep])
        out[task] = model.Logistic(spec["l2"]).fit(np.vstack(xs), np.concatenate(ys))
    return out


def linear_probs(models, values, spec):
    return (models["owner"].predict(design(values, spec["owner"], spec["windows"])),
            models["dead"].predict(design(values, spec["dead"], spec["windows"])))


def decode(match, p_home, p_dead, team_cost, dead_cost):
    """fg03's 3-state Viterbi inside each half; D outside the halves (never graded)."""
    model = peer("fg03", "model")
    logp = model.log_probs(p_home, p_dead)
    path = np.full(len(p_home), model.D, dtype=np.int8)
    for first, last in match["halves"]:
        path[first:last + 1] = model.viterbi(logp[first:last + 1], team_cost, dead_cost)
    return LETTERS[path]


def owner_matrix(values, names, smooth_frames):
    return peer("fg04", "model").matrix(values, names, smooth_frames)


def fit_owner(pool, names, smooth_frames, l2):
    """fg04's owner model (H vs A on alive frames) on a pool of (values, labels)."""
    stats = peer("fg02", "stats")
    xs, ys = [], []
    for values, labels in pool:
        alive = np.isin(labels, ["H", "A"])
        xs.append(owner_matrix(values, names, smooth_frames)[alive])
        ys.append(labels[alive] == "H")
    x = np.vstack(xs)
    scaler = stats.Standardizer().fit(x)
    return scaler, stats.logistic(scaler.transform(x), np.concatenate(ys).astype(float), l2)


def owner_logit(model, values, names, smooth_frames):
    """Log-odds of H without the intercept (fg04 `drop_intercept`): the intercept is the share
    of possession of the training games, a push that has nothing to do with this frame."""
    scaler, weights = model
    return scaler.transform(owner_matrix(values, names, smooth_frames)) @ weights[1:]


def mask_outside(match, labels):
    """D outside the halves, as the challenge expects nothing there."""
    out = np.asarray(labels).copy()
    out[~graded(match)] = "D"
    return out
