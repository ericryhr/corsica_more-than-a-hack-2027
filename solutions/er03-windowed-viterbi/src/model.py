"""Model specs, fitting and decoding.

A *variant* is (feature set, training labels):
  feature sets  fg07   fg07 `linear`'s owner and dead lists (the control: fg07 retrained here)
                full   those lists plus er02's cues (owner) and dead-ball features (dead)
                owner02 / dead02   er02's additions to one of the two models only (ablations)
  labels        bench / controlled (see labels.py)

Both models are fg03's L2 logistic regressions on the base features under three time windows
(before, around, after), fitted by fg07's `fit_linear`; decoding is fg03's 3-state Viterbi with a
cost per change of team and per stoppage (fg07 `methods.decode`).
"""

import numpy as np

from .features import ER02_CUES
from .peers import peer

ER02_OWNER = ["er02_" + c for c in ER02_CUES] + ["er02_ball"]
ER02_DEAD = ["er02d_" + f for f in ("speed", "ball_seen", "ball_speed", "ball_gap", "line_dist",
                                     "spread")]


def spec(config, features):
    s = {"windows": config["windows"], "l2": config["l2"], "stride": config["stride"],
         "owner": list(config["owner_fg07"]), "dead": list(config["dead_fg07"])}
    if features in ("full", "owner02"):
        s["owner"] += ER02_OWNER
    if features in ("full", "dead02"):
        s["dead"] += ER02_DEAD
    return s


def fit(pool, s):
    """pool: [(values, labels)]. Returns fg07's {"owner": Logistic, "dead": Logistic}."""
    return peer("fg07", "methods").fit_linear(pool, s)


def probs(models, values, s):
    return peer("fg07", "methods").linear_probs(models, values, s)


def viterbi(match, p_home, p_dead, team_cost, dead_cost):
    return peer("fg07", "methods").decode(match, p_home, p_dead, team_cost, dead_cost)


def merge_decode(match, p_home, p_dead, threshold, min_dead, min_spell):
    """er02's decoder as a control: D where p_dead > threshold (runs under min_dead merged away),
    H/A runs under min_spell merged into their neighbours."""
    decode = peer("er02", "decode")
    graded = peer("fg07", "data").graded(match)
    dead = np.zeros(len(p_dead), dtype=bool)
    for first, last in match["halves"]:
        code = (p_dead[first:last + 1] > threshold).astype(np.int8)
        dead[first:last + 1] = decode.merge_short_runs(code, min_dead) == 1
    labels = decode.decode(p_home, dead | ~graded, graded, 0.0, min_spell)
    labels[~graded] = "D"
    return labels
