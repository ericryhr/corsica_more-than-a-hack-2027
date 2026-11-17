"""Per-frame base features of a match dict (shared/sample.py layout; sample games and BAR-ATH alike).

  fg07   fg07-sample-tuned `features.base`: fg02's ball, proximity, shape and context features and
         fg03's touch features, on coordinates turned so that home attacks +x in every half.
  er02   er02-metrica-supervised's possession cues (prefix `er02_`) and dead-ball features (prefix
         `er02d_`), computed on the same oriented coordinates (so the home attack sign is +1).
         The cues are standardised per match over its halves, as er02 does: their scale differs
         between the clean sample tracking and the ATD's fragments, their ordering does not.

The time windows are applied later, by fg03's `design`, when a model is fitted or applied.
"""

import warnings

import numpy as np

from .peers import peer

ER02_CUES = ("motion", "marking", "shape", "height")


def fg07_values(match):
    return peer("fg07", "features").base(match)


def er02_values(match, config):
    data = peer("fg07", "data")
    cues = peer("er02", "cues")
    deadball = peer("er02", "deadball")
    players, ball = data.oriented(match)
    team = match["team"]
    n = len(ball)
    graded = data.graded(match)
    sign = np.ones(n, dtype=np.int8)
    cue_config = dict(config["er02_cues"], cues=list(ER02_CUES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = cues.compute_all(players, ball, team, sign, [], match["pitch"], cue_config)
        dead = deadball.raw_features(players, ball, match["pitch"], config["er02_cues"]["dead_lag"],
                                     match["fps"])
    out = {}
    for name in ER02_CUES:
        v = raw[name].astype(np.float64)
        ref = v[graded & ~np.isnan(v)]
        out["er02_" + name] = ((v - ref.mean()) / (ref.std() or 1.0)).astype(np.float32)
    out["er02_ball"] = np.where(raw["ball_gap"] > 0, np.nan, raw["ball_code"]).astype(np.float32)
    for name, v in dead.items():
        out["er02d_" + name] = v.astype(np.float32)
    return out
