"""er02-metrica-supervised on an in-memory match, for shared/sample_bench.py.

Same method as run.py (er01's cues + a decaying ball term for H vs A, er02's dead-ball detector,
er01's decoder with the settings `train` chose), refitted on the shared testbed's labels: a sample
game is labelled by models fitted on the OTHER sample games (degraded with the `default` setting,
the closest to run.py's occlusion transfer), any other match by models fitted on all three.
The home attack direction is read from where the teams stand, as er01's bench does.
"""

import json
import os

import numpy as np

from . import cues, deadball, decode, model, tracking

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "artifacts", "er02-metrica-supervised", "model.json")
DEFAULT_DECODER = {"dead_threshold": 0.4, "min_dead_run": 25, "min_spell": 50}
_cache, _models = {}, {}


def _sign(players, team, halves):
    n = len(players)
    sign = np.ones(n, dtype=np.int8)
    for first, last in halves:
        x = players[first:last + 1:25, :, 0]
        with np.errstate(invalid="ignore"):
            s = 1 if np.nanmean(x[:, team == tracking.HOME]) < np.nanmean(x[:, team == tracking.AWAY]) else -1
        sign[first:] = s
    return sign


def _graded(match):
    mask = np.zeros(len(match["ball"]), dtype=bool)
    for first, last in match["halves"]:
        mask[first:last + 1] = True
    return mask


def _features(match, config, tau):
    players, ball, team = match["players"], match["ball"], match["team"]
    graded = _graded(match)
    sign = _sign(players, team, match["halves"])
    raw = cues.compute_all(players, ball, team, sign, [], match["pitch"], config)
    X, _ = cues.feature_matrix(raw, config["cues"], graded, config["smooth_frames"])
    X = np.nan_to_num(X, nan=0.0)
    code, gap = np.nan_to_num(raw["ball_code"]), raw["ball_gap"]
    X = np.hstack([X, (code * np.exp(-gap / tau))[:, None]])
    dead_raw = deadball.raw_features(players, ball, match["pitch"], config["dead_lag"], match["fps"])
    return X, deadball.feature_matrix(dead_raw, graded, config["dead_windows"]), graded


def _fit(games, config, tau):
    from shared.sample import DEGRADE, degrade, load_match
    xs, ys, ds, dys = [], [], [], []
    for g in games:
        if g not in _cache:
            m = degrade(load_match(g), seed=g, **DEGRADE)
            _cache[g] = (_features(m, config, tau), m["labels"])
        (X, Dm, graded), labels = _cache[g]
        live = np.isin(labels, ["H", "A"])
        xs.append(X[live]); ys.append((labels[live] == "H").astype(np.int8))
        ds.append(Dm[graded]); dys.append((labels[graded] == "D").astype(np.int8))
    owner = model.fit_logistic(np.vstack(xs), np.concatenate(ys), config["l2"])
    dead = model.fit_logistic(np.vstack(ds), np.concatenate(dys), config["l2"])
    return owner, dead


def label_match(match, config):
    from shared.sample import GAMES_SAMPLE
    saved = {}
    if os.path.isfile(ARTIFACT):
        with open(ARTIFACT) as handle:
            saved = json.load(handle)
    tau = saved.get("ball_tau", 12)
    dec = saved.get("decoder_det", DEFAULT_DECODER)
    game = match.get("game")
    train_games = tuple(g for g in GAMES_SAMPLE if g != game)
    if train_games not in _models:
        _models[train_games] = _fit(train_games, config, tau)
    (w, b), (wd, bd) = _models[train_games]
    X, Dm, graded = _features(match, config, tau)
    prob = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    p_dead = 1.0 / (1.0 + np.exp(-(Dm @ wd + bd)))
    dead = np.zeros(len(prob), dtype=bool)
    for first, last in match["halves"]:
        code = (p_dead[first:last + 1] > dec["dead_threshold"]).astype(np.int8)
        dead[first:last + 1] = decode.merge_short_runs(code, dec["min_dead_run"]) == 1
    labels = decode.decode(prob, dead | ~graded, graded, 0.0, dec["min_spell"])
    return {"det-s{}".format(dec["min_spell"]): labels}
