"""Per-frame base features and their time windows.

The base features are the 60 of fg02-geometry-eda (ball, proximity, team shape, Smart Tagging,
rolling context) plus the signals that the fg01-smart-tagging rule turned into votes:

  fg01_ball   +1/-1 for the team closest to the ball, 0.3 when that player is beyond 2 m
  fg01_tag    the three reliable phase codes, counted for 20 s at most and cut short once the ball
              has voted 75 frames for the other team (fg01 `tag_evidence`)
  sp_wait     1 in the first 5 s of a SET PIECES interval other than a kick-off (the wait)
  sp_team     +1/-1 toward the tagged team for the 5 s after that wait
  goals       1 inside a GOALS interval
  touch_diff  +1 if a home slot is within 1.5 m of the ball, -1 if an away one is, 0 if both/none
  touch_any   1 if a slot of either team is within 1.5 m of the ball

A window (lo, hi) turns a base feature into its mean over frames [t + lo, t + hi), ignoring NaN.
(0, 1) is the value itself. Windows that look ahead let the model use a signal that reacts after
the labelled change (team shape and running direction move about 2 s after a change of team).
"""

import importlib.util
import os
import sys

import numpy as np

SOLUTIONS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOUCH_RADIUS = 1.5


def load_module(name, *path):
    """Module of another solution, loaded by path because every solution names its package `src`.
    A path ending in __init__.py is loaded as a package so its relative imports work."""
    location = os.path.join(SOLUTIONS, *path)
    folder = os.path.dirname(location) if path[-1] == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(
        name, location, submodule_search_locations=[folder] if folder else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fg02_features = load_module("fg02_features", "fg02-geometry-eda", "src", "features.py")
fg02_stats = load_module("fg02_stats", "fg02-geometry-eda", "src", "stats.py")
load_module("fg01_src", "fg01-smart-tagging", "src", "__init__.py")
from fg01_src import rules as fg01_rules, signals as fg01_signals  # noqa: E402


def fg01_features(tags, n_frames, codes, distance):
    """The votes of the fg01 rule as features (see the module docstring)."""
    sp_wait, sp_team = fg01_rules.set_pieces(tags, n_frames, restart=125, hold=125, weight=1.0)
    goals = np.zeros(n_frames, dtype=np.float32)
    for start, end, code in zip(tags["start"], tags["end"], tags["code"]):
        if fg01_signals.CODES[code] == "GOALS":
            goals[start:end] = 1
    return {
        "fg01_ball": fg01_rules.ball_evidence(codes, distance, 2.0, 0.3),
        "fg01_tag": fg01_rules.tag_evidence(codes, tags, fg01_rules.WEIGHTS["reliable"], 500, 75),
        "sp_wait": sp_wait.astype(np.float32),
        "sp_team": sp_team,
        "goals": goals,
    }


def nearest(values):
    """(codes, distance) as fg01 `nearest_team` returns them, from the fg02 proximity features,
    so the ATD is parsed once. Rotating the second half does not change distances."""
    near = values["near_team"]
    codes = np.where(near > 0, fg01_signals.HOME,
                     np.where(near < 0, fg01_signals.AWAY, fg01_signals.NOBODY)).astype(np.int8)
    return codes, values["near_dist"]


def touch_features(d_home, d_away):
    home = np.nan_to_num(d_home, nan=np.inf) < TOUCH_RADIUS
    away = np.nan_to_num(d_away, nan=np.inf) < TOUCH_RADIUS
    return {"touch_diff": home.astype(np.float32) - away, "touch_any": (home | away).astype(np.float32)}


def build(meta_path, raw_path, tags_path, n_frames, pitch, second_half_start):
    """{name: (n_frames,) float32} with every base feature."""
    values, tags = fg02_features.build(meta_path, raw_path, tags_path, n_frames, pitch,
                                       second_half_start)
    codes, distance = nearest(values)
    values.update(fg01_features(tags, n_frames, codes, distance))
    values.update(touch_features(values["d_home"], values["d_away"]))
    return {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}


def window_mean(x, lo, hi):
    """Mean of x over [t + lo, t + hi), ignoring NaN; NaN where the window has no value."""
    seen = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, x, 0.0))))
    count = np.concatenate(([0.0], np.cumsum(seen)))
    index = np.arange(len(x))
    a = np.clip(index + lo, 0, len(x))
    b = np.clip(index + hi, 0, len(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total[b] - total[a]) / (count[b] - count[a])


def column_name(name, window):
    return name if tuple(window) == (0, 1) else "{}@{:+d}:{:+d}".format(name, *window)


def design(values, names, windows):
    """(matrix, column names): every base feature in `names` under every window."""
    columns, out = [], []
    for name in names:
        x = values[name].astype(np.float64)
        for window in windows:
            out.append(x if tuple(window) == (0, 1) else window_mean(x, *window))
            columns.append(column_name(name, window))
    return np.column_stack(out), columns
