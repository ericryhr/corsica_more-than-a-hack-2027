"""Synthetic checks of the controlled-possession labels, the model specs and the bench routing.

    python -m pytest solutions/er03-windowed-viterbi/tests
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from src import labels, model  # noqa: E402
from src.bench import which  # noqa: E402


def events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "start"])


def test_controlled_drops_a_touch_without_control():
    # A has the ball; H touches it once (one event, 3 frames), A keeps playing.
    raw = np.array(list("AAAAA" + "HHH" + "AAAAAA"), dtype=object)
    ev = events([("A", "PASS", 0), ("H", "RECOVERY", 5), ("A", "RECOVERY", 8), ("A", "PASS", 10)])
    out = labels.controlled(raw, ev, min_events=2, min_frames=50)
    assert "".join(out) == "A" * 14


def test_controlled_keeps_a_real_change():
    raw = np.array(list("AAAAA" + "HHHHHH"), dtype=object)
    ev = events([("A", "PASS", 0), ("H", "RECOVERY", 5), ("H", "PASS", 7)])   # two own events
    assert "".join(labels.controlled(raw, ev, 2, 50)) == "AAAAAHHHHHH"
    ev = events([("A", "PASS", 0), ("H", "RECOVERY", 5)])                      # one event, long
    assert "".join(labels.controlled(raw, ev, 2, 4)) == "AAAAAHHHHHH"


def test_controlled_keeps_restart_owner():
    raw = np.array(list("AAA" + "DD" + "HH" + "AAA"), dtype=object)
    ev = events([("A", "PASS", 0), ("H", "SET PIECE", 5), ("A", "RECOVERY", 7), ("A", "PASS", 8)])
    # H restarts after a stoppage (always counts) even with one event and 2 frames
    assert "".join(labels.controlled(raw, ev, 2, 50)) == "AAADDHHAAA"


def test_spec_full_adds_er02_features():
    config = {"windows": [[-1, 0], [0, 1]], "l2": 1.0, "stride": 1, "owner_fg07": ["a"],
              "dead_fg07": ["b"]}
    assert model.spec(config, "fg07")["owner"] == ["a"]
    full = model.spec(config, "full")
    assert full["owner"][0] == "a" and "er02_shape" in full["owner"] and "er02d_speed" in full["dead"]


def test_bench_uses_the_model_that_never_saw_the_game():
    models = {1: "without 1", 2: "without 2", 3: "without 3", "final": "all"}
    assert which(models, 2) == "without 2" and which(models, "BAR-ATH") == "all"
