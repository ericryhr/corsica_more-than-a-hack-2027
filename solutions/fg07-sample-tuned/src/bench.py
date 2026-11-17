"""fg07-sample-tuned on an in-memory match, for shared/sample_bench.py.

Only the methods without a model (baseline, drift, rules), with the parameters `run.py tune`
chose. They were chosen on these same 3 games, so the bench reads a little high for them; the
leave-one-game-out figures are in `run.py report`. The trained methods are scored leave one game
out by `run.py train` itself.
"""

import json
import os

from . import methods

TUNED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "artifacts", "fg07-sample-tuned", "tuned.json")


def label_match(match, config):
    with open(TUNED) as handle:
        tuned = {method: value["params"] for method, value in json.load(handle).items()}
    p = tuned["drift"]
    return {"baseline": methods.baseline(match, tuned["baseline"]["window"]),
            "drift": methods.drift(match, p["gain"], p["drift_frames"], p["max_gap"], p["window"]),
            "rules": methods.rules(match, tuned["rules"])}
