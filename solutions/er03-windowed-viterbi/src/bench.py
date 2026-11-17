"""er03-windowed-viterbi on an in-memory match, for shared/sample_bench.py.

A sample game is labelled by the models fitted WITHOUT it (leave one game out, saved by
`run.py train`); any other match (BAR-ATH) by the final models. The models were trained under the
blurred settings; the bench may score them under other degradations too. Variants are
`<features>-<training labels>` (see model.py), each with the Viterbi costs `train` chose.
"""

import os
import pickle

from . import features, model
from .peers import peer

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "artifacts", "er03-windowed-viterbi", "models.pkl")
_saved = {}


def _load():
    if not _saved:
        peer("fg03", "model")          # the pickled classes live in fg03's package
        with open(MODELS, "rb") as handle:
            _saved.update(pickle.load(handle))
    return _saved


def which(models, game):
    """The model that never saw `game`: its leave-one-game-out fit, else the final one."""
    return models[game] if game in models else models["final"]


def label_match(match, config):
    saved = _load()
    values = features.fg07_values(match)
    values.update(features.er02_values(match, config))
    out = {}
    for variant, costs in saved["costs"].items():
        s = model.spec(config, variant.split("-")[0])
        p_home, p_dead = model.probs(which(saved["models"][variant], match.get("game")), values, s)
        out[variant] = model.viterbi(match, p_home, p_dead, costs["team"], costs["dead"])
    return out
