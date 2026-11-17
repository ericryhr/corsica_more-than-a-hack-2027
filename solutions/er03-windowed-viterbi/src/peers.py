"""Code of other solutions that this one reuses, loaded by path because every solution names its
package `src` (the pattern of fg07-sample-tuned/src/peers.py). Nothing is copied.

  fg07-sample-tuned/src         data (match dicts, orientation), features.base (fg02 + fg03 touch),
                                methods (fit_linear, linear_probs, Viterbi decode, score)
  er02-metrica-supervised/src   cues (motion, marking, shape, height) and deadball features
  fg03-linear-viterbi/src       features.design (time windows), model (logistic, Viterbi)
"""

import importlib
import importlib.util
import os
import sys

SOLUTIONS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FOLDERS = {"fg07": "fg07-sample-tuned", "er02": "er02-metrica-supervised",
           "fg03": "fg03-linear-viterbi"}
# fg07 loads fg03 under "fg03_src"; using the same alias keeps one copy of each module, which
# matters for pickled models.
ALIASES = {"fg07": "fg07_src", "er02": "er02_src", "fg03": "fg03_src"}


def _package(key):
    alias = ALIASES[key]
    if alias not in sys.modules:
        path = os.path.join(SOLUTIONS, FOLDERS[key], "src")
        spec = importlib.util.spec_from_file_location(
            alias, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return alias


def peer(key, module):
    """peer("fg07", "methods") is fg07-sample-tuned/src/methods.py."""
    return importlib.import_module(_package(key) + "." + module)
