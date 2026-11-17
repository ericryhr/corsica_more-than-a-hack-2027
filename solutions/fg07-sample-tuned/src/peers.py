"""Code of other solutions that this one reuses, loaded by path because every solution names its
package `src`. Nothing is copied: if those files change, this solution changes with them.

  00-baseline/src           baseline.smooth (majority vote) and NOBODY
  01-ball-drift/src         drift (ball imputation by the players' drift)
  fg01-smart-tagging/src    signals (Smart Tagging parser) and rules (ball evidence, Viterbi)
  fg02-geometry-eda/src     features (ATD reader, ball / proximity / shape features) and stats
  fg03-linear-viterbi/src   features (window means, design matrix) and model (logistic, Viterbi)
  fg04-passing-network/src  network (passing-lane features) and fuse (rule evidence + extra term)
"""

import importlib
import importlib.util
import os
import sys

SOLUTIONS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FOLDERS = {
    "baseline": "00-baseline",
    "drift": "01-ball-drift",
    "fg01": "fg01-smart-tagging",
    "fg02": "fg02-geometry-eda",
    "fg03": "fg03-linear-viterbi",
    "fg04": "fg04-passing-network",
}
# fg03 and fg04 load fg01 and fg02 under these aliases themselves, so this solution uses the same
# ones and every module is executed once.
ALIASES = {"fg01": "fg01_src", "fg02": "fg02_src", "fg03": "fg03_src", "fg04": "fg04_src",
           "baseline": "baseline_src", "drift": "drift_src"}


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
    """peer("fg01", "rules") is fg01-smart-tagging/src/rules.py."""
    return importlib.import_module(_package(key) + "." + module)
