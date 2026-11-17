"""00-baseline on an in-memory match, for shared/sample_bench.py.

Same algorithm as baseline.nearest_player_codes + smooth, from arrays instead of the ATD file.
"""

import numpy as np

from . import baseline


def label_match(match, config):
    """{variant: labels}, variants named as in run.py: w<window>."""
    players, ball, team = match["players"], match["ball"], match["team"]
    distance = ((players - ball[:, None, :]) ** 2).sum(axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    codes = team[distance.argmin(axis=1)].astype(np.int8)
    codes[np.isnan(ball[:, 0]) | np.isinf(distance.min(axis=1))] = baseline.NOBODY
    return {"w{}".format(window): baseline.smooth(codes, window) for window in config["windows"]}
