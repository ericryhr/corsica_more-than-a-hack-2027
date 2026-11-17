"""Per-frame features of a match dict, the same code for the sample games and for BAR-ATH.

  base(match)     fg02's ball, proximity, shape and context features, plus fg03's touch features.
                  No Smart Tagging: the sample games have none, so nothing trained here can use it.
                  fg03's `fg01_ball` is left out too: its 2 m radius and 0.3 weight were chosen on
                  the hand labels, and `near_team` / `near_dist` carry the same information.
  network(match)  fg04's passing-lane and marking features of both teams, and their difference.

Both work on oriented coordinates (data.oriented), so +x is the home team's attack in every half.
"""

import warnings

import numpy as np

from .data import oriented
from .peers import peer

CONTEXT = (("near_team", 1), ("near_team", 3), ("top3_home", 1), ("d_diff", 1), ("vx_mean", 1),
           ("cx_mean", 1))


def base(match):
    fg02 = peer("fg02", "features")
    fg03 = peer("fg03", "features")
    players, ball = oriented(match)
    pitch, speed, team = match["pitch"], match["speed"], match["team"]
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames without a slot of one team
        values.update(fg02.ball_features(ball, pitch))
        values.update(fg02.proximity_features(players, speed, ball, team))
        values.update(fg02.shape_features(players, speed, team, pitch))
    for name, seconds in CONTEXT:
        values["{}_r{}".format(name, seconds)] = fg02.rolling_mean(values[name], seconds)
    values.update(fg03.touch_features(values["d_home"], values["d_away"]))
    return {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}


def network(match, cfg):
    players, ball = oriented(match)
    team = match["team"]
    # fg04 takes the first max_slots present slots of each team and assumes the team has that
    # many; a sample team has 14 to 17 slots, one per player.
    cfg = dict(cfg, max_slots=int(min(cfg["max_slots"], (team == 0).sum(), (team == 1).sum())))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return peer("fg04", "network").build(players, ball, team, cfg)
