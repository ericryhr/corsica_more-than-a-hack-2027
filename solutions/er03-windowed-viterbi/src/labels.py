"""Training labels for the sample games.

  bench       shared/sample.py's labels (match["labels"]): owner from the events, D from stoppage
              to restart, then one-team flickers under 2 s between two runs of the other team
              merged. The 2 s was matched to the shape of the public ground truth's possessions.
              This is also the truth every solution is scored against on the shared bench.
  controlled  the same events, with CHALLENGE.md's own proxies for "controlled" possession instead
              of the 2 s merge: a change of team counts only if the new team then has at least
              `min_events` on-ball events of its own, or keeps the ball at least `min_seconds`
              (until the next change or stoppage). Otherwise the previous owner continues, as for a
              touch without control or a deflection. A run that starts a half or follows a
              stoppage always counts: "whenever play restarts after a stoppage, there is a clear
              possessor".
"""

import numpy as np

ON_BALL = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST")


def _runs(labels):
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    return starts, ends


def controlled(raw_labels, events, min_events, min_frames):
    """raw_labels: shared/sample.py `labels_from_events` (no flicker merge). events: its
    `load_events` table (team 'H'/'A', type, start)."""
    labels = np.asarray(raw_labels, dtype=object).copy()
    on_ball = events[events["type"].isin(ON_BALL)]
    ev_frames = on_ball["start"].to_numpy()
    ev_teams = on_ball["team"].to_numpy()
    starts, ends = _runs(labels)
    for i in range(1, len(starts)):
        here, before = labels[starts[i]], labels[starts[i] - 1]   # `before` already resolved
        if here not in ("H", "A") or before not in ("H", "A") or here == before:
            continue
        a, b = np.searchsorted(ev_frames, [starts[i], ends[i]])
        own = int(np.sum(ev_teams[a:b] == here))
        if own >= min_events or ends[i] - starts[i] >= min_frames:
            continue
        labels[starts[i]:ends[i]] = before
    return labels


def controlled_for_game(game, config):
    from shared import sample
    events = sample.load_events(game)
    n = len(sample.load_match(game)["labels"])
    raw = sample.labels_from_events(events, n)
    return controlled(raw, events, config["controlled_min_events"],
                      int(round(config["controlled_min_seconds"] * sample.FPS)))
