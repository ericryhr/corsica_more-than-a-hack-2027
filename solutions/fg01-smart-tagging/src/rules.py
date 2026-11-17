"""Rules that turn the per-frame signals into H/A/D labels.

Every rule makes two separate calls: alive or dead, and which team.

Which team. Per-frame evidence, positive for home and negative for away, from `source`:

  ball      +1/-1 from the team closest to the ball, 0 where it is missing. With `ball_radius`
            the vote is only `far_weight` when that player is further than `ball_radius` metres.
  tags      Sum over the active intervals of weight(code) * sign(owner). An interval only counts
            for its first `tag_max_age` frames, because the tags are reliable near their start and
            overrun the real change of possession near their end. With `expire` an interval also
            stops counting once the ball has voted `expire` frames for the other team since it
            started.
  fallback  ball, and the tags only on frames where the ball is missing.
  fuse      ball + tag_weight * tags on every frame.

Set pieces (`sp_restart`) add `sp_weight` towards the tagged team for `sp_hold` frames after the
restart. The evidence is then smoothed: `vote` is the baseline majority over +/- `window` frames;
`viterbi` is a two-state path that pays `switch_cost` per change of team, so short contradictions
do not split a spell.

Alive or dead. A frame is D when any of these holds:

  - the ball has been missing for more than `dead_gap` frames in a row; with `dead_needs_no_tag`,
    only where no tag of the chosen weights is active,
  - the ball is more than `out_margin` metres outside the pitch,
  - `goals_dead` and the frame is inside a GOALS interval (the celebration),
  - `sp_restart` and the frame is in the first `sp_restart` frames of a SET PIECES interval.
    Kick-offs are left out: their 5 s interval surrounds the kick, and the ball is already in play
    at its end.
"""

import numpy as np

from .signals import AWAY, CODES, HOME, NOBODY

# Code weight presets. Zone codes (where the ball is) are dropped or weakened in favour of the
# phase codes (who is attacking), which agree with the hand labels far more often.
WEIGHTS = {
    "all": {code: 1.0 for code in CODES if code not in ("GOALS", "SET PIECES")},
    "phases": {"BUILD UP": 1.0, "PROGRESSION": 1.0, "CREATING CHANCES": 1.0,
               "ATTACKING TRANSITION": 1.0, "LONG BALL": 1.0},
    "reliable": {"BUILD UP": 1.0, "PROGRESSION": 1.0, "LONG BALL": 1.0},
    "graded": {"BUILD UP": 1.0, "PROGRESSION": 0.8, "LONG BALL": 0.7,
               "ATTACKING TRANSITION": 0.4, "CREATING CHANCES": 0.2, "BALL IN THE BOX": 0.4,
               "BALL IN FINAL THIRD": 0.3, "PLAYERS IN THE BOX": 0.2},
}
# A defending code is the twin of an attacking one: same interval, owner already flipped. Counting
# both would double the vote, so only the attacking side is used.


def ball_evidence(codes, distance=None, radius=None, far_weight=1.0):
    vote = np.where(codes == HOME, 1.0, np.where(codes == AWAY, -1.0, 0.0))
    if radius is not None:
        vote = np.where(distance > radius, far_weight * vote, vote)
    return vote


def tag_evidence(codes, tags, weights, max_age=None, expire=None):
    """Per-frame sum of weight * sign(owner) over the active intervals."""
    out = np.zeros(len(codes))
    ball = ball_evidence(codes)
    for start, end, code, owner in zip(tags["start"], tags["end"], tags["code"], tags["owner"]):
        weight = weights.get(CODES[code], 0.0)
        if not weight or owner == NOBODY:
            continue
        if max_age is not None:
            end = min(end, start + max_age)
        sign = 1.0 if owner == HOME else -1.0
        if expire is not None:
            against = np.cumsum(ball[start:end] == -sign)
            beyond = np.flatnonzero(against >= expire)
            if len(beyond):
                end = start + beyond[0]
        out[start:end] += weight * sign
    return out


def dead_frames(codes, tags, dead_gap, goals_dead, protected=None):
    """True where the frame is labelled D. A ball gap is dead only on frames not `protected`
    (frames where a tag says some team is attacking)."""
    missing = codes == NOBODY
    dead = np.zeros(len(codes), dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], missing.astype(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start > dead_gap:
            dead[start:stop] = True
    if protected is not None:
        dead &= ~protected
    if goals_dead:
        for start, end, code in zip(tags["start"], tags["end"], tags["code"]):
            if CODES[code] == "GOALS":
                dead[start:end] = True
    return dead


def set_pieces(tags, n, restart, hold, weight):
    """(dead, evidence) from SET PIECES: D for the first `restart` frames of each (the wait before
    the kick), then `weight` towards the tagged team for `hold` frames. Throw-ins (no team) only
    give the D part. The tag end is synthetic (always 5 or 35 s), so it is not used."""
    dead = np.zeros(n, dtype=bool)
    evidence = np.zeros(n)
    for start, code, owner, kickoff in zip(tags["start"], tags["code"], tags["owner"],
                                           tags["kickoff"]):
        if CODES[code] != "SET PIECES" or kickoff:
            continue
        dead[start:start + restart] = True
        if owner != NOBODY:
            evidence[start + restart:start + restart + hold] += weight * (1.0 if owner == HOME else -1.0)
    return dead, evidence


def vote(evidence, window):
    """Sum of the evidence over +/- window frames."""
    if window <= 0:
        return evidence
    cumulative = np.concatenate(([0.0], np.cumsum(evidence)))
    index = np.arange(len(evidence))
    low = np.maximum(0, index - window)
    high = np.minimum(len(evidence), index + window + 1)
    return cumulative[high] - cumulative[low]


def viterbi(evidence, switch_cost, segments):
    """+1/-1 path that maximises sum(evidence * state) - switch_cost * changes, solved
    independently inside each [start, stop) segment."""
    out = np.ones(len(evidence))
    for start, stop in segments:
        e = evidence[start:stop].tolist()
        n = len(e)
        if not n:
            continue
        from_other = bytearray(2 * n)   # [2t + s] = 1 if state s at t came from the other state
        home, away = e[0], -e[0]
        for t in range(1, n):
            move_h, move_a = away - switch_cost, home - switch_cost
            if move_h > home:
                from_other[2 * t] = 1
            if move_a > away:
                from_other[2 * t + 1] = 1
            home, away = max(home, move_h) + e[t], max(away, move_a) - e[t]
        state = int(away > home)
        path = np.empty(n)
        for t in range(n - 1, -1, -1):
            path[t] = 1.0 - 2.0 * state
            if from_other[2 * t + state]:
                state = 1 - state
        out[start:stop] = path
    return out


def alive_segments(dead):
    edges = np.flatnonzero(np.diff(np.concatenate(([1], dead.astype(np.int8), [1]))))
    return list(zip(edges[::2], edges[1::2]))


def apply(codes, distance, ball_xy, tags, pitch, source="ball", weights="phases", tag_weight=1.0,
          tag_max_age=None, expire=None, smoother="vote", window=50, switch_cost=50.0,
          dead_gap=50, goals_dead=False, dead_needs_no_tag=False, sp_restart=None, sp_hold=125,
          sp_weight=3.0, ball_radius=None, far_weight=1.0, out_margin=None):
    """H/A/D label per frame."""
    ball = ball_evidence(codes, distance, ball_radius, far_weight)
    tag = tag_evidence(codes, tags, WEIGHTS[weights], tag_max_age, expire)
    if source == "ball":
        evidence = ball
    elif source == "tags":
        evidence = tag
    elif source == "fallback":
        evidence = np.where(codes == NOBODY, tag_weight * tag, ball)
    elif source == "fuse":
        evidence = ball + tag_weight * tag
    else:
        raise ValueError("unknown source {!r}".format(source))

    dead = dead_frames(codes, tags, dead_gap, goals_dead, tag != 0 if dead_needs_no_tag else None)
    if out_margin is not None:
        x, y = ball_xy[:, 0], ball_xy[:, 1]
        with np.errstate(invalid="ignore"):
            dead |= (x < -out_margin) | (x > pitch[0] + out_margin) | \
                    (y < -out_margin) | (y > pitch[1] + out_margin)
    if sp_restart is not None:
        sp_dead, sp_evidence = set_pieces(tags, len(codes), sp_restart, sp_hold, sp_weight)
        dead |= sp_dead
        evidence = evidence + sp_evidence
    evidence = np.where(dead, 0.0, evidence)
    if smoother == "vote":
        team = vote(evidence, window)
    elif smoother == "viterbi":
        team = viterbi(evidence, switch_cost, alive_segments(dead))
    else:
        raise ValueError("unknown smoother {!r}".format(smoother))

    # Ties (no evidence in the window) keep the previous team, so a quiet stretch does not flip.
    sign = np.sign(team)
    idx = np.where(sign != 0, np.arange(len(sign)), 0)
    np.maximum.accumulate(idx, out=idx)
    sign = np.where(sign != 0, sign, sign[idx])
    labels = np.where(sign < 0, "A", "H").astype("<U1")
    labels[dead] = "D"
    return labels
