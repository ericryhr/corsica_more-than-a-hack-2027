"""Ball imputation by drift: where the ball is missing, move its last known position with the mean
displacement of the tracked players.

  prepare  Parse the ATD once into player and ball arrays in metres.
  predict  Impute the ball, take the team of the closest player and smooth with a majority vote.

The hypothesis is that the ball goes where the team goes. If everyone shifts down the pitch, so
did the ball. The drift is the mean step of the slots visible in two consecutive frames, which
cancels slot switches and players entering or leaving the tracking.

The tracking counts frames from 1 and submissions count from 0.
"""

import xml.etree.ElementTree as ET

import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first letter of "NaN"

HOME, AWAY, NOBODY = 0, 1, -1


def team_of_each_track(meta_path):
    """Array over the track slots: 0 if the slot belongs to the home team, 1 if away.

    The first team in the XML is the home side (FIFA/EPTS convention).
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def read_tracking(raw_path, n_frames, n_slots, pitch):
    """(players, ball) in metres. players: (n_frames, n_slots, 2), ball: (n_frames, 2), NaN if missing."""
    scale = np.array(pitch, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] != MISSING:
                    x, y, _ = group.split(",")
                    players[frame, slot] = (float(x), float(y))
            if parts[2][0] != MISSING:
                x, y, _ = parts[2].split(",")
                ball[frame] = (float(x), float(y))
    return players * scale, ball * scale


def mean_step(players, min_players):
    """Mean displacement of the players between consecutive frames, per frame. Row 0 is zero.

    Only slots visible in both frames count. With fewer than `min_players` the step is zero.
    """
    delta = players[1:] - players[:-1]
    seen = ~np.isnan(delta[:, :, 0])
    count = seen.sum(axis=1)
    total = np.nansum(delta, axis=1)
    step = np.zeros_like(players[:, 0, :])
    step[1:] = np.where(count[:, None] >= min_players, total / np.maximum(count, 1)[:, None], 0.0)
    return step


def impute_ball(ball, step, pitch, gain, drift_frames=None, max_gap=None):
    """Ball per frame. Where it is missing: last known position + gain * accumulated player drift.

    The drift stops accumulating `drift_frames` frames after the last known ball (None: never), so
    the ball stays still after that. Frames before the first known ball stay NaN, and so do frames
    more than `max_gap` frames after the last known ball (None keeps them all). The result is
    clipped to the pitch.
    """
    n = len(ball)
    known = ~np.isnan(ball[:, 0])
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(known, index, -1))
    has_anchor = last >= 0
    anchor = np.where(has_anchor, last, 0)

    drift = np.cumsum(step, axis=0)
    stop = index if drift_frames is None else np.minimum(index, anchor + drift_frames)
    out = ball[anchor] + gain * (drift[stop] - drift[anchor])
    out = np.clip(out, 0.0, np.array(pitch, dtype=np.float32))
    out[known] = ball[known]
    out[~has_anchor] = np.nan
    if max_gap is not None:
        out[~known & (index - last > max_gap)] = np.nan
    return out


def nearest_codes(players, ball, team_of_track):
    """Team of the player closest to the ball, per frame. NOBODY where the ball is NaN."""
    distance = ((players - ball[:, None, :]) ** 2).sum(axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    codes = team_of_track[distance.argmin(axis=1)].astype(np.int8)
    codes[np.isnan(ball[:, 0]) | np.isinf(distance.min(axis=1))] = NOBODY
    return codes


def smooth(codes, window):
    """Majority vote over +/- `window` frames. D where nobody has the ball. Ties go to home."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels
