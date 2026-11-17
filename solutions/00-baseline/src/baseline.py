"""Challenge baseline: whoever is closest to the ball has it.

Port of `starter/baseline.py` from the challenge repo, split into the template's two stages.

  prepare  For every frame with a located ball, the team of the closest player. D where there is
           no ball. This is the expensive part (it reads 199,319 lines) and runs once.
  predict  Majority vote over a centred window of +/- `window` frames. It is cheap, so several
           windows are tried on the same output of `prepare`.

The tracking counts frames from 1 and submissions count from 0.
"""

import xml.etree.ElementTree as ET

import numpy as np

# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", each group "x,y,speed" with NaN when the
# observation is missing. Splitting on ':' always gives three parts.
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first letter of "NaN", which is all that needs checking

HOME, AWAY, NOBODY = 0, 1, -1


def team_of_each_track(meta_path):
    """Array over the track slots: 0 if the slot belongs to the home team, 1 if away.

    The first team in the XML is the home side (FIFA/EPTS convention). A slot is a trajectory
    fragment, not a person: there are 46 of them for 22 players.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away), for printing only."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def nearest_player_codes(raw_path, team_of_track, n_frames, pitch):
    """Team of the player closest to the ball, per frame. NOBODY where there is no ball.

    Coordinates are normalised 0-1 and the pitch is not square, so they are scaled to metres
    before distances are compared.
    """
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == MISSING:
                continue

            comma = ball.index(",")
            ball_x = float(ball[:comma]) * length
            ball_y = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]

            out[int(parts[0]) - 1] = best_team
    return out


def smooth(codes, window):
    """Majority vote over +/- `window` frames. D where nobody has the ball.

    Uses cumulative sums, so the cost does not depend on the window. Ties go to the home side.
    """
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
