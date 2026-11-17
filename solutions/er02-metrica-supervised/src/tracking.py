"""Read the provided match data into arrays: tracking in metres, teams, attack direction, Smart Tagging.

Parsing follows `solutions/01-ball-drift/src/drift.py` (ATD) and `labeler/labeler/atd.py` /
`labeler/labeler/smart_tagging.py` (attack direction, Smart Tagging). They are copied rather than
imported because solutions do not import each other; moving them into `shared/` is a follow-up.

Frames are video frames counted from 0, as in the submission. The ATD counts from 1.
"""

import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first letter of "NaN"

HOME, AWAY = 0, 1
DIRECTIONS = {"left_to_right": 1, "right_to_left": -1}

# Smart Tagging codes whose `Team` tag is the team WITH the ball (+1) or WITHOUT it (-1).
# PLAYERS IN THE BOX, GOALS and SET PIECES say nothing reliable about who is on the ball.
ATTACKING_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL")
DEFENDING_CODES = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")


def _teams(root):
    """[(team_id, name)] in file order, and the home team id (Score/idLocalTeam, else the first)."""
    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in the ATD metadata, found {}".format(len(teams)))
    score = root.find(".//Score")
    home = score.get("idLocalTeam") if score is not None else teams[0][0]
    return teams, home


def read_meta(meta_path):
    """Team of each track slot (HOME/AWAY), team names {HOME: name, AWAY: name} and the first-half
    attack sign of the home team (+1: towards x = pitch length)."""
    root = ET.parse(meta_path).getroot()
    teams, home = _teams(root)
    team = np.array([HOME if p.get("teamId") == home else AWAY for p in root.iter("Player")],
                    dtype=np.int8)
    if team.size == 0:
        raise ValueError("{}: no Player entries".format(meta_path))
    names = {HOME if tid == home else AWAY: name for tid, name in teams}

    home_sign = None
    for t in root.iter("Team"):
        for par in t.iter("ProviderParameter"):
            if par.findtext("Name") == "attack_direction_first_half":
                sign = DIRECTIONS.get((par.findtext("Value") or "").strip())
                if sign is not None:
                    home_sign = sign if t.get("id") == home else -sign
    if home_sign is None:
        raise ValueError("{}: no attack_direction_first_half for the teams".format(meta_path))
    return team, names, home_sign


def read_tracking(raw_path, n_frames, n_slots, pitch):
    """(players, ball) in metres. players: (n_frames, n_slots, 2), ball: (n_frames, 2), NaN if missing."""
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] != MISSING:
                    x, y, _ = group.split(",")
                    players[frame, slot] = (float(x), float(y))
            if parts[2][0] != MISSING:
                x, y, _ = parts[2].split(",")
                ball[frame] = (float(x), float(y))
    scale = np.array(pitch, dtype=np.float32)
    return players * scale, ball * scale


def home_sign_per_frame(n_frames, blocks, first_half_sign):
    """Attack sign of the home team per frame: the metadata sign in half 1, flipped in half 2.
    Frames outside the halves get the sign of the nearest half (they are never graded)."""
    sign = np.full(n_frames, first_half_sign, dtype=np.int8)
    (_, _, end1), (_, start2, _) = blocks[0], blocks[1]
    sign[(end1 + start2) // 2:] = -first_half_sign
    return sign


def in_halves(n_frames, blocks):
    """Boolean mask of the frames inside the two halves (inclusive blocks)."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def read_smart_tagging(path, names):
    """Instances as dicts {code, start, end (video frames, half-open), side (HOME/AWAY/None), type}."""
    side_of = {name: side for side, name in names.items()}
    root = ET.parse(path).getroot()
    out = []
    for el in root.findall("./ALL_INSTANCES/instance"):
        tags = {lab.findtext("group"): (lab.findtext("text") or "").strip()
                for lab in el.findall("label")}
        start = int(round(float(el.findtext("start")) * FPS))
        end = max(start + 1, int(round(float(el.findtext("end")) * FPS)))
        out.append({"code": el.findtext("code").strip().upper(), "start": start, "end": end,
                    "side": side_of.get(tags.get("Team")), "type": tags.get("Type")})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out
