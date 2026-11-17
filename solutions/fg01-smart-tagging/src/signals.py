"""Per-frame signals for the rules: who is closest to the ball, and which Smart Tagging intervals
are active.

  prepare  Parse the ATD once into the team closest to the ball per frame, and the Smart Tagging
           XML into an interval table. Both in video frames, counted from 0.

The tracking counts frames from 1 and submissions count from 0. Smart Tagging times are seconds
from the first frame of the tactical video, so seconds x 25 is a video frame (see
repo/more-than-a-hack-2027/starter/load_smart_tagging.py).
"""

import xml.etree.ElementTree as ET

import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first letter of "NaN"

HOME, AWAY, NOBODY = 0, 1, -1
TEAM_CODE = {"barcelona": HOME, "atletic club": AWAY}

# Attacking code -> its defending twin. The twin carries the other team and the same interval.
PAIRS = {"BUILD UP": "DEFENDING IN ATTACKING THIRD",
         "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
         "CREATING CHANCES": "DEFENDING IN DEFENSIVE THIRD",
         "ATTACKING TRANSITION": "DEFENSIVE TRANSITION"}
DEFENDING = set(PAIRS.values())
CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
         "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
         "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION",
         "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL", "GOALS",
         "SET PIECES")


def team_of_each_track(meta_path):
    """Array over the track slots: 0 if the slot belongs to the home team, 1 if away.

    The first team in the XML is the home side (FIFA/EPTS convention).
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def nearest_team(raw_path, n_frames, team, pitch):
    """(codes, distance, ball). codes: team of the player closest to the ball per frame, NOBODY
    where the ball or every player is missing. distance: that player's distance to the ball in
    metres, NaN where codes is NOBODY. ball: (n_frames, 2) position in metres, NaN if missing."""
    scale = np.array(pitch, dtype=np.float32)
    codes = np.full(n_frames, NOBODY, dtype=np.int8)
    distance = np.full(n_frames, np.nan, dtype=np.float32)
    balls = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            bx, by, _ = parts[2].split(",")
            ball = np.array((float(bx), float(by)), dtype=np.float32) * scale
            balls[frame] = ball
            best, best_slot = np.inf, -1
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y, _ = group.split(",")
                d = np.hypot(float(x) * scale[0] - ball[0], float(y) * scale[1] - ball[1])
                if d < best:
                    best, best_slot = d, slot
            if best_slot >= 0:
                codes[frame] = team[best_slot]
                distance[frame] = best
    return codes, distance, balls


def read_tags(path, fps):
    """Interval table as parallel arrays: start, end (video frames, half-open), code (index into
    CODES), team (HOME, AWAY or NOBODY as written in the Team tag), owner (the team the tag
    says has the ball: the Team tag, or the other team for a defending code; NOBODY for N/A) and
    kickoff (1 for the SET PIECES of a kick-off, whose interval surrounds the kick)."""
    rows = []
    for instance in ET.parse(path).getroot().iter("instance"):
        code = instance.findtext("code").strip().upper()
        tags = {(label.findtext("group") or "").strip().lower(): (label.findtext("text") or "").strip()
                for label in instance.findall("label")}
        team = TEAM_CODE.get(tags.get("team", "").lower(), NOBODY)
        owner = 1 - team if code in DEFENDING and team != NOBODY else team
        start = int(round(float(instance.findtext("start")) * fps))
        end = max(start + 1, int(round(float(instance.findtext("end")) * fps)))
        kickoff = int(tags.get("type", "").lower().startswith("kick off"))
        rows.append((start, end, CODES.index(code), team, owner, kickoff))
    rows.sort()
    table = np.array(rows, dtype=np.int64)
    return {"start": table[:, 0], "end": table[:, 1], "code": table[:, 2].astype(np.int8),
            "team": table[:, 3].astype(np.int8), "owner": table[:, 4].astype(np.int8),
            "kickoff": table[:, 5].astype(bool)}
