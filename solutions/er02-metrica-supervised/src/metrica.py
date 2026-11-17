"""Metrica Sports public sample games: tracking, events, and per-frame H/A/D labels from events.

Source: https://github.com/metrica-sports/sample-data (anonymised; please credit Metrica Sports).
These are other matches than the one graded, so their labels may be used (RULES.md: "hand labels
from other games" are allowed).

  Sample_Game_1, Sample_Game_2   CSV: one tracking file per team, one events file.
  Sample_Game_3                  FIFA EPTS: metadata XML + tracking TXT, events JSON.

Everything is returned with the same layout as the BAR-ATH arrays of er01:
  players (n, slots, 2) float32 metres, NaN when the player is not on the pitch / not tracked;
  ball (n, 2) float32 metres; team (slots,) HOME/AWAY. Frames are counted from 0 (file frame - 1).

Labels from events (see `event_labels`):
  H / A  the team of the last on-ball event (pass, carry, shot, recovery, set piece, ball lost).
         Challenges (duels) never move possession: they are touches, not control.
  D      from the end of the last event before a restart (set piece or goal kick) to the restart.
         That event is what killed the ball: a ball out, a shot out or in, a foul.
  X      outside the played periods (before the kick-off, half-time, after the last event).
"""

import json
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1
OWNER_TYPES = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST")
IGNORED_BEFORE_RESTART = ("CARD",)


def fetch(repo, commit, dest):
    """Clone the sample-data repository at a pinned commit into dest (skipped if present)."""
    if os.path.isdir(os.path.join(dest, "data")):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", repo, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "--quiet", commit], check=True)
    return dest


# -- tracking ----------------------------------------------------------------------------------

def _read_csv_team(path):
    """(frames (n,) 0-based, positions (n, players, 2) normalised, ball (n, 2) normalised)."""
    table = pd.read_csv(path, skiprows=2)
    values = table.to_numpy(dtype=np.float64)
    frames = values[:, 1].astype(np.int64) - 1
    body = values[:, 3:]
    players = body[:, :-2].reshape(len(body), -1, 2)
    return frames, players, body[:, -2:]


def read_csv_game(home_path, away_path, pitch):
    """Sample games 1 and 2: players (n, home + away slots, 2) and ball in metres, team per slot."""
    frames, home, ball = _read_csv_team(home_path)
    frames_a, away, _ = _read_csv_team(away_path)
    if not np.array_equal(frames, frames_a):
        raise ValueError("home and away tracking files cover different frames")
    n = int(frames.max()) + 1
    players = np.full((n, home.shape[1] + away.shape[1], 2), np.nan, dtype=np.float32)
    players[frames] = np.concatenate([home, away], axis=1)
    out_ball = np.full((n, 2), np.nan, dtype=np.float32)
    out_ball[frames] = ball
    team = np.array([HOME] * home.shape[1] + [AWAY] * away.shape[1], dtype=np.int8)
    scale = np.asarray(pitch, dtype=np.float32)
    return players * scale, out_ball * scale, team


def read_epts_game(meta_path, raw_path, pitch):
    """Sample game 3 (FIFA EPTS). The slot -> player mapping changes with each substitution
    (one DataFormatSpecification per frame range), so every player gets a fixed slot of its own."""
    root = ET.parse(meta_path).getroot()
    score = root.find(".//Score")
    home_id = score.get("idLocalTeam")
    player_ids = [p.get("id") for p in root.iter("Player")]
    slot_of = {pid: i for i, pid in enumerate(player_ids)}
    team = np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        slots = [slot_of[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")) - 1, int(spec.get("endFrame")) - 1,
                      np.array(slots)))
    n = max(end for _, end, _ in specs) + 1

    players = np.full((n, len(player_ids), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    lookup = {}
    for start, end, slots in specs:
        lookup.update({f: slots for f in range(start, end + 1)})
    with open(raw_path) as handle:
        for line in handle:
            frame_txt, body, ball_txt = line.strip().split(":")
            frame = int(frame_txt) - 1
            xy = np.array([g.split(",") for g in body.split(";")], dtype=np.float32)
            players[frame, lookup[frame]] = xy
            ball[frame] = np.array(ball_txt.split(","), dtype=np.float32)
    scale = np.asarray(pitch, dtype=np.float32)
    return players * scale, ball * scale, team


# -- events ------------------------------------------------------------------------------------

def read_csv_events(path):
    table = pd.read_csv(path)
    out = pd.DataFrame({
        "side": np.where(table["Team"].str.strip().str.upper() == "HOME", HOME, AWAY),
        "type": table["Type"].str.strip().str.upper(),
        "subtype": table["Subtype"].fillna("").str.strip().str.upper(),
        "period": table["Period"].astype(int),
        "start": table["Start Frame"].astype(int) - 1,
        "end": table["End Frame"].astype(int) - 1,
    })
    return _tidy(out)


def read_json_events(path, home_id):
    with open(path) as handle:
        data = json.load(handle)["data"]
    rows = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        rows.append({
            "side": HOME if e["team"]["id"] == home_id else AWAY,
            "type": e["type"]["name"].strip().upper(),
            "subtype": "-".join(s["name"] for s in sub).upper() if sub else "",
            "period": int(e["period"]),
            "start": int(e["start"]["frame"]) - 1,
            "end": int(e["end"]["frame"]) - 1 if e["end"]["frame"] is not None else -1,
        })
    return _tidy(pd.DataFrame(rows))


def _tidy(events):
    events["end"] = np.maximum(events["end"], events["start"])
    return events.sort_values(["start"], kind="stable").reset_index(drop=True)


def is_restart(events):
    """Set pieces and goal kicks (which Metrica stores as a PASS with subtype GOAL KICK)."""
    return (events["type"] == "SET PIECE") | (
        (events["type"] == "PASS") & events["subtype"].str.contains("GOAL KICK"))


def event_labels(events, n):
    """Per-frame labels H/A/D/X from the events (see the module docstring)."""
    labels = np.full(n, "X", dtype="<U1")
    for period, ev in events.groupby("period", sort=True):
        ev = ev.reset_index(drop=True)
        lo, hi = int(ev["start"].min()), min(n, int(ev["end"].max()) + 1)
        owner = ev[ev["type"].isin(OWNER_TYPES)]
        side = np.full(hi - lo, -1, dtype=np.int8)
        side[owner["start"].to_numpy() - lo] = owner["side"].to_numpy()   # last one per frame wins
        idx = np.maximum.accumulate(np.where(side >= 0, np.arange(hi - lo), -1))
        first = int(np.flatnonzero(side >= 0)[0])
        idx[idx < 0] = first
        side = side[idx]
        labels[lo:hi] = np.where(side == HOME, "H", "A")

        restart = is_restart(ev).to_numpy()
        for i in np.flatnonzero(restart):
            r_start = int(ev.at[i, "start"])
            before = ev[(ev["start"] < r_start) & ~ev["type"].isin(IGNORED_BEFORE_RESTART)]
            if before.empty:
                continue
            stop = int(before["end"].max())
            if stop < r_start:
                labels[stop:r_start] = "D"
    return labels


def home_sign_from_positions(players, team, labels, events):
    """Attack sign of the home team per frame (+1: attacks towards x = pitch length), from which
    half each team stands in during each period: the home block sits on the side it defends."""
    n = len(players)
    sign = np.ones(n, dtype=np.int8)
    for period, ev in events.groupby("period", sort=True):
        lo, hi = int(ev["start"].min()), min(n, int(ev["end"].max()) + 1)
        x = players[lo:hi, :, 0]
        with np.errstate(invalid="ignore"):
            home_x = np.nanmean(x[:, team == HOME])
            away_x = np.nanmean(x[:, team == AWAY])
        s = 1 if home_x < away_x else -1
        sign[lo:] = s   # later periods overwrite
    return sign


def load_game(root, number, pitch):
    """Everything for one sample game: players, ball, team, sign, labels, events."""
    folder = os.path.join(root, "data", "Sample_Game_{}".format(number))
    prefix = os.path.join(folder, "Sample_Game_{}".format(number))
    if number in (1, 2):
        players, ball, team = read_csv_game(prefix + "_RawTrackingData_Home_Team.csv",
                                            prefix + "_RawTrackingData_Away_Team.csv", pitch)
        events = read_csv_events(prefix + "_RawEventsData.csv")
    else:
        players, ball, team = read_epts_game(prefix + "_metadata.xml", prefix + "_tracking.txt",
                                             pitch)
        home_id = ET.parse(prefix + "_metadata.xml").getroot().find(".//Score").get("idLocalTeam")
        events = read_json_events(prefix + "_events.json", home_id)
    labels = event_labels(events, len(players))
    sign = home_sign_from_positions(players, team, labels, events)
    return {"players": players, "ball": ball, "team": team, "sign": sign, "labels": labels,
            "events": events}


# -- occlusion transfer ------------------------------------------------------------------------

def visible_counts(players, team, graded):
    """(m, 2) visible players of HOME and AWAY on each graded frame of the target match."""
    vis = ~np.isnan(players[graded, :, 0])
    return np.stack([vis[:, team == HOME].sum(1), vis[:, team == AWAY].sum(1)], axis=1)


def apply_occlusion(players, team, counts, rng, block_frames):
    """Copy of players where, on every frame, each team keeps only as many players as a target
    frame of the graded match shows (a contiguous stretch of `counts` from a random offset). Which
    players disappear is a random priority redrawn every `block_frames`, so dropouts last a while
    instead of flickering frame to frame."""
    out = players.copy()
    n, s = players.shape[:2]
    offset = int(rng.integers(len(counts)))
    target = counts[(offset + np.arange(n)) % len(counts)]
    blocks = np.arange(n) // block_frames
    priority = rng.random((blocks[-1] + 1, s))[blocks]           # (n, s)
    on = ~np.isnan(players[:, :, 0])
    for side in (HOME, AWAY):
        cols = np.flatnonzero(team == side)
        pr = np.where(on[:, cols], priority[:, cols], -1.0)
        rank = np.argsort(np.argsort(-pr, axis=1), axis=1)      # 0 = highest priority
        drop = on[:, cols] & (rank >= target[:, side][:, None])
        sub = out[:, cols]
        sub[drop] = np.nan
        out[:, cols] = sub
    return out
