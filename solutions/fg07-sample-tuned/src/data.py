"""Matches as shared/sample.py's dict, for the sample games and for BAR-ATH alike.

  sample_match(game, setting)  a Metrica sample game, degraded as shared/sample_bench.py does it
  bar_ath_match(game)          the challenge ATD in the same arrays, with no labels
  oriented(match)              players and ball turned so the home team attacks +x in every half

The dict holds players (F, S, 2) and ball (F, 2) in metres, team (S,) 0 home / 1 away, speed
(F, S) in m/s computed from positions, halves [(first, last)] frame indices, fps and pitch. The
sample games also hold `labels` ('H'/'A'/'D'/'X'); BAR-ATH holds None there, so nothing that
predicts can read labels by mistake.

Speed is computed from positions for both, with shared/sample.py's player_speed, so a model fitted
on the sample sees the same quantity on BAR-ATH (the ATD's own speed channel is empty for the first
frames of every track fragment).
"""

import os

import numpy as np

from shared.challenge import GAMES, require, source_frames
from games import half_video_blocks  # the challenge's games.py, on sys.path via shared.challenge
from shared.sample import degrade, load_match, player_speed
from shared.sample_bench import SETTINGS

from .peers import peer

HOME, AWAY = 0, 1
KICKOFF_S = 5  # seconds after the start of a half used to see which side each team defends


def sample_match(game, setting):
    match = load_match(game)
    params = SETTINGS[setting]
    return match if params is None else degrade(match, seed=game, **params)


def bar_ath_match(game, cache_dir):
    """The ATD of `game` as a match dict, cached in cache_dir/tracking.npz."""
    path = os.path.join(cache_dir, "tracking.npz")
    info = GAMES[game]
    if not os.path.isfile(path):
        meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
        team = peer("fg01", "signals").team_of_each_track(meta_path)
        n_frames = source_frames("tactical", game)
        raw_players, raw_ball = peer("fg02", "features").read_atd(raw_path, n_frames, len(team))
        scale = np.asarray(info["pitch"], dtype=np.float32)
        players = (raw_players[..., :2] * scale).astype(np.float32)
        ball = (raw_ball[:, :2] * scale).astype(np.float32)
        np.savez(path, players=players, ball=ball, team=team)
    data = np.load(path)
    players = data["players"]
    halves = [(first, last) for _, first, last in half_video_blocks(game)]
    kickoffs = [video_frame(info[key], info["fps"]) for key in ("kickoff", "kickoff_2nd")]
    return {"game": game, "players": players, "team": data["team"], "ball": data["ball"],
            "speed": player_speed(players), "labels": None, "halves": halves,
            "kickoffs": kickoffs, "fps": info["fps"], "pitch": tuple(info["pitch"])}


def video_frame(clock, fps):
    """Video frame of a games.py {"minute", "second", "frame"} time of the tactical video."""
    return (clock["minute"] * 60 + clock["second"]) * fps + clock["frame"]


def attacks_right(match):
    """Per half, True if the home team attacks +x. At kick-off each team stands in its own half,
    so the team whose players are further left in the first seconds defends the left goal.

    The kick-off is the first frame of the half in the sample games. In BAR-ATH the second half
    is graded from 110,808 but kicks off at 113,742 (the challenge FAQ), so `kickoffs` holds the
    real frames; before them the players are still walking to their places."""
    players, team, fps = match["players"], match["team"], match["fps"]
    kickoffs = match.get("kickoffs") or [first for first, _ in match["halves"]]
    out = []
    for (first, last), kickoff in zip(match["halves"], kickoffs):
        window = players[kickoff:min(kickoff + KICKOFF_S * fps, last + 1), :, 0]
        with np.errstate(invalid="ignore"):
            home = np.nanmean(window[:, team == HOME])
            away = np.nanmean(window[:, team == AWAY])
        out.append(bool(home < away))
    return out


def oriented(match):
    """(players, ball) copies with every half where home attacks -x turned 180 degrees, so +x is
    always the home team's attack, as fg02's features expect."""
    players, ball = match["players"].copy(), match["ball"].copy()
    pitch = np.asarray(match["pitch"], dtype=np.float32)
    for (first, last), right in zip(match["halves"], attacks_right(match)):
        if not right:
            players[first:last + 1] = pitch - players[first:last + 1]
            ball[first:last + 1] = pitch - ball[first:last + 1]
    return players, ball


def graded(match):
    """Boolean mask of the frames inside the halves."""
    mask = np.zeros(len(match["ball"]), dtype=bool)
    for first, last in match["halves"]:
        mask[first:last + 1] = True
    return mask
