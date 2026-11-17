"""Metrica's public sample games as extra matches, in the same arrays the solutions read from the ATD.

    from shared.sample import load_match, GAMES_SAMPLE
    match = load_match(1)      # dict: players, team, ball, speed, labels, halves, fps, pitch
    touches = load_touches(1)  # DataFrame: frame, team, kind, change (see touches_from_events)

    python shared/sample.py    # caches the 3 games and writes artifacts/sample/gt_game<N>.csv

Three anonymised full matches (github.com/metrica-sports/sample-data, public, allowed as external
data) with corrected tracking and manual events. They have no possession or ball-state channel, so
the H/A/D labels are rebuilt from the events:

  owner  team of the last event that implies having the ball (PASS, CARRY, SET PIECE, RECOVERY,
         SHOT, BALL LOST, BALL OUT, FAULT RECEIVED). Loose balls stay with the last owner, as
         CHALLENGE.md says. CHALLENGE and CARD never change the owner.
  dead   from a stoppage (ball out, shot out or goal, foul, offside, referee stop) to the restart
         after it (SET PIECE, or a PASS with subtype GOAL KICK).
  then   one-team flickers shorter than MIN_POSSESSION_S, between two runs of the other team, are
         merged into it. With 2 s the possessions have the shape of the public ground truth
         (median 7.4-8.2 s vs 7.6 s; 10-15% shorter than 2 s vs 11.8%).

These are an approximation of the challenge ground truth, which comes from Metrica's corrected
tracking (CHALLENGE.md). The labels' D agrees with "ball missing in the corrected tracking" on
92-95% of frames.

Arrays follow solutions/nc02-frozen-width (read_tracking, team_of_each_track): index = frame - 1,
like a video frame; players (n_frames, n_slots, 2) in metres with NaN where missing; team per slot
0 home / 1 away; ball (n_frames, 2). `speed` (n_frames, n_slots, m/s) is computed from positions,
because the sample has no speed column. One slot per player, so slots never swap people: this
tracking is much cleaner than the challenge ATD.

The data folder is CORSICA_SAMPLE_DIR, else the git submodule repo/sample-data, else
data/external/metrica-sample-data. To get the submodule:

    git submodule update --init repo/sample-data
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.challenge import ROOT, artifacts_dir, write_submission  # noqa: E402

_SUBMODULE = os.path.join(ROOT, "repo", "sample-data")
SAMPLE_DIR = os.environ.get("CORSICA_SAMPLE_DIR") or (
    _SUBMODULE if os.path.isdir(os.path.join(_SUBMODULE, "data"))
    else os.path.join(ROOT, "data", "external", "metrica-sample-data"))
GAMES_SAMPLE = (1, 2, 3)
PITCH = (105.0, 68.0)                 # the sample README: both games are 105 x 68 m
FPS = 25
HOME, AWAY = 0, 1                     # as in nc02-frozen-width/src/combo.py
MIN_POSSESSION_S = 2
MAX_SPEED = 12.0                      # m/s; faster is a tracking jump, not a player

OWNER_TYPES = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT", "FAULT RECEIVED"}


def game_dir(game):
    path = os.path.join(SAMPLE_DIR, "data", "Sample_Game_%d" % game)
    if not os.path.isdir(path):
        sys.exit("{} not found. Clone the sample data (see shared/sample.py) or set CORSICA_SAMPLE_DIR."
                 .format(path))
    return path


# --- events -----------------------------------------------------------------------------------

def _events_csv(game):
    raw = pd.read_csv(os.path.join(game_dir(game), "Sample_Game_%d_RawEventsData.csv" % game))
    return pd.DataFrame({
        "team": raw.Team.map({"Home": "H", "Away": "A"}),
        "type": raw.Type.str.upper(),
        "subtype": raw.Subtype.fillna("").str.upper(),
        "period": raw.Period,
        "start": raw["Start Frame"].astype(int) - 1,
        "end": raw["End Frame"].astype(int) - 1,
    })


def _subtype_names(subtypes):
    if subtypes is None:
        return ""
    if isinstance(subtypes, dict):
        subtypes = [subtypes]
    return "-".join(s["name"].upper() for s in subtypes)


def _events_json(game):
    with open(os.path.join(game_dir(game), "Sample_Game_%d_events.json" % game)) as handle:
        data = json.load(handle)["data"]
    return pd.DataFrame([{
        "team": "H" if e["team"]["name"] == "Team A" else "A",      # Team A is home in the metadata
        "type": e["type"]["name"].upper(),
        "subtype": _subtype_names(e["subtypes"]),
        "period": e["period"],
        "start": int(e["start"]["frame"]) - 1,
        "end": int(e["end"]["frame"]) - 1,
    } for e in data])


def load_events(game):
    """Events with start/end as frame indices (frame - 1), sorted by start."""
    events = _events_json(game) if game == 3 else _events_csv(game)
    events["end"] = np.maximum(events.end, events.start)        # End Frame is 0 on instant events
    return events.sort_values("start", kind="stable").reset_index(drop=True)


def _is_restart(e):
    return e.type == "SET PIECE" or (e.type == "PASS" and "GOAL KICK" in e.subtype)


def _stop_frame(e):
    """Frame where play stops because of this event, or None if it does not stop play."""
    if e.type == "BALL OUT":
        return e.end                      # start = last touch, end = ball out (~2 s later)
    if e.type == "SHOT" and ("OUT" in e.subtype or "GOAL" in e.subtype):
        return e.end
    if e.type == "FAULT RECEIVED":
        return e.start
    if e.type == "CHALLENGE" and "FAULT" in e.subtype and "ADVANTAGE" not in e.subtype:
        return e.start
    if e.type == "BALL LOST" and any(k in e.subtype for k in ("OFFSIDE", "FORCED", "HAND", "END HALF")):
        return e.end
    if e.type == "PASS" and "OFFSIDE" in e.subtype:
        return e.end
    return None


def halves_from_events(events, n_frames):
    """[(first, last)] frame of each period, kickoff to its last event (inclusive), like the
    challenge's graded halves."""
    return [(int(period.start.min()), int(min(period.end.max(), n_frames - 1)))
            for _, period in events.groupby("period")]


def labels_from_events(events, n_frames):
    """'H'/'A'/'D' per frame from kickoff to the last event of each period, 'X' elsewhere."""
    changes = []                          # (frame, team) every time an event names the owner
    dead = np.zeros(n_frames, bool)
    graded = np.zeros(n_frames, bool)
    for first, last in halves_from_events(events, n_frames):
        graded[first:last + 1] = True

    dead_from, last_live_end = None, None
    for e in events.itertuples():
        if _is_restart(e):
            start = dead_from if dead_from is not None else last_live_end
            if start is not None and start < e.start and e.start - start < 120 * FPS:
                dead[start:e.start] = True
            dead_from = e.start if "RETAKEN" in e.subtype else None
            last_live_end = e.end
            changes.append((e.start, e.team))
            continue
        cue = _stop_frame(e)
        if cue is not None:
            if dead_from is None:
                dead_from = cue
            if e.type in OWNER_TYPES:
                changes.append((e.start, e.team))
            continue
        if e.type in ("CARD", "CHALLENGE") or (e.type == "RECOVERY" and dead_from is not None):
            continue                      # no owner change; a keeper collecting a dead ball is not a restart
        if dead_from is not None and e.start > dead_from:
            dead[dead_from:e.start] = True    # play resumed without a tagged restart (e.g. dropped ball)
        dead_from = None
        last_live_end = e.end
        if e.type in OWNER_TYPES:
            changes.append((e.start, e.team))

    starts = np.array([frame for frame, _ in changes])
    teams = np.array([team for _, team in changes], dtype=object)
    last = np.searchsorted(starts, np.arange(n_frames), side="right") - 1
    owner = np.where(last >= 0, teams[np.maximum(last, 0)], "X")
    labels = np.where(dead, "D", owner).astype(object)
    labels[~graded | (labels == "X")] = "X"
    return labels


def merge_flickers(labels, min_frames):
    """H A H -> H H H when the middle run is shorter than min_frames (no D in between)."""
    labels = labels.copy()
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    for i in range(1, len(starts) - 1):
        before, here, after = labels[starts[i - 1]], labels[starts[i]], labels[starts[i + 1]]
        if here in ("H", "A") and before == after and before in ("H", "A") and before != here \
                and ends[i] - starts[i] < min_frames:
            labels[starts[i]:ends[i]] = before
    return labels


# --- touches ----------------------------------------------------------------------------------
#
# The labels' owner changes at events, so the touches behind those events are the ground truth
# for a touch detector (shared/touch_bench.py scores one).

TOUCH_KINDS = ("win", "receive", "kick", "carry", "challenge")    # priority when two coincide
SCORED_KINDS = ("kick", "receive", "win")    # in all 3 games; CARRY exists only in game 3
TOUCH_MERGE = 3                               # frames: one touch, e.g. a SET PIECE and its PASS


def touches_from_events(events):
    """Ball touches from the manual events: DataFrame frame, team ('H'/'A'), kind, change.

      kick       the ball leaves a player: start of PASS, SHOT, BALL LOST (a pass, cross or
                 clearance that went to the other team or out), SET PIECE
      receive    a completed PASS arrives: its end frame, team of the passer
      win        RECOVERY: a player wins the ball (interception, theft, block, save)
      carry      start and end of a CARRY (game 3 only)
      challenge  a CHALLENGE; the event's team did not always touch the ball

    Touches of the same team within TOUCH_MERGE frames are one; the kind kept is the first in
    TOUCH_KINDS (a RECOVERY and the BALL LOST right after it stay a win).
    `change` is True where a scored touch's team differs from the previous scored touch's: where
    possession can pass to the other team."""
    rows = []
    for e in events.itertuples():
        if e.type in ("PASS", "SHOT", "BALL LOST", "SET PIECE"):
            rows.append((e.start, e.team, "kick"))
        if e.type == "PASS" and e.end > e.start:
            rows.append((e.end, e.team, "receive"))
        if e.type == "RECOVERY":
            rows.append((e.start, e.team, "win"))
        if e.type == "CARRY":
            rows += [(e.start, e.team, "carry"), (e.end, e.team, "carry")]
        if e.type == "CHALLENGE":
            rows.append((e.start, e.team, "challenge"))
    touches = pd.DataFrame(rows, columns=["frame", "team", "kind"])
    touches["order"] = touches.kind.map({k: i for i, k in enumerate(TOUCH_KINDS)})
    touches = touches.sort_values(["frame", "order"], kind="stable").drop(columns="order").reset_index(drop=True)

    keep, last = [], {}
    for i, t in enumerate(touches.itertuples()):
        key = (t.team, t.kind == "challenge")
        if key in last and t.frame - last[key] <= TOUCH_MERGE:
            continue
        last[key] = t.frame
        keep.append(i)
    touches = touches.iloc[keep].reset_index(drop=True)

    scored = touches.kind.isin(SCORED_KINDS)
    previous = touches.team.where(scored).ffill().shift()
    touches["change"] = scored & previous.notna() & (touches.team != previous)
    return touches


def load_touches(game):
    return touches_from_events(load_events(game))


# --- tracking ---------------------------------------------------------------------------------

def _tracking_csv(game):
    """(frames, players (n, k, 2) normalised, team (k,), ball (n, 2) normalised) for games 1-2."""
    sides = []
    for side, code in (("Home", HOME), ("Away", AWAY)):
        raw = pd.read_csv(os.path.join(game_dir(game), "Sample_Game_%d_RawTrackingData_%s_Team.csv"
                                       % (game, side)), skiprows=2)
        names = list(raw.columns)        # Period, Frame, Time [s], PlayerN (x), unnamed (y), ..., Ball, unnamed
        columns = [i for i, c in enumerate(names) if c.startswith("Player")]
        xy = np.stack([raw.iloc[:, [i, i + 1]].to_numpy(float) for i in columns], axis=1)
        ball_at = names.index("Ball")
        sides.append((raw.Frame.to_numpy(int), xy, np.full(len(columns), code),
                      raw.iloc[:, [ball_at, ball_at + 1]].to_numpy(float)))
    (frames, home, home_team, ball), (frames_away, away, away_team, _) = sides
    if not (frames == frames_away).all():
        raise ValueError("game %d: home and away tracking have different frames" % game)
    return frames, np.concatenate([home, away], axis=1), np.concatenate([home_team, away_team]), ball


def _tracking_epts(game):
    """Same for game 3, through kloppy (substitutions change the channel layout mid-file)."""
    from kloppy import metrica
    folder = game_dir(game)
    dataset = metrica.load_tracking_epts(meta_data=os.path.join(folder, "Sample_Game_%d_metadata.xml" % game),
                                         raw_data=os.path.join(folder, "Sample_Game_%d_tracking.txt" % game))
    df = dataset.to_df()
    columns, team = [], []
    for side in dataset.metadata.teams:
        code = HOME if side.ground.value == "home" else AWAY
        for player in side.players:
            if "%s_x" % player.player_id in df.columns:
                columns.append(player.player_id)
                team.append(code)
    xy = np.stack([df[["%s_x" % p, "%s_y" % p]].to_numpy(float) for p in columns], axis=1)
    return df.frame_id.to_numpy(int), xy, np.array(team), df[["ball_x", "ball_y"]].to_numpy(float)


def player_speed(players):
    """m/s per slot, centred difference over +-2 frames; NaN at the edges and above MAX_SPEED."""
    speed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    speed[2:-2] = np.linalg.norm(players[4:] - players[:-4], axis=2) / (4.0 / FPS)
    speed[speed > MAX_SPEED] = np.nan
    return speed


def _cache_path(game):
    return os.path.join(artifacts_dir("sample"), "game%d.npz" % game)


def load_match(game, min_possession_s=MIN_POSSESSION_S):
    """One sample game in the solutions' format. Cached in artifacts/sample/game<N>.npz."""
    path = _cache_path(game)
    if os.path.isfile(path):
        cached = np.load(path)
        players, team, ball = cached["players"], cached["team"], cached["ball"]
    else:
        frames, xy, team, ball_xy = _tracking_epts(game) if game == 3 else _tracking_csv(game)
        n_frames = int(frames.max())
        scale = np.array(PITCH, dtype=np.float32)
        players = np.full((n_frames, xy.shape[1], 2), np.nan, dtype=np.float32)
        ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
        players[frames - 1] = xy * scale
        ball[frames - 1] = ball_xy * scale
        team = team.astype(np.int8)
        np.savez(path, players=players, team=team, ball=ball)

    events = load_events(game)
    labels = labels_from_events(events, len(ball))
    if min_possession_s:
        labels = merge_flickers(labels, int(min_possession_s * FPS))
    return {"game": game, "players": players, "team": team, "ball": ball,
            "speed": player_speed(players), "labels": labels, "halves": halves_from_events(events, len(ball)),
            "fps": FPS, "pitch": PITCH}


# --- make a sample game look like the challenge ATD -------------------------------------------
#
# Profile of the BAR-ATH ATD (playground/05_atd_profile.py, graded frames): ball missing 40.7% in
# 150 gaps, median 10.7 s, ~92% of the missing time judged dead ball from player speed; 10.2-10.3
# players per team per frame; positions as smooth as the corrected tracking; duplicates not above
# the sample's own. The clean sample already loses the ball mostly on dead balls, so what is left
# to imitate is: some live gaps, the ball sometimes visible during a stoppage, and missing players.

DEGRADE = {
    "live_gaps_per_min": 0.25,       # ball lost during live play (~25 gaps of 5-10 s in BAR-ATH)
    "live_gap_s": (5, 10),
    "dead_seen_share": 0.0,          # last part of each stoppage with the ball visible at the restart spot
    "player_missing": 0.07,          # share of player-frames lost (11 -> ~10.2 per team)
    "player_gap_s": 10,              # mean length of a lost stretch
    "ball_blur": 0,                  # frames: centred moving average of the ball (0 = off)
    "ball_error_m": 0.0,             # slowly drifting position error per axis (0 = off)
    "ball_drift": 50,                # frames over which that error drifts
    "ball_lead": 0,                  # frames the ball is shown ahead of the players
}
# A harsher setting for robustness. Calibration (playground/07_calibrate_degrade.py): every setting
# tried ranks the published baselines like the full match (w50 > w125 > w25 > raw), but scores stay
# 0.04-0.09 above it (w50 0.29-0.36 vs 0.24), so how much of a stoppage shows the ball is uncertain.
# DEGRADE matches the ATD profile (ball missing ~39% vs 40.7%); DEGRADE_HARSH is closest in score.
# Trust "A beats B" only when it holds under both.
DEGRADE_HARSH = dict(DEGRADE, live_gaps_per_min=0.5, dead_seen_share=0.6)
# The sample ball is rebuilt from the events (straight between touches, on the player at each one);
# the ATD ball is blurred over ~1 s, ~1.9 m from the nearest player and ~6 frames ahead of the
# players (playground/12_ball_profile.py, 13_ball_blur.py). These values match those totals
# (playground/14_calibrate_ball.py). Off in DEGRADE and DEGRADE_HARSH, so their results stand; on in
# DEGRADE_BLURRED, the setting for touch detectors (shared/touch_bench.py).
BALL_ATD = {"ball_blur": 25, "ball_error_m": 0.75, "ball_drift": 50, "ball_lead": 6}
DEGRADE_BLURRED = dict(DEGRADE, **BALL_ATD)
DEGRADE_HARSH_BLURRED = dict(DEGRADE_HARSH, **BALL_ATD)


def _runs_of(mask):
    """(starts, ends) of every True stretch, ends exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges[0::2], edges[1::2]


def degrade(match, seed=0, **overrides):
    """A copy of `match` with challenge-like tracking problems. Labels are untouched.

    Parameters default to DEGRADE; any of them can be overridden, e.g. degrade(m, dead_seen_share=0).
    """
    params = dict(DEGRADE, **overrides)
    rng = np.random.default_rng(seed)
    labels, fps = match["labels"], match["fps"]
    ball, players = match["ball"].copy(), match["players"].copy()
    n_frames = len(labels)

    # 1. the ball lost for a few seconds during live play
    live = np.flatnonzero(np.isin(labels, ["H", "A"]))
    minutes = np.sum(labels != "X") / fps / 60
    for start in rng.choice(live, min(rng.poisson(params["live_gaps_per_min"] * minutes), len(live)), replace=False):
        ball[start:start + int(rng.uniform(*params["live_gap_s"]) * fps)] = np.nan

    # 2. stoppages where the tracker still sees the ball: it rests where play will restart
    seen = ~np.isnan(match["ball"][:, 0])
    for start, end in zip(*_runs_of(labels == "D")):
        visible = int(round(params["dead_seen_share"] * (end - start)))
        if not visible:
            continue
        after = np.flatnonzero(seen[end:end + 2 * fps])
        before = np.flatnonzero(seen[max(0, start - 2 * fps):start])
        if len(after):
            spot = match["ball"][end + after[0]]
        elif len(before):
            spot = match["ball"][max(0, start - 2 * fps) + before[-1]]
        else:
            continue
        ball[end - visible:end] = spot

    # 3. players lost in stretches (exponential lengths), so ~player_missing of player-frames are gone
    mean_gap = params["player_gap_s"] * fps
    rate = params["player_missing"] / (mean_gap * (1 - params["player_missing"]))
    for slot in range(players.shape[1]):
        for start in np.flatnonzero(rng.random(n_frames) < rate):
            players[start:start + max(1, int(rng.exponential(mean_gap))), slot] = np.nan

    # 4. the ball as blurred, imprecise and early as the ATD's. Its own random stream, so turning it
    #    on leaves steps 1-3 as they were.
    if params["ball_blur"] or params["ball_error_m"] or params["ball_lead"]:
        ball = _messy_ball(ball, params, np.random.default_rng([seed, 4]))

    return dict(match, ball=ball, players=players, speed=player_speed(players), degraded=params, seed=seed)


def _messy_ball(ball, params, rng):
    seen = ~np.isnan(ball[:, 0])
    out = ball.astype(np.float64)
    if params["ball_blur"] > 1:
        out = pd.DataFrame(out).rolling(params["ball_blur"], center=True, min_periods=1).mean().to_numpy()
    if params["ball_error_m"]:
        smooth = lambda x: pd.DataFrame(x).rolling(params["ball_drift"], center=True, min_periods=1).mean().to_numpy()  # noqa: E731
        drift = smooth(smooth(rng.normal(0, 1, out.shape)))
        out = out + drift * (params["ball_error_m"] / drift.std(axis=0)) + rng.normal(0, 0.003, out.shape)
    lead = params["ball_lead"]
    if lead:
        out = np.concatenate([out[lead:], np.full((lead, 2), np.nan)])
        seen = np.concatenate([seen[lead:], np.zeros(lead, bool)])
    out[~seen] = np.nan
    return out.astype(np.float32)


def gt_path(game):
    return os.path.join(artifacts_dir("sample"), "gt_game%d.csv" % game)


if __name__ == "__main__":
    for game in GAMES_SAMPLE:
        match = load_match(game)
        labels = match["labels"]
        graded = labels[labels != "X"]
        write_submission(gt_path(game), labels)
        print("game {}: {} frames, {} slots ({} home), ball seen {:.1f}%, graded {:.1f} min, "
              "H {:.1f}% A {:.1f}% D {:.1f}%  -> {}".format(
                  game, len(labels), len(match["team"]), int((match["team"] == HOME).sum()),
                  100 * np.mean(~np.isnan(match["ball"][:, 0])), len(graded) / FPS / 60,
                  *(100 * np.mean(graded == k) for k in "HAD"), os.path.relpath(gt_path(game))))
