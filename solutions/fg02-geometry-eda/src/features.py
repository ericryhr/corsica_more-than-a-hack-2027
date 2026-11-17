"""Per-frame features from the ATD and the Smart Tagging, one value per video frame.

Coordinates are metres, rotated so that +x is always the direction Barcelona attacks: the second
half is turned 180 degrees. A feature listed in SIGNED is oriented toward the home team: positive
values favour H, negative values favour A. Everything else has no team side.

The tracking counts frames from 1 and submissions count from 0, so row i of the raw file is video
frame i. Missing observations are NaN, for players and ball alike.
"""

import importlib.util
import os
import warnings

import numpy as np
import pandas as pd

FPS = 25
HOME, AWAY = 0, 1
TOUCH_RADIUS = 1.5  # metres; a player this close to the ball counts as touching it
VELOCITY_STEP = 2  # frames on each side for finite differences
MAX_PLAYER_SPEED = 12.0  # m/s; faster means a slot swapped identity
MAX_BALL_SPEED = 40.0

# Smart Tagging codes as features. Attacking and zone codes carry the team that has the ball; the
# defending twins duplicate the attacking codes with the teams swapped, so they are left out.
PHASE_CODES = {"BUILD UP": "tag_build_up", "PROGRESSION": "tag_progression",
               "LONG BALL": "tag_long_ball", "CREATING CHANCES": "tag_creating",
               "ATTACKING TRANSITION": "tag_att_transition"}
ZONE_CODES = {"BALL IN FINAL THIRD": "tag_final_third", "BALL IN THE BOX": "tag_ball_box",
              "PLAYERS IN THE BOX": "tag_players_box"}
RELIABLE = ("BUILD UP", "PROGRESSION", "LONG BALL")  # owner agrees 84-98% with the hand labels

SIGNED = {
    "ball_x", "ball_vx", "near_team", "d_diff", "top3_home", "top5_home", "n5_diff", "n10_diff",
    "cx_mean", "cx_diff", "len_diff", "wid_diff", "vx_mean", "spd_diff", "tag_phase",
    "tag_reliable", "tag_zone", "tag_set_piece_team",
    "near_team_r1", "near_team_r3", "top3_home_r1", "d_diff_r1", "vx_mean_r1", "cx_mean_r1",
} | set(PHASE_CODES.values()) | set(ZONE_CODES.values())

GROUPS = {
    "ball": ["ball_seen", "ball_gap_s", "ball_x", "ball_y_abs", "ball_out", "ball_edge",
             "ball_speed", "ball_vx", "ball_turn", "ball_accel"],
    "proximity": ["near_team", "near_dist", "near_speed", "d_home", "d_away", "d_diff",
                  "top3_home", "top5_home", "n5_diff", "n10_diff", "n5_total"],
    "shape": ["n_home", "n_away", "cx_mean", "cx_diff", "len_home", "len_away", "len_diff",
              "wid_home", "wid_away", "wid_diff", "vx_home", "vx_away", "vx_mean",
              "spd_home", "spd_away", "spd_diff"],
    "tags": ["tag_any", "tag_phase", "tag_reliable", "tag_zone", "tag_set_piece",
             "tag_set_piece_team", "tag_goals", "tag_age_s", "tag_edge_s"]
            + list(PHASE_CODES.values()) + list(ZONE_CODES.values()),
    "context": ["near_team_r1", "near_team_r3", "top3_home_r1", "d_diff_r1", "vx_mean_r1",
                "cx_mean_r1"],
}


def fg01_signals():
    """The Smart Tagging and team parsers of fg01-smart-tagging, so both solutions read the same
    intervals. Loaded by path because every solution names its package `src`."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__)))), "fg01-smart-tagging", "src", "signals.py")
    spec = importlib.util.spec_from_file_location("fg01_signals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_atd(raw_path, n_frames, n_slots):
    """(players, ball): players (n_frames, n_slots, 3) and ball (n_frames, 3) as x, y, speed in
    the file's units (x and y in 0..1). The file is parsed in one pass by turning every separator
    into a comma."""
    with open(raw_path) as handle:
        text = handle.read()
    flat = text.replace(":", ",").replace(";", ",").replace("\n", ",").rstrip(",")
    width = 1 + 3 * n_slots + 3
    table = np.array(flat.split(","), dtype=np.float32).reshape(-1, width)
    frames = table[:, 0].astype(np.int64) - 1
    keep = (frames >= 0) & (frames < n_frames)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players[frames[keep]] = table[keep, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[frames[keep]] = table[keep, -3:]
    return players, ball


def orient(xy, pitch, second_half):
    """Metres, with the second half turned 180 degrees so +x is Barcelona's attack."""
    out = xy * np.asarray(pitch, dtype=np.float32)
    out[second_half] = np.asarray(pitch, dtype=np.float32) - out[second_half]
    return out


def velocity(xy, max_speed):
    """Central difference in m/s over VELOCITY_STEP frames on each side; NaN above max_speed."""
    v = np.full_like(xy, np.nan)
    k = VELOCITY_STEP
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    speed = np.hypot(v[..., 0], v[..., 1])
    v[speed > max_speed] = np.nan
    return v


def rolling_mean(values, seconds):
    """Centred mean over `seconds`, ignoring NaN. Uses the future, which is fine offline."""
    window = int(seconds * FPS) | 1
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(
        dtype=np.float32)


def team_stat(values, mask, fn):
    """fn over the slots of one team per frame, NaN where the team has no tracked slot."""
    out = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows give NaN, as intended
        return fn(out, axis=1).astype(np.float32)


def ball_features(ball, pitch):
    seen = ~np.isnan(ball[:, 0])
    n = len(ball)
    index = np.arange(n)
    last_seen = np.maximum.accumulate(np.where(seen, index, -10 ** 9))
    gap = np.where(last_seen < 0, np.nan, index - last_seen) / FPS

    v = velocity(ball, MAX_BALL_SPEED)
    v[~seen] = np.nan  # the difference spans t-2..t+2 and would exist on frames with no ball
    speed = np.hypot(v[:, 0], v[:, 1])
    accel = np.full(n, np.nan, dtype=np.float32)
    k = VELOCITY_STEP
    accel[k:-k] = np.abs(speed[2 * k:] - speed[:-2 * k])

    # Angle between the ball's movement over the 3 frames before and the 3 frames after.
    turn = np.full(n, np.nan, dtype=np.float32)
    before = ball[3:-3] - ball[:-6]
    after = ball[6:] - ball[3:-3]
    nb, na = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):  # still ball: masked just below
        cos = np.sum(before * after, axis=1) / (nb * na)
    moving = (nb > 0.1) & (na > 0.1)
    turn[3:-3] = np.where(moving, np.degrees(np.arccos(np.clip(cos, -1, 1))), np.nan)

    length, width = pitch
    edge = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    return {
        "ball_seen": seen.astype(np.float32),
        "ball_gap_s": gap.astype(np.float32),
        "ball_x": ball[:, 0] - length / 2,  # + is Barcelona's attacking half
        "ball_y_abs": np.abs(ball[:, 1] - width / 2),
        "ball_out": np.where(seen, (edge < 0).astype(np.float32), np.nan),
        "ball_edge": edge,
        "ball_speed": speed,
        "ball_vx": v[:, 0],
        "ball_turn": turn,
        "ball_accel": accel,
    }


def proximity_features(players, speeds, ball, team):
    """Who is near the ball. All NaN where the ball is missing."""
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    is_home = (team == HOME)[None, :]
    d_home = np.nanmin(np.where(is_home, d, np.inf), axis=1)
    d_away = np.nanmin(np.where(~is_home, d, np.inf), axis=1)
    d_home[np.isinf(d_home)] = np.nan
    d_away[np.isinf(d_away)] = np.nan

    order = np.argsort(np.where(np.isnan(d), np.inf, d), axis=1)
    ranked_home = np.take_along_axis(np.broadcast_to(is_home, d.shape), order, axis=1)
    ranked_valid = np.take_along_axis(~np.isnan(d), order, axis=1)
    nearest = order[:, 0]
    valid = ranked_valid[:, 0]

    def top_share(k):
        ok = ranked_valid[:, :k]
        share = (ranked_home[:, :k] & ok).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)
        return np.where(ok.sum(axis=1) >= k, 2 * share - 1, np.nan).astype(np.float32)

    within5 = d < 5
    within10 = d < 10
    near_team = np.where(team[nearest] == HOME, 1.0, -1.0)
    near_dist = np.take_along_axis(d, nearest[:, None], axis=1)[:, 0]
    near_speed = np.take_along_axis(speeds, nearest[:, None], axis=1)[:, 0]
    out = {
        "near_team": np.where(valid, near_team, np.nan),
        "near_dist": np.where(valid, near_dist, np.nan),
        "near_speed": np.where(valid, near_speed, np.nan),
        "d_home": d_home,
        "d_away": d_away,
        "d_diff": d_away - d_home,
        "top3_home": top_share(3),
        "top5_home": top_share(5),
        "n5_diff": (within5 & is_home).sum(1) - (within5 & ~is_home).sum(1),
        "n10_diff": (within10 & is_home).sum(1) - (within10 & ~is_home).sum(1),
        "n5_total": within5.sum(1),
    }
    no_ball = np.isnan(ball[:, 0])
    for key in ("n5_diff", "n10_diff", "n5_total"):
        out[key] = np.where(no_ball | ~valid, np.nan, out[key]).astype(np.float32)
    return {key: value.astype(np.float32) for key, value in out.items()}


def shape_features(players, speeds, team, pitch):
    """Team shape and movement, which exist with or without the ball."""
    seen = ~np.isnan(players[..., 0])
    is_home = (team == HOME)[None, :]
    home, away = seen & is_home, seen & ~is_home
    x, y = players[..., 0], players[..., 1]
    v = velocity(players, MAX_PLAYER_SPEED)

    cx_home, cx_away = team_stat(x, home, np.nanmean), team_stat(x, away, np.nanmean)

    def extent(mask):
        hi = team_stat(x, mask, lambda a, axis: np.nanpercentile(a, 90, axis=axis))
        lo = team_stat(x, mask, lambda a, axis: np.nanpercentile(a, 10, axis=axis))
        return hi - lo

    len_home, len_away = extent(home), extent(away)
    wid_home, wid_away = team_stat(y, home, np.nanstd), team_stat(y, away, np.nanstd)
    vx_home, vx_away = team_stat(v[..., 0], home, np.nanmean), team_stat(v[..., 0], away, np.nanmean)
    spd_home, spd_away = team_stat(speeds, home, np.nanmean), team_stat(speeds, away, np.nanmean)
    return {
        "n_home": home.sum(1).astype(np.float32),
        "n_away": away.sum(1).astype(np.float32),
        # Both teams move up the pitch when one of them attacks. 0 is the halfway line.
        "cx_mean": (cx_home + cx_away) / 2 - pitch[0] / 2,
        "cx_diff": cx_home - cx_away,
        "len_home": len_home, "len_away": len_away, "len_diff": len_home - len_away,
        "wid_home": wid_home, "wid_away": wid_away, "wid_diff": wid_home - wid_away,
        "vx_home": vx_home, "vx_away": vx_away, "vx_mean": (vx_home + vx_away) / 2,
        "spd_home": spd_home, "spd_away": spd_away, "spd_diff": spd_home - spd_away,
    }


def tag_features(tags, n_frames, home, away, nobody, codes):
    """Smart Tagging intervals as per-frame values. Team-carrying codes are +1 for H, -1 for A."""
    out = {name: np.zeros(n_frames, dtype=np.float32) for name in GROUPS["tags"]}
    sign = {home: 1.0, away: -1.0, nobody: 0.0}
    phase_start = np.full(n_frames, -10 ** 9, dtype=np.int64)
    edges = []
    for start, end, code, team, owner in zip(tags["start"], tags["end"], tags["code"],
                                             tags["team"], tags["owner"]):
        name = codes[code]
        s = slice(start, min(end, n_frames))
        edges += [start, end]
        if name in PHASE_CODES or name in ZONE_CODES:
            out["tag_any"][s] = 1
        if name in PHASE_CODES:
            out[PHASE_CODES[name]][s] += sign[owner]
            out["tag_phase"][s] += sign[owner]
            if name in RELIABLE:
                out["tag_reliable"][s] += sign[owner]
            phase_start[s] = np.maximum(phase_start[s], start)
        elif name in ZONE_CODES:
            out[ZONE_CODES[name]][s] += sign[owner]
            out["tag_zone"][s] += sign[owner]
        elif name == "SET PIECES":
            out["tag_set_piece"][s] = 1
            out["tag_set_piece_team"][s] += sign[team]
        elif name == "GOALS":
            out["tag_goals"][s] = 1
    index = np.arange(n_frames)
    out["tag_age_s"] = np.where(phase_start >= 0, (index - phase_start) / FPS, np.nan).astype(
        np.float32)
    edges = np.unique(np.clip(edges, 0, n_frames - 1))
    pos = np.clip(np.searchsorted(edges, index), 1, len(edges) - 1)
    nearest = np.minimum(np.abs(index - edges[pos - 1]), np.abs(edges[pos] - index))
    out["tag_edge_s"] = (nearest / FPS).astype(np.float32)
    return out


def build(meta_path, raw_path, tags_path, n_frames, pitch, second_half_start):
    """All feature groups as {name: (n_frames,) float32}."""
    sig = fg01_signals()
    team = sig.team_of_each_track(meta_path)
    raw_players, raw_ball = read_atd(raw_path, n_frames, len(team))
    second_half = np.arange(n_frames) >= second_half_start
    players = orient(raw_players[..., :2], pitch, second_half)
    ball = orient(raw_ball[:, :2], pitch, second_half)

    features = {}
    features.update(ball_features(ball, pitch))
    features.update(proximity_features(players, raw_players[..., 2], ball, team))
    features.update(shape_features(players, raw_players[..., 2], team, pitch))
    tags = sig.read_tags(tags_path, FPS)
    features.update(tag_features(tags, n_frames, sig.HOME, sig.AWAY, sig.NOBODY, sig.CODES))

    features["near_team_r1"] = rolling_mean(features["near_team"], 1)
    features["near_team_r3"] = rolling_mean(features["near_team"], 3)
    features["top3_home_r1"] = rolling_mean(features["top3_home"], 1)
    features["d_diff_r1"] = rolling_mean(features["d_diff"], 1)
    features["vx_mean_r1"] = rolling_mean(features["vx_mean"], 1)
    features["cx_mean_r1"] = rolling_mean(features["cx_mean"], 1)
    missing = [name for names in GROUPS.values() for name in names if name not in features]
    if missing:
        raise RuntimeError("features not computed: {}".format(missing))
    return features, tags
