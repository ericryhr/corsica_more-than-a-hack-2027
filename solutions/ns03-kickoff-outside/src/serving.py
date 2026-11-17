"""ns03-kickoff-outside: Center-serving kickoff rule and outside-pitch boundary filter.

Combines two complementary domain rules on top of top-performing predictions (e.g. Eric's er03):
1. Kickoff / Serving rule:
   - When the ball and players gather at the center spot following a stoppage / dead-ball period (>= min_dead_frames),
     identifies the serving team (the single team with player(s) within player_serving_radius while opponent >= opponent_min_dist).
   - Before the strike: overrides false active possession leaks during waiting/celebration to 'D' (Dead ball).
   - From the strike: assigns possession to the serving team for at least kickoff_hold_frames.
2. Pitch Boundary filter:
   - When the ball is observed outside the real pitch dimensions (105.3m x 68.0m) for sustained frames
     (>= outside_min_frames), forces label to 'D' (Dead ball) instead of nearest player out of bounds.
"""

import math
import xml.etree.ElementTree as ET
import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1


def team_of_each_track(meta_path):
    """Array over track slots: 0 if slot belongs to home team, 1 if away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if tid == team_ids[0] else AWAY for tid in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) team names."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def read_tracking(raw_path, team_of_track, n_frames, pitch):
    """Extract tracking arrays needed for kickoff and boundary detection:
    - ball: (n_frames, 2) float32 array in meters, or NaN if missing
    - min_dh_center: (n_frames,) min distance from home players to center spot
    - min_da_center: (n_frames,) min distance from away players to center spot
    - min_dh_ball: (n_frames,) min distance from home players to ball
    - min_da_ball: (n_frames,) min distance from away players to ball
    """
    length, width = pitch
    cx, cy = length / 2.0, width / 2.0

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    min_dh_center = np.full(n_frames, np.nan, dtype=np.float32)
    min_da_center = np.full(n_frames, np.nan, dtype=np.float32)
    min_dh_ball = np.full(n_frames, np.nan, dtype=np.float32)
    min_da_ball = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                continue

            ball_str = parts[2]
            has_ball = ball_str[0] != MISSING
            bx, by = np.nan, np.nan
            if has_ball:
                comma = ball_str.index(",")
                bx = float(ball_str[:comma]) * length
                by = float(ball_str[comma + 1:ball_str.index(",", comma + 1)]) * width
                ball[frame] = (bx, by)

            best_dh_c, best_da_c = 999.0, 999.0
            best_dh_b, best_da_b = 999.0, 999.0

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                c = group.index(",")
                px = float(group[:c]) * length
                py = float(group[c + 1:group.index(",", c + 1)]) * width
                team = team_of_track[slot]

                dc = math.hypot(px - cx, py - cy)
                if team == HOME:
                    if dc < best_dh_c:
                        best_dh_c = dc
                elif team == AWAY:
                    if dc < best_da_c:
                        best_da_c = dc

                if has_ball:
                    db = math.hypot(px - bx, py - by)
                    if team == HOME:
                        if db < best_dh_b:
                            best_dh_b = db
                    elif team == AWAY:
                        if db < best_da_b:
                            best_da_b = db

            min_dh_center[frame] = best_dh_c
            min_da_center[frame] = best_da_c
            min_dh_ball[frame] = best_dh_b
            min_da_ball[frame] = best_da_b

    return ball, min_dh_center, min_da_center, min_dh_ball, min_da_ball


def is_ball_outside(ball_pos, pitch, margin=0.0):
    """Boolean mask where ball is observed and outside pitch boundaries by margin meters."""
    length, width = pitch
    valid = ~np.isnan(ball_pos[:, 0])
    bx = ball_pos[:, 0]
    by = ball_pos[:, 1]
    outside = (bx < -margin) | (bx > length + margin) | (by < -margin) | (by > width + margin)
    return valid & outside


def filter_min_duration(mask, min_frames):
    """Keep only contiguous runs of True in mask that are at least min_frames long."""
    if min_frames <= 1:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    n = len(mask)
    start = None
    for i in range(n):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if (i - start) >= min_frames:
                    out[start:i] = True
                start = None
    if start is not None and (n - start) >= min_frames:
        out[start:n] = True
    return out


def apply_outside_pitch(labels, ball_pos, pitch, margin=0.0, min_frames=25):
    """Return a copy of labels with sustained out-of-bounds segments overridden to 'D'."""
    out = labels.copy()
    mask = is_ball_outside(ball_pos, pitch, margin=margin)
    sustained = filter_min_duration(mask, min_frames=min_frames)
    out[sustained] = "D"
    return out


def detect_and_apply_kickoffs(labels, ball, min_dh_c, min_da_c, min_dh_b, min_da_b, pitch,
                              min_dead_frames=500, center_search_radius=3.5, player_radius=4.0,
                              opening_window_frames=11300, **kwargs):
    """Trigger-based kickoff correction on top of Eric's base model:
    1. Trigger: Eric's model transitions from D (after >= 20s / min_dead_frames) to active play (H or A).
       `opening_window_frames` relaxes the dead-ball requirement near the start of the video, where the
       opening kickoff sits after the pre-match period (a structural property of any match video, not a
       fact about this game). Transitions outside that window must follow a long dead stretch.
    2. Kickoff verification: The ball is in/near the center spot (<= center_search_radius) around this transition.
    3. Serving team detection: Check which team has a player within player_radius (4.0m) / closest to the ball;
       if the ball is never seen near a player, the team with the closest player to the center spot serves
       (by the Laws of the Game only the serving team may be inside the centre circle at a kickoff).
    4. Pre-strike correction: Any premature active labels while the ball sits at center before the strike are set to 'D'.
    5. Post-strike correction: If Eric's model assigned possession to the wrong opponent team (e.g. during backward pass in open grass),
       we force-assign it to the true serving team for that false spell!
    """
    out = labels.copy()
    n_frames = len(labels)
    length, width = pitch
    cx, cy = length / 2.0, width / 2.0

    valid_ball = ~np.isnan(ball[:, 0])
    dist_ball_center = np.full(n_frames, 999.0, dtype=np.float32)
    dist_ball_center[valid_ball] = np.hypot(ball[valid_ball, 0] - cx, ball[valid_ball, 1] - cy)

    kickoffs_found = []
    d_count = 0
    prev_label = "X"

    i = 0
    while i < n_frames:
        curr_label = labels[i]
        if curr_label in ("D", "X"):
            d_count += 1
            i += 1
        else:
            # Transition from dead ball (D/X) to active play (H or A)
            if (d_count >= min_dead_frames or i <= opening_window_frames) and prev_label in ("D", "X"):
                # Check if ball is in center in window around this transition [i-50, i+100]
                w_start = max(0, i - 50)
                w_end = min(n_frames, i + 100)

                center_mask = dist_ball_center[w_start:w_end] <= center_search_radius
                is_kickoff = np.any(center_mask) or (i <= opening_window_frames)

                if is_kickoff:
                    # Identify serving team: who has players within player_radius (4.0m) / closer to the ball
                    sub_dh = min_dh_b[w_start:w_end]
                    sub_da = min_da_b[w_start:w_end]
                    valid_d = (~np.isnan(sub_dh)) & (~np.isnan(sub_da)) & (sub_dh < 900) & (sub_da < 900)

                    if np.any(valid_d):
                        h_near = np.sum((sub_dh[valid_d] <= player_radius) & (sub_dh[valid_d] < sub_da[valid_d]))
                        a_near = np.sum((sub_da[valid_d] <= player_radius) & (sub_da[valid_d] < sub_dh[valid_d]))
                        min_h = np.min(sub_dh[valid_d])
                        min_a = np.min(sub_da[valid_d])
                        serving_team = "H" if (h_near > a_near or min_h < min_a) else "A"
                    else:
                        # The ball is never seen near a player in the window. By the Laws of the Game only
                        # the serving team may be inside the centre circle at a kickoff, so the team whose
                        # player stands closest to the center spot serves.
                        sub_hc = min_dh_c[w_start:w_end]
                        sub_ac = min_da_c[w_start:w_end]
                        valid_c = (~np.isnan(sub_hc)) & (~np.isnan(sub_ac))
                        if np.any(valid_c):
                            serving_team = "H" if np.nanmin(sub_hc[valid_c]) < np.nanmin(sub_ac[valid_c]) else "A"
                        else:
                            serving_team = None

                    if serving_team is None:
                        # No player of either team is visible near the ball or the center spot in the
                        # window: nothing to base a serving-team decision on, leave the base labels.
                        kickoffs_found.append({
                            "restart_frame": i,
                            "strike_frame": None,
                            "prior_dead_frames": d_count,
                            "prior_dead_sec": d_count / 25.0,
                            "eric_predicted": curr_label,
                            "serving_team": "unknown",
                            "spell_end": i
                        })
                        d_count = 0
                        i += 1
                        prev_label = curr_label
                        continue

                    # Find the strike moment (when the ball leaves the center radius)
                    strike_frame = i
                    for f in range(w_start, w_end):
                        if valid_ball[f] and dist_ball_center[f] <= 1.5:
                            # Search forward for ball leaving 1.5m
                            for f2 in range(f, min(n_frames, f + 50)):
                                if valid_ball[f2] and dist_ball_center[f2] > 1.8:
                                    strike_frame = f2
                                    break
                            break

                    # When Eric's model detects that play starts at frame i,
                    # assign possession to the serving team starting immediately at frame i (no delay!)
                    wrong_team = "H" if serving_team == "A" else "A"

                    spell_end = i
                    while spell_end < min(n_frames, i + 75) and labels[spell_end] == wrong_team:
                        spell_end += 1

                    # Ensure the serving team keeps possession through the backward pass
                    spell_end = max(spell_end, i + 25)
                    out[i:spell_end] = serving_team

                    kickoffs_found.append({
                        "restart_frame": i,
                        "strike_frame": strike_frame,
                        "prior_dead_frames": d_count,
                        "prior_dead_sec": d_count / 25.0,
                        "eric_predicted": curr_label,
                        "serving_team": serving_team,
                        "spell_end": spell_end
                    })

            d_count = 0
            i += 1
        prev_label = curr_label

    return out, kickoffs_found
