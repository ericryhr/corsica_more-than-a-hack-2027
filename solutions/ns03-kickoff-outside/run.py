"""ns03-kickoff-outside: Center-serving kickoff rule and outside-pitch boundary filter.

    python solutions/ns03-kickoff-outside/run.py                 # prepare + predict using config.json
    python solutions/ns03-kickoff-outside/run.py prepare
    python solutions/ns03-kickoff-outside/run.py predict

Writes submissions/ns03-kickoff-outside-<variant>.csv.
Stages communicate through artifacts/ns03-kickoff-outside/.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from shared.challenge import (GAMES, SUBMISSIONS_DIR, artifacts_dir, require,  # noqa: E402
                              source_frames, write_submission)
from src import serving  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("prepare", "predict")


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def tracking_path():
    return os.path.join(artifacts_dir(NAME), "tracking.npz")


def read_base_labels(path, n_frames):
    """Read a frame,label CSV into a numpy array of strings."""
    if not os.path.isfile(path):
        # Check relative to repo root
        root = os.path.dirname(os.path.dirname(HERE))
        candidate = os.path.join(root, path)
        if os.path.isfile(candidate):
            path = candidate
        else:
            raise FileNotFoundError("Base submission not found: {}".format(path))

    labels = np.full(n_frames, "X", dtype="<U1")
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                idx = int(row[0])
                if 0 <= idx < n_frames:
                    labels[idx] = row[1]
    return labels


def prepare(config):
    game = config["game"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]

    home, away = serving.team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    team_slots = serving.team_of_each_track(meta_path)
    ball, min_dh_c, min_da_c, min_dh_b, min_da_b = serving.read_tracking(
        raw_path, team_slots, n_frames, pitch)

    seen = int(np.sum(~np.isnan(ball[:, 0])))
    print("  ball located on {} of {} frames ({:.1f}%)".format(
        seen, n_frames, 100.0 * seen / n_frames))

    np.savez(
        tracking_path(),
        ball=ball,
        min_dh_center=min_dh_c,
        min_da_center=min_da_c,
        min_dh_ball=min_dh_b,
        min_da_ball=min_da_b,
    )
    print("  -> {}".format(os.path.relpath(tracking_path())))


def predict(config, base_path, out_dir, min_dead_frames=None):
    if not os.path.isfile(tracking_path()):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py prepare".format(
            os.path.relpath(tracking_path()), NAME))

    data = np.load(tracking_path())
    ball = data["ball"]
    min_dh_c = data["min_dh_center"]
    min_da_c = data["min_da_center"]
    min_dh_b = data["min_dh_ball"]
    min_da_b = data["min_da_ball"]

    n_frames = len(ball)
    pitch = GAMES[config["game"]]["pitch"]

    base_file = base_path or config.get(
        "base_submission", "submissions/er03-windowed-viterbi-dead02-controlled.csv")
    print("Loading base submission: {}".format(base_file))
    base_labels = read_base_labels(base_file, n_frames)

    center_radius = config.get("center_radius_m", 1.5)
    if min_dead_frames is None:
        min_dead_frames = config.get("min_dead_frames", 375)
    kickoff_hold_frames = config.get("kickoff_hold_frames", 35)
    opening_window_frames = config.get("opening_window_frames", 11300)

    print("Kickoff parameters: min_dead_frames={} ({:.1f}s), center_radius={:.1f}m, hold_frames={}, opening_window={}".format(
        min_dead_frames, min_dead_frames / 25.0, center_radius, kickoff_hold_frames, opening_window_frames))

    margins = config.get("outside_margins", [0.0, 0.5])
    min_frames_list = config.get("outside_min_frames", [12, 25])

    # Apply kickoff serving rule
    labels_kickoff, kickoffs = serving.detect_and_apply_kickoffs(
        base_labels, ball, min_dh_c, min_da_c, min_dh_b, min_da_b, pitch,
        center_radius=center_radius, min_dead_frames=min_dead_frames,
        kickoff_hold_frames=kickoff_hold_frames,
        opening_window_frames=opening_window_frames)

    print("\nDetected {} kickoff restart events:".format(len(kickoffs)))
    for k in kickoffs:
        print("  frame {} (strike {}..{}): prior dead {:.1f}s, er03={}, serving team {}".format(
            k["restart_frame"], k["strike_frame"], k["spell_end"],
            k["prior_dead_sec"], k["eric_predicted"], k["serving_team"]))

    # Submission: Combined (Base + Kickoff serving + Outside pitch m0-f25)
    outside_margin = config.get("outside_margin", 0.0)
    outside_min_frames = config.get("outside_min_frames", 25)

    sub_name = "{}-combined-m0-f25".format(NAME)
    labels_combined = serving.apply_outside_pitch(
        labels_kickoff, ball, pitch, margin=outside_margin, min_frames=outside_min_frames)
    path = write_submission(os.path.join(out_dir, sub_name + ".csv"), labels_combined)
    counts = {label: int(np.sum(labels_combined == label)) for label in "HAD"}
    print("{:<42} H {:>6}  A {:>6}  D {:>6} -> {}".format(
        sub_name[len(NAME) + 1:], counts["H"], counts["A"], counts["D"], os.path.relpath(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all",))
    parser.add_argument("--base", help="path to base submission CSV to improve")
    parser.add_argument("--min-dead-sec", type=float, help="minimum prior dead seconds before kickoff (e.g. 15)")
    parser.add_argument("--min-dead-frames", type=int, help="minimum prior dead frames before kickoff (e.g. 375)")
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()

    config = load_config()

    mdf = args.min_dead_frames
    if mdf is None and args.min_dead_sec is not None:
        mdf = int(args.min_dead_sec * 25)

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print("== {} ==".format(stage))
        if stage == "prepare":
            prepare(config)
        elif stage == "predict":
            predict(config, args.base, args.out, min_dead_frames=mdf)

    if "predict" in stages:
        print("\nNext: score the generated submissions:")
        print("  python shared/score.py submissions/{}-*.csv --gt labels/fg_30_11175-56199.csv --window 11185 56199".format(NAME))
        print("  python shared/score.py submissions/{}-*.csv --gt public".format(NAME))


if __name__ == "__main__":
    main()
