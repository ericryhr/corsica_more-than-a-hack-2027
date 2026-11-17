"""01-ball-drift: impute the missing ball by moving its last known position with the players' drift.

    python solutions/01-ball-drift/run.py                 # prepare + predict with the values in config.json
    python solutions/01-ball-drift/run.py prepare
    python solutions/01-ball-drift/run.py predict --gains 1 2 --windows 50 125

Writes submissions/01-ball-drift-g<gain>[-d<frames>][-m<frames>]-w<window>.csv. g0 = ball frozen
at its last position, d = frames after which the drift stops, m = frames after which the gap is
labelled D, w0 = no smoothing. The output of `prepare` is kept in artifacts/01-ball-drift/tracking.npz.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from shared.challenge import (GAMES, SUBMISSIONS_DIR, artifacts_dir, require,  # noqa: E402
                              source_frames, write_submission)
from src import drift  # noqa: E402

NAME = os.path.basename(HERE)


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def tracking_path():
    return os.path.join(artifacts_dir(NAME), "tracking.npz")


def prepare(config):
    game = config["game"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)

    team = drift.team_of_each_track(meta_path)
    print("reading {} ...".format(os.path.basename(raw_path)))
    players, ball = drift.read_tracking(raw_path, n_frames, len(team), GAMES[game]["pitch"])
    seen = int(np.sum(~np.isnan(ball[:, 0])))
    print("  ball located on {} of {} frames ({:.1f}%)".format(
        seen, n_frames, 100.0 * seen / n_frames))

    np.savez(tracking_path(), players=players, ball=ball, team=team)
    print("  -> {}".format(os.path.relpath(tracking_path())))


def variant(gain, drift_frames, max_gap, window):
    tag = "g{:g}".format(gain)
    if drift_frames is not None:
        tag += "-d{}".format(drift_frames)
    if max_gap is not None:
        tag += "-m{}".format(max_gap)
    return "{}-{}-w{}".format(NAME, tag, window)


def predict(config, gains, windows, out_dir):
    if not os.path.isfile(tracking_path()):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py prepare".format(
            os.path.relpath(tracking_path()), NAME))
    data = np.load(tracking_path())
    players, ball, team = data["players"], data["ball"], data["team"]
    pitch = GAMES[config["game"]]["pitch"]
    step = drift.mean_step(players, config["min_players"])

    for gain in gains:
        # Without drift the stop point changes nothing, so one value is enough.
        drift_options = config["drift_frames"] if gain else [None]
        for drift_frames in drift_options:
            for max_gap in config["max_gaps"]:
                imputed = drift.impute_ball(ball, step, pitch, gain, drift_frames, max_gap)
                codes = drift.nearest_codes(players, imputed, team)
                for window in windows:
                    labels = drift.smooth(codes, window)
                    path = write_submission(os.path.join(
                        out_dir, variant(gain, drift_frames, max_gap, window) + ".csv"), labels)
                    counts = {label: int(np.sum(labels == label)) for label in "HAD"}
                    print("{:<28} H {:>6}  A {:>6}  D {:>6}".format(
                        os.path.basename(path)[len(NAME) + 1:-4],
                        counts["H"], counts["A"], counts["D"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=("prepare", "predict", "all"))
    parser.add_argument("--gains", type=float, nargs="+",
                        help="drift multipliers; replaces the ones in config.json")
    parser.add_argument("--windows", type=int, nargs="+",
                        help="half-widths in frames; replaces the ones in config.json")
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()

    config = load_config()
    if args.stage in ("prepare", "all"):
        prepare(config)
    if args.stage in ("predict", "all"):
        predict(config, args.gains or config["gains"], args.windows or config["windows"], args.out)
        print("\nNext: python shared/score.py submissions/{}-*.csv".format(NAME))


if __name__ == "__main__":
    main()
