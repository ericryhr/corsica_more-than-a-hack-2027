"""00-baseline: the player closest to the ball has possession, smoothed by a vote.

    python solutions/00-baseline/run.py                 # prepare + predict with the windows in config.json
    python solutions/00-baseline/run.py prepare
    python solutions/00-baseline/run.py predict --windows 50 75

Writes submissions/00-baseline-w<window>.csv (w0 = no smoothing). The output of `prepare` is
kept in artifacts/00-baseline/codes.npy.
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
from src import baseline  # noqa: E402

NAME = "00-baseline"


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def codes_path():
    return os.path.join(artifacts_dir(NAME), "codes.npy")


def prepare(config):
    game = config["game"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)

    home, away = baseline.team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    codes = baseline.nearest_player_codes(
        raw_path, baseline.team_of_each_track(meta_path), n_frames, GAMES[game]["pitch"])
    seen = int(np.sum(codes != baseline.NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)".format(
        seen, n_frames, 100.0 * seen / n_frames))

    np.save(codes_path(), codes)
    print("  -> {}".format(os.path.relpath(codes_path())))


def predict(config, windows, out_dir):
    if not os.path.isfile(codes_path()):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py prepare".format(
            os.path.relpath(codes_path()), NAME))
    codes = np.load(codes_path())

    for window in windows:
        labels = baseline.smooth(codes, window)
        path = write_submission(os.path.join(out_dir, "{}-w{}.csv".format(NAME, window)), labels)
        counts = {label: int(np.sum(labels == label)) for label in "HAD"}
        print("w{:<4} H {:>6}  A {:>6}  D {:>6}  -> {}".format(
            window, counts["H"], counts["A"], counts["D"], os.path.relpath(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=("prepare", "predict", "all"))
    parser.add_argument("--windows", type=int, nargs="+",
                        help="half-widths in frames; replaces the ones in config.json")
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()

    config = load_config()
    if args.stage in ("prepare", "all"):
        prepare(config)
    if args.stage in ("predict", "all"):
        predict(config, args.windows or config["windows"], args.out)
        print("\nNext: python shared/score.py submissions/{}-w*.csv".format(NAME))


if __name__ == "__main__":
    main()
