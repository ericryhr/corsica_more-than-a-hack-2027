"""fg01-smart-tagging: fuse the team closest to the ball with the Smart Tagging phases.

    python solutions/fg01-smart-tagging/run.py                 # prepare + predict
    python solutions/fg01-smart-tagging/run.py prepare
    python solutions/fg01-smart-tagging/run.py predict
    python solutions/fg01-smart-tagging/run.py search          # not part of "all"

Writes submissions/fg01-smart-tagging-<variant>.csv, one per entry of "variants" in config.json.
The output of `prepare` is kept in artifacts/fg01-smart-tagging/signals.npz.

`search` runs every combination of each grid in config.json "search" and scores it in memory on
the public ground truth and on the hand labels of config.json "hand" (see shared/score.py for the
same scoring). The table goes to artifacts/fg01-smart-tagging/search.csv. Nothing is fitted: the
rules are deterministic, and the search only compares them.
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from shared.challenge import (GAMES, PUBLIC_GT, ROOT, SUBMISSIONS_DIR, artifacts_dir,  # noqa: E402
                              pm, require, source_frames, write_submission)
from src import rules, signals  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("prepare", "predict")


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def signals_path():
    return os.path.join(artifacts_dir(NAME), "signals.npz")


def load_signals():
    if not os.path.isfile(signals_path()):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py prepare".format(
            os.path.relpath(signals_path()), NAME))
    data = np.load(signals_path())
    tags = {key[4:]: data[key] for key in data.files if key.startswith("tag_")}
    return data["codes"], data["distance"], data["ball"], tags


def prepare(config):
    """Nearest team to the ball per frame and the Smart Tagging intervals, in video frames."""
    game = config["game"]
    meta_path, raw_path, tags_path = require(["atd_meta", "atd_raw", "smart_tagging"], game)
    n_frames = source_frames("tactical", game)

    team = signals.team_of_each_track(meta_path)
    print("reading {} ...".format(os.path.basename(raw_path)))
    codes, distance, ball = signals.nearest_team(raw_path, n_frames, team, GAMES[game]["pitch"])
    print("  ball and a player located on {} of {} frames".format(
        int(np.sum(codes != signals.NOBODY)), n_frames))
    tags = signals.read_tags(tags_path, GAMES[game]["fps"])
    print("  {} Smart Tagging intervals".format(len(tags["start"])))

    np.savez(signals_path(), codes=codes, distance=distance, ball=ball,
             **{"tag_" + key: value for key, value in tags.items()})
    print("  -> {}".format(os.path.relpath(signals_path())))


def predict(config, out_dir):
    """One submission per entry of config["variants"]: {"name": ..., "rule": ..., params}."""
    codes, distance, ball, tags = load_signals()
    pitch = GAMES[config["game"]]["pitch"]
    for variant in config["variants"]:
        params = {k: v for k, v in variant.items() if k != "name"}
        labels = rules.apply(codes, distance, ball, tags, pitch, **params)
        path = write_submission(os.path.join(
            out_dir, "{}-{}.csv".format(NAME, variant["name"])), labels)
        counts = {label: int(np.sum(labels == label)) for label in "HAD"}
        print("{:<28} H {:>6}  A {:>6}  D {:>6}".format(
            os.path.basename(path)[len(NAME) + 1:-4], counts["H"], counts["A"], counts["D"]))


def ground_truths(config):
    """{name: gt codes} for the public ground truth and each window of the hand labels."""
    n_frames = source_frames("tactical", config["game"])
    out = {"public": pm.label_codes(pm.read_label_csv(PUBLIC_GT))}
    hand = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, config["hand"]["gt"])))
    for name, (start, end) in config["hand"]["windows"].items():
        codes = np.full(n_frames, pm.X, dtype=hand.dtype)
        codes[start:end + 1] = hand[start:end + 1]
        out[name] = codes
    return out


def combinations(grid):
    keys = sorted(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def search(config):
    codes, distance, ball, tags = load_signals()
    pitch = GAMES[config["game"]]["pitch"]
    gts = ground_truths(config)
    rows = []
    for grid in config["search"]:
        for params in combinations(grid):
            pred = pm.label_codes(rules.apply(codes, distance, ball, tags, pitch, **params))
            row = {"params": json.dumps(params, sort_keys=True)}
            for name, gt in gts.items():
                result = pm.evaluate(gt, pred)
                row[name + "_spell_f1"] = round(result["spell_f1_mean"], 4)
                row[name + "_macro_f1"] = round(result["macro_f1"], 4)
                row[name + "_spells"] = "{}/{}".format(result["n_pred_spells"], result["n_gt_spells"])
            rows.append(row)
    path = os.path.join(artifacts_dir(NAME), "search.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("{} combinations -> {}".format(len(rows), os.path.relpath(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all", "search"))
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()

    config = load_config()
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print("== {} ==".format(stage))
        if stage == "predict":
            predict(config, args.out)
        elif stage == "search":
            search(config)
        else:
            prepare(config)
    print("\nNext: python shared/score.py submissions/{}-*.csv".format(NAME))


if __name__ == "__main__":
    main()
