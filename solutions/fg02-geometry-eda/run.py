"""fg02-geometry-eda: how ball, player geometry and Smart Tagging relate to the labels and their
boundaries.

    python solutions/fg02-geometry-eda/run.py                 # prepare + analyze
    python solutions/fg02-geometry-eda/run.py prepare
    python solutions/fg02-geometry-eda/run.py analyze

Writes no submission. `prepare` leaves the per-frame features in
artifacts/fg02-geometry-eda/features.npz. `analyze` scores them against the public ground truth
and the hand labels and writes one CSV per table plus report.md to the same folder.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from shared.challenge import (GAMES, PUBLIC_GT, ROOT, artifacts_dir, pm, require,  # noqa: E402
                              source_frames)
from src import analysis, features  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("prepare", "analyze")


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def features_path():
    return os.path.join(artifacts_dir(NAME), "features.npz")


def prepare(config):
    """Per-frame features of the whole match, and the Smart Tagging table they came from."""
    game = GAMES[config["game"]]
    meta_path, raw_path, tags_path = require(["atd_meta", "atd_raw", "smart_tagging"],
                                             config["game"])
    n_frames = source_frames("tactical", config["game"])
    second_half = game["halves"][2][0] - game["offsets"][2]
    print("reading {} ...".format(os.path.basename(raw_path)))
    values, tags = features.build(meta_path, raw_path, tags_path, n_frames, game["pitch"],
                                  second_half)
    np.savez(features_path(), **values, **{"tag__" + k: v for k, v in tags.items()})
    print("  {} features x {} frames -> {}".format(len(values), n_frames,
                                                   os.path.relpath(features_path())))


def load_features():
    if not os.path.isfile(features_path()):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py prepare".format(
            os.path.relpath(features_path()), NAME))
    data = np.load(features_path())
    values = {k: data[k] for k in data.files if not k.startswith("tag__")}
    tags = {k[5:]: data[k] for k in data.files if k.startswith("tag__")}
    return values, tags


def datasets(config):
    sources = {"public": pm.label_codes(pm.read_label_csv(PUBLIC_GT)),
               "hand": pm.label_codes(pm.read_label_csv(os.path.join(ROOT, config["hand"])))}
    return {name: analysis.window_codes(sources[spec["labels"]], *spec["window"])
            for name, spec in config["datasets"].items()}


def markdown(table, digits=3):
    """GitHub table without extra dependencies."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else "{:.{}f}".format(value, digits)
        return str(value)
    lines = ["| " + " | ".join(map(str, table.columns)) + " |",
             "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(False)]
    return "\n".join(lines)


def analyze(config):
    # Features that are all NaN on one label (the ball ones on D, for example) give NaN means.
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
    values, tags = load_features()
    data = datasets(config)
    out = artifacts_dir(NAME)
    report = ["# fg02-geometry-eda report", ""]

    def save(table, name, title, digits=3):
        table.to_csv(os.path.join(out, name + ".csv"), index=False)
        report.extend(["## " + title, "", markdown(table, digits), ""])

    confusion, offsets, n_ref, n_other = analysis.agreement(data["public"], data["hand30"])
    report.extend(["## Hand labels against the public ground truth", "",
                   "Boundaries: {} public, {} hand.".format(n_ref, n_other), "",
                   markdown(confusion.reset_index())])
    offsets.to_csv(os.path.join(out, "agreement_offsets.csv"), index=False)
    report.extend(["", markdown(offsets, 0), ""])

    sig = features.fg01_signals()
    for name, codes in data.items():
        bounds = analysis.boundaries(codes)
        counts = bounds.cls.value_counts().to_dict()
        report.extend(["# Dataset {}".format(name), "",
                       "Graded frames {}, boundaries {}.".format(int(np.sum(codes != analysis.X)),
                                                                 counts), ""])
        uni = analysis.univariate(values, codes)
        uni["sep_HA"] = (uni.auc_HA - 0.5).abs()
        save(uni.sort_values("sep_HA", ascending=False).drop(columns="sep_HA"),
             "univariate_" + name, "{}: univariate (AUC H vs A, AUC D vs alive)".format(name))

        lock = analysis.locked(values, bounds, config["locked_half"])
        lock.to_csv(os.path.join(out, "locked_{}.csv".format(name)), index=False)
        summary = analysis.locked_summary(lock, values, codes, config["locked_level_span"])
        summary = summary.reindex(summary.jump_std.abs().sort_values(ascending=False).index)
        save(summary, "locked_summary_" + name,
             "{}: boundary-locked level change (+ = new owner / old owner for to_dead)".format(name))
        save(analysis.locked_profile(lock, config["profile_features"], config["profile_offsets"]),
             "locked_profile_" + name,
             "{}: boundary-locked mean at selected offsets (frames)".format(name), 2)

        null = analysis.null_boundaries(codes, bounds, config["null_samples"],
                                        config["null_margin"], config["seed"])
        lag = analysis.lags(values, tags, sig.CODES, pd.concat([bounds, null], ignore_index=True),
                            config["lag_reach"])
        lag.to_csv(os.path.join(out, "lags_{}.csv".format(name)), index=False)
        save(analysis.lag_summary(lag), "lag_summary_" + name,
             "{}: offset of candidate events to the labelled boundary (frames)".format(name), 2)

        change = analysis.deltas(values, config["change_seconds"])
        bauc, n_pos = analysis.boundary_auc(values, change, codes, bounds,
                                            config["boundary_radius"])
        bauc["sep"] = (bauc.change_team - 0.5).abs()
        save(bauc.sort_values("sep", ascending=False).drop(columns="sep"), "boundary_auc_" + name,
             "{}: AUC near a boundary (+-{} frames) vs other graded frames; positives {}".format(
                 name, config["boundary_radius"], n_pos))

    change = analysis.deltas(values, config["change_seconds"])
    scores, coefs = analysis.models(values, change, data, config["model_splits"], config["l2"],
                                   config["boundary_radius"])
    save(scores, "models", "Logistic regression, train on one dataset and test on the other")
    for task, table in coefs.items():
        save(table.head(20), "coefs_" + task,
             "Standardised coefficients, task {}, fitted on hand30 (top 20)".format(task))

    pca, clusters, composition = analysis.unsupervised(
        values, data["hand30"], config["pca_components"], config["kmeans_k"],
        config["unsupervised_stride"], config["seed"])
    save(pca, "pca", "PCA on hand30 graded frames")
    save(clusters, "kmeans", "k-means against the labels (NMI)")
    save(composition, "kmeans_clusters", "k-means clusters, k = {}".format(max(config["kmeans_k"])))

    path = os.path.join(out, "report.md")
    with open(path, "w") as handle:
        handle.write("\n".join(report))
    print("  -> {}".format(os.path.relpath(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all",))
    args = parser.parse_args()

    config = load_config()
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print("== {} ==".format(stage))
        {"prepare": prepare, "analyze": analyze}[stage](config)


if __name__ == "__main__":
    main()
