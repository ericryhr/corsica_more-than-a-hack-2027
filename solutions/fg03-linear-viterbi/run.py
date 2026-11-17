"""fg03-linear-viterbi: linear evidence for owner and dead ball, decoded with a 3-state Viterbi.

    python solutions/fg03-linear-viterbi/run.py                 # prepare + train + predict
    python solutions/fg03-linear-viterbi/run.py prepare
    python solutions/fg03-linear-viterbi/run.py train
    python solutions/fg03-linear-viterbi/run.py predict
    python solutions/fg03-linear-viterbi/run.py cv              # not part of "all"
    python solutions/fg03-linear-viterbi/run.py select          # not part of "all", ~10 min

Writes two submissions per entry of "variants" in config.json:

  submissions/fg03-linear-viterbi-<variant>.csv      models fitted on every labelled frame
  submissions/fg03-linear-viterbi-<variant>-oof.csv  the same, except that inside the labelled
                                                      window each block is predicted by models
                                                      fitted without it ("folds" in config.json)

Score the `-oof` file to compare with other solutions: the plain one has seen its own labels.
`cv` scores every switch-cost combination of config.json "cv_costs" on the out-of-fold
predictions and writes artifacts/fg03-linear-viterbi/cv.csv. `select` runs the forward selection
that produced the `compact` lists and a nested cross-validation of it (select.csv, nested.csv).
The fits are deterministic (Newton steps from zero), so there is no seed.
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from shared.challenge import (GAMES, PUBLIC_GT, ROOT, SUBMISSIONS_DIR, artifacts_dir, pm,  # noqa: E402
                              require, source_frames, write_submission)
from src import features, model, selection  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("prepare", "train", "predict")
LETTERS = np.array(list("HAD"))


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def features_path():
    return os.path.join(artifacts_dir(NAME), "features.npz")


def models_path(variant):
    return os.path.join(artifacts_dir(NAME), "model-{}.json".format(variant))


def prepare(config):
    """Base features of the whole match (see src/features.py)."""
    game = GAMES[config["game"]]
    meta_path, raw_path, tags_path = require(["atd_meta", "atd_raw", "smart_tagging"],
                                             config["game"])
    n_frames = source_frames("tactical", config["game"])
    second_half = game["halves"][2][0] - game["offsets"][2]
    print("reading {} ...".format(os.path.basename(raw_path)))
    values = features.build(meta_path, raw_path, tags_path, n_frames, game["pitch"], second_half)
    np.savez(features_path(), **values)
    print("  {} base features x {} frames -> {}".format(len(values), n_frames,
                                                        os.path.relpath(features_path())))


def load_features():
    if not os.path.isfile(features_path()):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py prepare".format(
            os.path.relpath(features_path()), NAME))
    data = np.load(features_path())
    return {key: data[key] for key in data.files}


def labels(config, n_frames):
    """Label codes of config["labels"], X outside its window."""
    hand = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, config["labels"]["gt"])))
    start, end = config["labels"]["window"]
    out = np.full(n_frames, pm.X, dtype=hand.dtype)
    out[start:end + 1] = hand[start:end + 1]
    return out


def feature_names(spec, values):
    """A variant's feature list: "all" or an explicit list of base features."""
    return sorted(values) if spec == "all" else spec


def matrices(variant, values):
    """{task: (matrix, columns)} for the owner and dead models of a variant."""
    return {task: features.design(values, feature_names(variant[task], values), variant["windows"])
            for task in ("owner", "dead")}


def targets(codes):
    """{task: (frames to fit on, 0/1 target)}."""
    alive = (codes == pm.H) | (codes == pm.A)
    return {"owner": (alive, codes == pm.H), "dead": (codes != pm.X, codes == pm.D)}


def fit(variant, mats, codes, exclude=None):
    """{task: Logistic} fitted on the labelled frames outside `exclude` (a boolean mask)."""
    out = {}
    for task, (rows, target) in targets(codes).items():
        rows = rows if exclude is None else rows & ~exclude
        out[task] = model.Logistic(variant["l2"]).fit(mats[task][0][rows], target[rows])
    return out


def probabilities(models, mats, rows=slice(None)):
    return models["owner"].predict(mats["owner"][0][rows]), models["dead"].predict(mats["dead"][0][rows])


def out_of_fold(variant, mats, codes, folds):
    """(p_home, p_dead) for every frame; inside each fold, from models fitted without it."""
    models = fit(variant, mats, codes)
    p_home, p_dead = probabilities(models, mats)
    for start, end in folds:
        held = np.zeros(len(codes), dtype=bool)
        held[start:end + 1] = True
        fold_models = fit(variant, mats, codes, exclude=held)
        p_home[start:end + 1], p_dead[start:end + 1] = probabilities(
            fold_models, mats, slice(start, end + 1))
    return p_home, p_dead


def decode(p_home, p_dead, costs, halves):
    """H/A/D letters: a Viterbi path inside each half, D outside the halves (never graded)."""
    logp = model.log_probs(p_home, p_dead)
    path = np.full(len(p_home), model.D, dtype=np.int8)
    for start, end in halves:
        path[start:end + 1] = model.viterbi(logp[start:end + 1], costs["team"], costs["dead"])
    return LETTERS[path]


def halves(config):
    game = GAMES[config["game"]]
    return [(game["halves"][h][0] - game["offsets"][h], game["halves"][h][1] - game["offsets"][h])
            for h in (1, 2)]


def train(config):
    """Fit each variant on every labelled frame; save the weights and a coefficient table."""
    values = load_features()
    codes = labels(config, len(values["ball_seen"]))
    for variant in config["variants"]:
        mats = matrices(variant, values)
        models = fit(variant, mats, codes)
        with open(models_path(variant["name"]), "w") as handle:
            json.dump({task: models[task].to_dict(mats[task][1]) for task in models}, handle)
        table = coefficient_table(models, mats)
        table.to_csv(os.path.join(artifacts_dir(NAME), "coefs-{}.csv".format(variant["name"])),
                     index=False)
        print("{:<10} owner {} columns, dead {} columns -> {}".format(
            variant["name"], len(mats["owner"][1]), len(mats["dead"][1]),
            os.path.relpath(models_path(variant["name"]))))


def coefficient_table(models, mats):
    """One row per column: standardised weight in each model. `base` groups the windows of a base
    feature, and `base_l1` is the sum of their absolute weights, to rank base features."""
    rows = []
    for task in ("owner", "dead"):
        for column, weight in zip(mats[task][1], models[task].weights[1:]):
            rows.append({"task": task, "column": column, "base": column.split("@")[0],
                         "weight": round(float(weight), 4)})
    table = pd.DataFrame(rows)
    table["base_l1"] = table.groupby(["task", "base"]).weight.transform(lambda w: w.abs().sum())
    return table.sort_values(["task", "base_l1", "base"], ascending=[True, False, True])


def load_models(variant):
    path = models_path(variant)
    if not os.path.isfile(path):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py train".format(
            os.path.relpath(path), NAME))
    with open(path) as handle:
        data = json.load(handle)
    return {task: model.Logistic.from_dict(data[task]) for task in data}


def predict(config, out_dir):
    values = load_features()
    codes = labels(config, len(values["ball_seen"]))
    for variant in config["variants"]:
        mats = matrices(variant, values)
        p_home, p_dead = probabilities(load_models(variant["name"]), mats)
        oof_home, oof_dead = out_of_fold(variant, mats, codes, config["folds"])
        for suffix, (ph, pd_) in (("", (p_home, p_dead)), ("-oof", (oof_home, oof_dead))):
            letters = decode(ph, pd_, variant["costs"], halves(config))
            path = write_submission(os.path.join(
                out_dir, "{}-{}{}.csv".format(NAME, variant["name"], suffix)), letters)
            counts = {label: int(np.sum(letters == label)) for label in "HAD"}
            print("{:<20} H {:>6}  A {:>6}  D {:>6}".format(
                os.path.basename(path)[len(NAME) + 1:-4], counts["H"], counts["A"], counts["D"]))


def cv(config):
    """Every switch-cost combination of config["cv_costs"] on the out-of-fold predictions."""
    values = load_features()
    codes = labels(config, len(values["ball_seen"]))
    gts = {"public": pm.label_codes(pm.read_label_csv(PUBLIC_GT))}
    for name, (start, end) in config["cv_windows"].items():
        gts[name] = np.full(len(codes), pm.X, dtype=codes.dtype)
        gts[name][start:end + 1] = codes[start:end + 1]
    grid = config["cv_costs"]
    first, last = config["labels"]["window"]
    labelled = [(a, b) for a, b in halves(config) if a <= last and first <= b]
    rows = []
    for variant in config["variants"]:
        mats = matrices(variant, values)
        p_home, p_dead = out_of_fold(variant, mats, codes, config["folds"])
        for team, dead in itertools.product(grid["team"], grid["dead"]):
            costs = {"team": team, "dead": dead}
            pred = pm.label_codes(decode(p_home, p_dead, costs, labelled))
            row = {"variant": variant["name"], **costs}
            for name, gt in gts.items():
                result = pm.evaluate(gt, pred)
                row[name + "_spell_f1"] = round(result["spell_f1_mean"], 4)
                row[name + "_macro_f1"] = round(result["macro_f1"], 4)
                row[name + "_spells"] = "{}/{}".format(result["n_pred_spells"], result["n_gt_spells"])
            rows.append(row)
        best = max((r for r in rows if r["variant"] == variant["name"]),
                   key=lambda r: r["hand30_spell_f1"])
        print("{:<10} best on hand30: team {} dead {}  spellF1 hand30 {} public {}".format(
            variant["name"], best["team"], best["dead"], best["hand30_spell_f1"],
            best["public_spell_f1"]))
    path = os.path.join(artifacts_dir(NAME), "cv.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("{} rows -> {}".format(len(rows), os.path.relpath(path)))


def select(config):
    """Forward selection on every fold (the `compact` lists), then the same selection repeated
    inside each fold to score the procedure without selection bias (nested cross-validation)."""
    values = load_features()
    codes = labels(config, len(values["ball_seen"]))
    spec = config["select"]
    blocks = {name: features.design(values, [name], spec["windows"])[0] for name in values}
    folds = config["folds"]
    rows = []

    def run_forward(scope, inner):
        chosen = {}
        for task, (mask, target) in targets(codes).items():
            path = selection.forward(blocks, mask, target.astype(np.float64), inner,
                                     spec["steps"], spec["stride"], spec["l2"])
            rows.extend({"scope": scope, "task": task, "step": step + 1, "feature": name,
                         "log_loss": round(loss, 5)} for step, (name, loss) in enumerate(path))
            chosen[task] = [name for name, _ in path]
            print("{:<8} {:<5} {}".format(scope, task, ", ".join(chosen[task])), flush=True)
        return chosen

    run_forward("all", folds)
    p = {"owner": np.full(len(codes), 0.5), "dead": np.full(len(codes), 0.5)}
    for k, (start, end) in enumerate(folds):
        inner = [fold for j, fold in enumerate(folds) if j != k]
        chosen = run_forward("outer{}".format(k), inner)
        train_mask = np.any(selection.fold_masks(len(codes), inner), axis=0)
        for task, (mask, target) in targets(codes).items():
            x = np.hstack([blocks[name] for name in chosen[task]])
            fitted = model.Logistic(spec["l2"]).fit(x[mask & train_mask], target[mask & train_mask])
            p[task][start:end + 1] = fitted.predict(x[start:end + 1])
    out = artifacts_dir(NAME)
    pd.DataFrame(rows).to_csv(os.path.join(out, "select.csv"), index=False)

    gts = {"public": pm.label_codes(pm.read_label_csv(PUBLIC_GT)), "hand30": codes}
    first, last = config["labels"]["window"]
    labelled = [(a, b) for a, b in halves(config) if a <= last and first <= b]
    scores = []
    for team, dead in itertools.product(config["cv_costs"]["team"], config["cv_costs"]["dead"]):
        pred = pm.label_codes(decode(p["owner"], p["dead"], {"team": team, "dead": dead}, labelled))
        row = {"team": team, "dead": dead}
        for name, gt in gts.items():
            result = pm.evaluate(gt, pred)
            row[name + "_spell_f1"] = round(result["spell_f1_mean"], 4)
            row[name + "_macro_f1"] = round(result["macro_f1"], 4)
            row[name + "_spells"] = "{}/{}".format(result["n_pred_spells"], result["n_gt_spells"])
        scores.append(row)
    scores = pd.DataFrame(scores)
    scores.to_csv(os.path.join(out, "nested.csv"), index=False)
    print("nested hand30 spellF1: median {:.3f}, range {:.3f} to {:.3f} over {} cost pairs".format(
        scores.hand30_spell_f1.median(), scores.hand30_spell_f1.min(),
        scores.hand30_spell_f1.max(), len(scores)))
    print("-> {} and {}".format(os.path.relpath(os.path.join(out, "select.csv")),
                                os.path.relpath(os.path.join(out, "nested.csv"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all", "cv", "select"))
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()

    config = load_config()
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print("== {} ==".format(stage))
        if stage == "predict":
            predict(config, args.out)
        else:
            {"prepare": prepare, "train": train, "cv": cv, "select": select}[stage](config)
    print("\nNext: python shared/score.py submissions/{}-*-oof.csv".format(NAME))


if __name__ == "__main__":
    main()
