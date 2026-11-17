"""er03-windowed-viterbi: windowed owner and dead-ball models (fg07 geometry plus er02's cues and
dead-ball features) decoded by a 3-state Viterbi, trained on the shared Metrica testbed.

    python solutions/er03-windowed-viterbi/run.py              # all stages
    python solutions/er03-windowed-viterbi/run.py prepare      # features of 6 degraded sample games + BAR-ATH
    python solutions/er03-windowed-viterbi/run.py train        # leave-one-game-out fits, costs, final fits
    python solutions/er03-windowed-viterbi/run.py predict      # BAR-ATH -> submissions

The sample games come from shared/sample.py (labels rebuilt from their events) and are degraded
with the ball blurred like the ATD's (`settings`). Every model that scores a game was fitted
without it. Costs are the best mean spellF1 over the 3 games and the settings; the
leave-one-game-out figure also chooses the costs on the other 2 games.

Writes submissions/er03-windowed-viterbi-<variant>.csv. Stages communicate through
artifacts/er03-windowed-viterbi/. Nothing here reads a BAR-ATH label; the public ground truth is
only scored afterwards by shared/score.py, as a check.
"""

import argparse
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from shared.challenge import SUBMISSIONS_DIR, artifacts_dir, write_submission  # noqa: E402
from shared.sample import GAMES_SAMPLE  # noqa: E402
from src import features, labels as label_rules, model  # noqa: E402
from src.peers import peer  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("prepare", "train", "predict")
FG07_ARTIFACTS = os.path.join(ROOT, "artifacts", "fg07-sample-tuned")


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def path(*parts):
    folder = os.path.join(artifacts_dir(NAME), *parts[:-1])
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return os.path.join(folder, parts[-1])


def keys(config):
    return [(game, setting) for setting in config["settings"] for game in GAMES_SAMPLE]


def match_of(key, config):
    data = peer("fg07", "data")
    game, setting = key
    if game == config["game"]:
        return data.bar_ath_match(game, FG07_ARTIFACTS if os.path.isdir(FG07_ARTIFACTS)
                                  else artifacts_dir(NAME))
    return data.sample_match(game, setting)


def values_of(key, config, match=None):
    """fg07 base features (fg07's own cache when it has them) plus er02's, cached here."""
    game, setting = key
    target = path("features", "{}-{}.npz".format(game, setting))
    if not os.path.isfile(target):
        match = match if match is not None else match_of(key, config)
        fg07_cache = os.path.join(FG07_ARTIFACTS, "features", "{}-{}-base.npz".format(game, setting))
        if os.path.isfile(fg07_cache):
            loaded = np.load(fg07_cache)
            values = {k: loaded[k] for k in loaded.files}
        else:
            values = features.fg07_values(match)
        values.update(features.er02_values(match, config))
        np.savez(target, **values)
    loaded = np.load(target)
    return {k: loaded[k] for k in loaded.files}


def prepare(config):
    for key in keys(config) + [(config["game"], "atd")]:
        start = time.time()
        values_of(key, config)
        print("  {} {}: {:.0f} s".format(*key, time.time() - start), flush=True)
    for game in GAMES_SAMPLE:
        np.save(path("labels", "controlled-{}.npy".format(game)),
                label_rules.controlled_for_game(game, config).astype("<U1"))


# -- train -------------------------------------------------------------------------------------

def truth(game, label_set):
    if label_set == "bench":
        return peer("fg07", "data").sample_match(game, "clean")["labels"]
    return np.load(path("labels", "controlled-{}.npy".format(game)), allow_pickle=True)


def label_stats(lab):
    starts = np.flatnonzero(np.concatenate(([True], lab[1:] != lab[:-1])))
    ends = np.concatenate((starts[1:], [len(lab)]))
    spells = [(e - s) for s, e in zip(starts, ends) if lab[s] in ("H", "A")]
    return len(spells), float(np.median(spells)) / 25, float(np.mean(np.array(spells) < 75))


def select(table):
    """(best candidate, mean per game, leave-one-game-out score per game): candidate chosen on
    the mean over every game and setting; LOGO chooses it on the other games."""
    pivot = table.pivot_table(index="candidate", columns="game", values="spellF1")
    best = pivot.mean(axis=1).idxmax()
    logo = {g: float(pivot.loc[pivot.drop(columns=g).mean(axis=1).idxmax(), g]) for g in pivot}
    return best, pivot.loc[best].to_dict(), logo


def train(config):
    matches = {key: match_of(key, config) for key in keys(config)}
    values = {key: values_of(key, config, matches[key]) for key in keys(config)}
    bench_truth = {g: truth(g, "bench") for g in GAMES_SAMPLE}
    for label_set in config["label_sets"]:
        stats = [label_stats(truth(g, label_set)) for g in GAMES_SAMPLE]
        print("labels {:<10}: possessions {}, median {:.1f} s, under 3 s {:.0%}".format(
            label_set, sum(s[0] for s in stats), np.mean([s[1] for s in stats]),
            np.mean([s[2] for s in stats])))

    report, models, chosen = {}, {}, {}
    for feats, label_set in itertools.product(config["feature_sets"], config["label_sets"]):
        variant = "{}-{}".format(feats, label_set)
        s = model.spec(config, feats)
        train_labels = {g: truth(g, label_set) for g in GAMES_SAMPLE}
        start = time.time()
        models[variant] = {}
        prob = {}
        for held in GAMES_SAMPLE:
            pool = [(values[k], train_labels[k[0]]) for k in keys(config) if k[0] != held]
            models[variant][held] = model.fit(pool, s)
            for key in keys(config):
                if key[0] == held:
                    prob[key] = model.probs(models[variant][held], values[key], s)
        models[variant]["final"] = model.fit([(values[k], train_labels[k[0]]) for k in keys(config)], s)

        # H vs A frame accuracy on the bench truth's live frames, leave one game out
        acc = []
        for key in keys(config):
            gt = bench_truth[key[0]]
            live = np.isin(gt, ["H", "A"])
            acc.append(np.mean((prob[key][0][live] > 0.5) == (gt[live] == "H")))

        rows = []
        for team_cost, dead_cost in itertools.product(config["costs"]["team"], config["costs"]["dead"]):
            name = json.dumps({"team": team_cost, "dead": dead_cost}, sort_keys=True)
            for key, match in matches.items():
                lab = model.viterbi(match, *prob[key], team_cost, dead_cost)
                row = {"candidate": name, "game": key[0], "setting": key[1]}
                row["spellF1"], row["macroF1"] = score(bench_truth[key[0]], lab)
                if label_set != "bench":
                    row["spellF1_own"], row["macroF1_own"] = score(train_labels[key[0]], lab)
                rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(path("tune-{}.csv".format(variant)), index=False)
        best, per_game, logo = select(table)
        best_rows = table[table.candidate == best]
        chosen[variant] = json.loads(best)
        report[variant] = {"costs": chosen[variant], "spellF1": float(np.mean(list(per_game.values()))),
                           "spellF1_logo": float(np.mean(list(logo.values()))),
                           "macroF1": float(best_rows.macroF1.mean()), "owner_acc": float(np.mean(acc)),
                           "per_game_logo": logo}
        if "spellF1_own" in best_rows:
            report[variant]["spellF1_own_truth"] = float(best_rows.spellF1_own.mean())
        print("{:<16} owner acc {:.3f}  spellF1 {:.3f} (LOGO {:.3f})  macroF1 {:.3f}  costs {}{}  [{:.0f} s]".format(
            variant, report[variant]["owner_acc"], report[variant]["spellF1"],
            report[variant]["spellF1_logo"], report[variant]["macroF1"], best,
            "  own truth {:.3f}".format(report[variant]["spellF1_own_truth"])
            if "spellF1_own_truth" in report[variant] else "", time.time() - start), flush=True)

        if variant == "full-bench":   # er02's decoder on the same probabilities, as a control
            mc = config["merge_control"]
            for ms in mc["min_spells"]:
                sc = [score(bench_truth[k[0]], model.merge_decode(
                    matches[k], *prob[k], mc["threshold"], mc["min_dead"], ms)) for k in keys(config)]
                report["full-bench-merge-s{}".format(ms)] = {
                    "spellF1": float(np.mean([x[0] for x in sc])), "macroF1": float(np.mean([x[1] for x in sc]))}
                print("  control: er02 merge decoder s{}: spellF1 {:.3f} macroF1 {:.3f}".format(
                    ms, *report["full-bench-merge-s{}".format(ms)].values()))

    best_variant = max((v for v in chosen), key=lambda v: report[v]["spellF1_logo"])
    report["chosen"] = best_variant
    with open(path("models.pkl"), "wb") as handle:
        pickle.dump({"models": models, "costs": chosen, "chosen": best_variant}, handle)
    with open(path("report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print("chosen variant (best LOGO spellF1): {}".format(best_variant))


def score(gt_labels, labels):
    methods = peer("fg07", "methods")
    return methods.score({"labels": gt_labels}, labels)


# -- predict -----------------------------------------------------------------------------------

def predict(config, out_dir):
    peer("fg03", "model")              # the pickled classes live in fg03's package
    with open(path("models.pkl"), "rb") as handle:
        saved = pickle.load(handle)
    key = (config["game"], "atd")
    match = match_of(key, config)
    values = values_of(key, config, match)
    graded = peer("fg07", "data").graded(match)
    variants = [saved["chosen"]] + [v for v in ("fg07-bench", "full-bench") if v != saved["chosen"]]
    for variant in variants:
        feats = variant.split("-")[0]
        p_home, p_dead = model.probs(saved["models"][variant]["final"], values, model.spec(config, feats))
        costs = saved["costs"][variant]
        lab = model.viterbi(match, p_home, p_dead, costs["team"], costs["dead"])
        out = write_submission(os.path.join(out_dir, "{}-{}.csv".format(NAME, variant)), lab)
        g = lab[graded]
        n_spells = peer("er02", "decode").count_spells(lab, graded)
        print("{:<16} H {:.1%} A {:.1%} D {:.1%}  possessions {}  -> {}".format(
            variant, np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D"), n_spells,
            os.path.relpath(out)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all",))
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()
    config = load_config()
    for stage in (STAGES if args.stage == "all" else (args.stage,)):
        print("== {} ==".format(stage), flush=True)
        if stage == "predict":
            predict(config, args.out)
        else:
            {"prepare": prepare, "train": train}[stage](config)
    print("\nNext: python shared/score.py submissions/{}-*.csv".format(NAME))


if __name__ == "__main__":
    main()
