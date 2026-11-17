"""fg07-sample-tuned: fg01, fg03, fg04, fg05 (and 00, 01) with every parameter tuned and every
model trained on Metrica's public sample games, never on BAR-ATH labels.

    python solutions/fg07-sample-tuned/run.py            # all stages
    python solutions/fg07-sample-tuned/run.py prepare    # features of the sample games and BAR-ATH
    python solutions/fg07-sample-tuned/run.py tune       # grids of the rule methods on the sample
    python solutions/fg07-sample-tuned/run.py train      # leave-one-game-out fits, costs, final fits
    python solutions/fg07-sample-tuned/run.py predict    # BAR-ATH submissions
    python solutions/fg07-sample-tuned/run.py report     # the sample results table

The sample games come from shared/sample.py (labels rebuilt from the events) and are degraded with
the settings in config.json, as shared/sample_bench.py does. Every choice (a grid point, a cost, a
fusion weight) is the best mean spellF1 over the 3 games and the settings. The estimate reported
next to it is leave-one-game-out: the choice made on the other 2 games, scored on the third, and
for the trained models a model that never saw the third game either.

Writes submissions/fg07-sample-tuned-<method>.csv. Stages communicate through
artifacts/fg07-sample-tuned/. The public ground truth of BAR-ATH is only used afterwards, by
shared/score.py, to check that the pipeline transfers; nothing here reads it.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from shared.challenge import SUBMISSIONS_DIR, artifacts_dir, require, write_submission  # noqa: E402
from shared.sample import GAMES_SAMPLE  # noqa: E402
from src import data, features, methods  # noqa: E402
from src.peers import peer  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("prepare", "tune", "train", "predict", "report")


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def path(*parts):
    folder = os.path.join(artifacts_dir(NAME), *parts[:-1])
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return os.path.join(folder, parts[-1])


def keys(config):
    """(game, setting) of every degraded sample match."""
    return [(game, setting) for setting in config["settings"] for game in GAMES_SAMPLE]


# --- prepare --------------------------------------------------------------------------------------

def feature_file(key, kind):
    game, setting = key
    return path("features", "{}-{}-{}.npz".format(game, setting, kind))


def cached(key, kind, build):
    target = feature_file(key, kind)
    if not os.path.isfile(target):
        np.savez(target, **build())
    loaded = np.load(target)
    return {name: loaded[name] for name in loaded.files}


def match_of(key, config):
    game, setting = key
    if game == config["game"]:
        return data.bar_ath_match(game, artifacts_dir(NAME))
    return data.sample_match(game, setting)


def values_of(key, config):
    """Base and network features of one match, from the cache."""
    match = match_of(key, config)
    values = cached(key, "base", lambda: features.base(match))
    values.update(cached(key, "network", lambda: features.network(match, config["network"])))
    return values


def prepare(config):
    for key in keys(config) + [(config["game"], "atd")]:
        start = time.time()
        match = match_of(key, config)
        if key[0] != config["game"]:
            print("  game {} {}: home attacks +x in halves {}".format(*key, data.attacks_right(match)))
        values_of(key, config)
        print("  {} {}: {} frames, {:.0f} s".format(*key, len(match["ball"]), time.time() - start), flush=True)


# --- selection ------------------------------------------------------------------------------------

def select(table):
    """(best candidate, its mean per game, leave-one-game-out score per game). `table` has one
    row per candidate, game and setting; settings are averaged first."""
    pivot = table.pivot_table(index="candidate", columns="game", values="spellF1", aggfunc="mean")
    best = pivot.mean(axis=1).idxmax()
    logo = {game: float(pivot.loc[pivot.drop(columns=game).mean(axis=1).idxmax(), game])
            for game in pivot.columns}
    return best, pivot.loc[best].to_dict(), logo


def grid(spec):
    names = sorted(spec)
    for combo in itertools.product(*(spec[n] for n in names)):
        yield dict(zip(names, combo))


def rule_candidates(config, method):
    seen = set()
    for params in grid(config["grids"][method]):
        if method == "rules":
            if params["ball_radius"] is None:
                params["far_weight"] = 1.0       # the weight only applies beyond the radius
            params = dict(config["rules_fixed"], **params)
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield key, params


def label_rule(method, match, params):
    if method == "baseline":
        return methods.baseline(match, params["window"])
    if method == "drift":
        return methods.drift(match, params["gain"], params["drift_frames"], params["max_gap"],
                             params["window"])
    return methods.rules(match, params)


def save_choice(config, method, best, per_game, logo, extra=None):
    target = path("tuned.json")
    tuned = json.load(open(target)) if os.path.isfile(target) else {}
    tuned[method] = dict({"params": json.loads(best), "sample_mean": per_game, "logo": logo}, **(extra or {}))
    with open(target, "w") as handle:
        json.dump(tuned, handle, indent=2)
    print("  {:<10} spellF1 {:.3f} (leave one game out {:.3f})  {}".format(
        method, np.mean(list(per_game.values())), np.mean(list(logo.values())), best), flush=True)


def tuned_params(method):
    with open(path("tuned.json")) as handle:
        return json.load(handle)[method]["params"]


# --- tune: the methods without training -----------------------------------------------------------

def tune(config):
    matches = {key: match_of(key, config) for key in keys(config)}
    for method in ("baseline", "drift", "rules"):
        rows = []
        candidates = list(rule_candidates(config, method))
        start = time.time()
        for key, match in matches.items():
            for name, params in candidates:
                spell, macro = methods.score(match, label_rule(method, match, params))
                rows.append({"candidate": name, "game": key[0], "setting": key[1],
                             "spellF1": spell, "macroF1": macro})
        table = pd.DataFrame(rows)
        table.to_csv(path("tune-{}.csv".format(method)), index=False)
        print("  {}: {} candidates x {} matches, {:.0f} s".format(
            method, len(candidates), len(matches), time.time() - start), flush=True)
        save_choice(config, method, *select(table))


# --- train: the methods with a model --------------------------------------------------------------

def linear_spec(config, variant):
    spec = dict(config["linear"])
    if variant == "linear-net":
        spec["owner"] = spec["owner"] + spec["network_owner"]
    return spec


def netfuse_names(config):
    return [name + side for name in config["netfuse"]["features"] for side in ("_h", "_a")]


def out_of_game(config, fit, predict):
    """{key: prediction} where each sample game is predicted by a model fitted on the other
    games (all settings pooled). `fit(pool)` -> model, `predict(model, values)` -> arrays."""
    values = {key: values_of(key, config) for key in keys(config)}
    labels = {key: data.sample_match(*key)["labels"] for key in keys(config)}
    out = {}
    for held in GAMES_SAMPLE:
        model = fit([(values[k], labels[k]) for k in keys(config) if k[0] != held])
        for key in keys(config):
            if key[0] == held:
                out[key] = predict(model, values[key])
    return out, fit([(values[k], labels[k]) for k in keys(config)])


def train(config):
    matches = {key: match_of(key, config) for key in keys(config)}
    models = {}

    for variant in ("linear", "linear-net"):
        spec = linear_spec(config, variant)
        start = time.time()
        probs, models[variant] = out_of_game(
            config, lambda pool: methods.fit_linear(pool, spec),
            lambda model, values: methods.linear_probs(model, values, spec))
        rows = []
        for team_cost, dead_cost in itertools.product(spec["costs"]["team"], spec["costs"]["dead"]):
            name = json.dumps({"team": team_cost, "dead": dead_cost}, sort_keys=True)
            for key, match in matches.items():
                labels = methods.decode(match, *probs[key], team_cost, dead_cost)
                spell, macro = methods.score(match, labels)
                rows.append({"candidate": name, "game": key[0], "setting": key[1],
                             "spellF1": spell, "macroF1": macro})
        table = pd.DataFrame(rows)
        table.to_csv(path("tune-{}.csv".format(variant)), index=False)
        print("  {}: fits and {} cost pairs, {:.0f} s".format(
            variant, len(table.candidate.unique()), time.time() - start), flush=True)
        save_choice(config, variant, *select(table))

    cfg = config["netfuse"]
    names, smooth = netfuse_names(config), int(round(cfg["smooth_seconds"] * 25))
    start = time.time()
    logits, models["netfuse"] = out_of_game(
        config, lambda pool: methods.fit_owner(pool, names, smooth, cfg["l2"]),
        lambda model, values: methods.owner_logit(model, values, names, smooth))
    base = tuned_params("rules")
    rows = []
    for weight, switch_cost in itertools.product(cfg["weight"], cfg["switch_cost"]):
        params = dict(base, switch_cost=switch_cost)
        name = json.dumps({"weight": weight, "switch_cost": switch_cost}, sort_keys=True)
        for key, match in matches.items():
            labels = methods.rules(match, params, extra=logits[key], weight=weight, clip=cfg["clip"])
            spell, macro = methods.score(match, labels)
            rows.append({"candidate": name, "game": key[0], "setting": key[1],
                         "spellF1": spell, "macroF1": macro})
    table = pd.DataFrame(rows)
    table.to_csv(path("tune-netfuse.csv"), index=False)
    print("  netfuse: fits and {} weights, {:.0f} s".format(len(table.candidate.unique()), time.time() - start))
    save_choice(config, "netfuse", *select(table))

    with open(path("models.pkl"), "wb") as handle:
        pickle.dump(models, handle)
    print("  final models (fitted on the 3 games) -> {}".format(os.path.relpath(path("models.pkl"))))


# --- predict --------------------------------------------------------------------------------------

def predict(config, out_dir):
    game = config["game"]
    match = data.bar_ath_match(game, artifacts_dir(NAME))
    values = values_of((game, "atd"), config)
    peer("fg03", "model")              # the pickled classes live in fg03's and fg02's packages
    peer("fg02", "stats")
    with open(path("models.pkl"), "rb") as handle:
        models = pickle.load(handle)
    (tags_path,) = require(["smart_tagging"], game)
    tags = peer("fg01", "signals").read_tags(tags_path, match["fps"])
    prior = config["tags_prior"]

    out = {"baseline": label_rule("baseline", match, tuned_params("baseline")),
           "drift": label_rule("drift", match, tuned_params("drift"))}
    rules = tuned_params("rules")
    out["rules"] = methods.rules(match, rules)
    out["rules-st"] = methods.rules(match, dict(rules, **prior), tags=tags)

    cfg, fuse = config["netfuse"], tuned_params("netfuse")
    logit = methods.owner_logit(models["netfuse"], values, netfuse_names(config),
                                int(round(cfg["smooth_seconds"] * 25)))
    fused = dict(rules, switch_cost=fuse["switch_cost"])
    out["netfuse"] = methods.rules(match, fused, extra=logit, weight=fuse["weight"], clip=cfg["clip"])
    out["netfuse-st"] = methods.rules(match, dict(fused, **prior), tags=tags, extra=logit,
                                      weight=fuse["weight"], clip=cfg["clip"])

    for variant in ("linear", "linear-net"):
        costs = tuned_params(variant)
        p_home, p_dead = methods.linear_probs(models[variant], values, linear_spec(config, variant))
        out[variant] = methods.decode(match, p_home, p_dead, costs["team"], costs["dead"])

    for variant, labels in out.items():
        target = os.path.join(out_dir, "{}-{}.csv".format(NAME, variant))
        write_submission(target, methods.mask_outside(match, labels))
        print("  -> {}".format(os.path.relpath(target)))


def report(config):
    with open(path("tuned.json")) as handle:
        tuned = json.load(handle)
    rows = [{"method": method, "sample spellF1": np.mean(list(t["sample_mean"].values())),
             "leave one game out": np.mean(list(t["logo"].values())), "params": json.dumps(t["params"])}
            for method, t in tuned.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all",))
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()
    config = load_config()
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print("[{}]".format(stage), flush=True)
        if stage == "predict":
            predict(config, args.out)
        else:
            globals()[stage](config)


if __name__ == "__main__":
    main()
