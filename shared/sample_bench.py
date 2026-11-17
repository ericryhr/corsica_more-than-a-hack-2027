"""Score a solution on Metrica's 3 sample games: spellF1 on ~280 minutes instead of 5.

    python shared/sample_bench.py nc02-frozen-width
    python shared/sample_bench.py nc02-frozen-width --variants s50-w125 sinf-w125
    python shared/sample_bench.py nc02-frozen-width --versus 00-baseline:w50 --bootstrap 1000
    python shared/sample_bench.py 00-baseline --settings clean default harsh
    python shared/sample_bench.py nc03-dead-ball --set owner_solution=er01-weighted-ensemble \
        'owner_variants=["b0-s75"]'

A solution takes part by having `solutions/<name>/src/bench.py` with

    def label_match(match, config) -> {variant: labels}

where `match` is shared/sample.py's dict (players, team, ball, speed, halves, fps, pitch; the
tracking arrays the solutions already read from the ATD, and the frames of each half) and `config` is the solution's config.json. Labels
are 'H'/'A'/'D' per frame; variant names are the ones run.py gives its submissions. The same
function should be what run.py calls on the challenge tracking, so the bench scores the code that
makes the submission.

Settings (see shared/sample.py): `default` makes the sample tracking look like the BAR-ATH ATD
profile, `harsh` is a stronger version, `clean` is the sample as recorded, `blurred` is default with
the ball as blurred and imprecise as the ATD's (made for touch detectors; not calibrated for spellF1). The calibration ranks the
published baselines like the full match under every setting, but its scores read 0.04-0.09 high,
so compare solutions with each other, not with the leaderboard. The team rule: A beats B when the
mean difference is above 0.04 and its 5-95% interval excludes zero, under BOTH default and harsh.

Nothing is written into solutions/*/info.json (that is for the challenge ground truth). `--csv`
saves the table, e.g. to reports/. `--set KEY=VALUE` overrides the solution's config for this run
(VALUE is read as JSON, else as a string), to try a setting without editing config.json.
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.challenge import SOLUTIONS_DIR, pm  # noqa: E402
from shared.sample import (DEGRADE, DEGRADE_BLURRED, DEGRADE_HARSH, DEGRADE_HARSH_BLURRED, GAMES_SAMPLE,  # noqa: E402
                           degrade, load_match)
from shared.score import spell_draws  # noqa: E402

SETTINGS = {"clean": None, "default": DEGRADE, "harsh": DEGRADE_HARSH, "blurred": DEGRADE_BLURRED,
            "harsh-blurred": DEGRADE_HARSH_BLURRED}
BETTER_BY = 0.04


def load_solution(name):
    """(label_match, config) of solutions/<name>. Each solution's `src` package is loaded under its
    own module name, so two solutions can be compared in one run."""
    folder = os.path.join(SOLUTIONS_DIR, name)
    src = os.path.join(folder, "src")
    if not os.path.isfile(os.path.join(src, "bench.py")):
        sys.exit("{} has no src/bench.py with label_match(match, config); see shared/sample_bench.py".format(name))
    package = "bench_" + name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(package, os.path.join(src, "__init__.py"),
                                                  submodule_search_locations=[src])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    with open(os.path.join(folder, "config.json")) as handle:
        config = json.load(handle)
    return importlib.import_module(package + ".bench").label_match, config


def matches(setting, games):
    """(game, match) for each sample game under a setting. Degradation is seeded by the game."""
    for game in games:
        match = load_match(game)
        if SETTINGS[setting] is not None:
            match = degrade(match, seed=game, **SETTINGS[setting])
        yield game, match


def run_solution(label_match, config, match, variants=None):
    out = label_match(match, config)
    if variants:
        missing = [v for v in variants if v not in out]
        if missing:
            sys.exit("unknown variants {}; this solution makes {}".format(missing, sorted(out)))
        out = {v: out[v] for v in variants}
    for variant, labels in out.items():
        if len(labels) != len(match["labels"]):
            sys.exit("{}: {} labels for {} frames".format(variant, len(labels), len(match["labels"])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("solution", help="folder name in solutions/, e.g. nc02-frozen-width")
    parser.add_argument("--variants", nargs="+", help="only these variants (default: all)")
    parser.add_argument("--settings", nargs="+", default=["default", "harsh"], choices=sorted(SETTINGS))
    parser.add_argument("--games", nargs="+", type=int, default=list(GAMES_SAMPLE))
    parser.add_argument("--versus", help="reference as SOLUTION:VARIANT, e.g. 00-baseline:w50")
    parser.add_argument("--bootstrap", type=int, default=1000, help="draws for --versus (default 1000)")
    parser.add_argument("--csv", help="also save the table here")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override the solution's config.json for this run")
    args = parser.parse_args()

    label_match, config = load_solution(args.solution)
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            config[key] = json.loads(value)
        except ValueError:
            config[key] = value
    reference = None
    if args.versus:
        ref_name, _, ref_variant = args.versus.partition(":")
        if not ref_variant:
            sys.exit("--versus needs SOLUTION:VARIANT")
        reference = load_solution(ref_name) + (ref_variant,)

    rows, verdicts = [], {}
    for setting in args.settings:
        diffs = {}                                     # variant -> per-game arrays of draw differences
        for game, match in matches(setting, args.games):
            print("  {} game {} ...".format(setting, game), file=sys.stderr, flush=True)
            gt = pm.label_codes(match["labels"])
            predictions = run_solution(label_match, config, match, args.variants)
            if reference:
                ref_labels = run_solution(reference[0], reference[1], match, [reference[2]])[reference[2]]
                ref_draws = spell_draws(gt, pm.label_codes(ref_labels), args.bootstrap)
            for variant, labels in predictions.items():
                codes = pm.label_codes(labels)
                result = pm.evaluate(gt, codes, bootstrap_n=0)
                rows.append({"setting": setting, "variant": variant, "game": game,
                             "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
                             "possessions": result["n_pred_spells"], "gt possessions": result["n_gt_spells"]})
                if reference:
                    diffs.setdefault(variant, []).append(spell_draws(gt, codes, args.bootstrap) - ref_draws)
        for variant, per_game in diffs.items():
            draws = np.mean(per_game, axis=0)          # paired draw by draw, averaged over the games
            low, high = np.percentile(draws, [5, 95])
            verdicts.setdefault(variant, {})[setting] = (draws.mean(), low, high)

    table = pd.DataFrame(rows)
    spell = table.pivot_table(index="variant", columns=["setting", "game"], values="spellF1")
    summary = table.groupby(["variant", "setting"]).agg(spellF1=("spellF1", "mean"), macroF1=("macroF1", "mean"),
                                                        possessions=("possessions", "sum"),
                                                        gt_possessions=("gt possessions", "sum")).unstack("setting")
    pd.set_option("display.width", 200)
    print("{} on sample games {} (settings: {})\n".format(args.solution, args.games, ", ".join(args.settings)))
    print("spellF1 per game:")
    print(spell.round(3).to_string())
    print("\nmean over games (possessions summed, vs the truth's):")
    print(summary.round(3).sort_values(("spellF1", args.settings[0]), ascending=False).to_string())

    if reference:
        print("\nversus {} (paired bootstrap, {} draws; better = diff > {} and interval excludes 0 under every setting):"
              .format(args.versus, args.bootstrap, BETTER_BY))
        for variant, by_setting in verdicts.items():
            cells = ["{} {:+.3f} [{:+.3f}, {:+.3f}]".format(s, *by_setting[s]) for s in args.settings]
            better = all(by_setting[s][0] > BETTER_BY and by_setting[s][1] > 0 for s in args.settings)
            worse = all(by_setting[s][0] < -BETTER_BY and by_setting[s][2] < 0 for s in args.settings)
            print("  {:<22} {}   -> {}".format(variant, "   ".join(cells),
                                               "BETTER" if better else "WORSE" if worse else "no clear difference"))

    if args.csv:
        folder = os.path.dirname(args.csv)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        table.to_csv(args.csv, index=False)
        print("\n-> {}".format(args.csv))


if __name__ == "__main__":
    main()
