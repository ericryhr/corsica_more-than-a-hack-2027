"""Validate and score one or more submissions and record the result in their solution's info.json.

    python shared/score.py submissions/00-baseline-w50.csv
    python shared/score.py submissions/*.csv --gt public
    python shared/score.py submissions/02-x.csv --gt labeler/data/exports/my_labels.csv
    python shared/score.py submissions/02-x.csv --window 11185 41184
    python shared/score.py submissions/02-x-*.csv --bootstrap 1000
    python shared/score.py submissions/02-x-*.csv --bootstrap 1000 --versus submissions/00-baseline-w125.csv

`--gt` takes `public` (the first 5 minutes shipped with the challenge repo) or the path to a
`frame,label` CSV, for example an export from the labeler. `--window START END` (frames, both
inclusive) sets the rest of the ground truth to `X`, so nothing outside the window is scored.

`--bootstrap N` adds a 5-95% interval for spellF1, resampling the true possessions N times (the
challenge's own bootstrap). `--versus REF` also gives the interval of the difference with REF on the
same draws, which is the team's rule for "better": more than 0.04 and an interval that excludes zero.

The owner of a submission is the folder in `solutions/` whose name is a prefix of the submission
name: `00-baseline-w50.csv` belongs to `solutions/00-baseline/`. Only that folder's `info.json` is
written (see shared/info.py). Scoring the same name again against the same ground truth and window
replaces its row.
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import info  # noqa: E402
from shared.challenge import GAME, GAMES, PUBLIC_GT, SOLUTIONS_DIR, pm  # noqa: E402


def load_codes(path, n_frames):
    """Label codes 0..3 from a `frame,label` CSV. Raises ValueError with the reason if invalid."""
    frame = pm.read_label_csv(path)
    reason = pm.validate_submission(frame, n_frames)
    if reason:
        raise ValueError(reason)
    return pm.label_codes(frame)


def restrict(gt_codes, window):
    """Copy of the ground truth with `X` outside the [start, end] window."""
    if window is None:
        return gt_codes
    start, end = window
    out = np.full_like(gt_codes, pm.X)
    out[start:end + 1] = gt_codes[start:end + 1]
    return out


def spell_draws(gt_codes, pred_codes, n_draws, seed=None):
    """spellF1 on each bootstrap draw of the true possessions, as the challenge's bootstrap_spell_f1.

    The draws depend only on the seed and the number of true possessions, so two submissions scored
    on the same ground truth get the same draws, and their difference per draw is a paired sample.
    """
    seed = pm.BOOTSTRAP_SEED if seed is None else seed
    runs = pm.valid_runs(gt_codes)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(pm.fill_abstentions(pred_codes, runs), runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    n_gt, n_pred = len(gt_spells), len(pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, n_gt, threshold) for threshold in pm.SPELL_IOUS])

    rng = np.random.RandomState(seed)
    scores = np.empty(n_draws)
    for draw in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        f1s = []
        for row in flags:
            true_positive = int(row[picks].sum())
            false_negative = n_gt - true_positive
            false_positive = max(0, n_pred - true_positive)
            f1s.append(2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                       if true_positive else 0.0)
        scores[draw] = np.mean(f1s)
    return scores


def owner(name):
    """Solution folder whose name is a prefix of the submission name; the longest if several."""
    folders = [f for f in os.listdir(SOLUTIONS_DIR)
               if not f.startswith((".", "_")) and os.path.isdir(os.path.join(SOLUTIONS_DIR, f))
               and name.startswith(f + "-")]
    if not folders:
        raise ValueError("no folder in solutions/ is a prefix of '{}'; "
                         "name the submission <solution folder>-<variant>".format(name))
    return os.path.join(SOLUTIONS_DIR, max(folders, key=len))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+", help="frame,label CSVs to score")
    parser.add_argument("--gt", default="public", help="'public' or a path to a frame,label CSV")
    parser.add_argument("--window", type=int, nargs=2, metavar=("START", "END"),
                        help="score only this frame range, both ends inclusive")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="draws for the spellF1 interval (1000 is enough); 0 = off")
    parser.add_argument("--versus", metavar="REF",
                        help="with --bootstrap: interval of the spellF1 difference against this CSV")
    args = parser.parse_args()
    if args.versus and not args.bootstrap:
        parser.error("--versus needs --bootstrap N")

    n_frames = GAMES[GAME]["n_video_frames"]
    gt_path = PUBLIC_GT if args.gt == "public" else args.gt
    gt_codes = restrict(load_codes(gt_path, n_frames), args.window)
    gt_name = "public" if args.gt == "public" else os.path.basename(gt_path)
    window = "{}-{}".format(*args.window) if args.window else "all"

    reference = None
    if args.versus:
        reference = spell_draws(gt_codes, load_codes(args.versus, n_frames), args.bootstrap)

    failed = 0
    for path in args.submissions:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            folder = owner(name)
            pred_codes = load_codes(path, n_frames)
            result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=args.bootstrap)
            if info.problems(folder):
                raise ValueError("{}/info.json is not valid: {}".format(
                    os.path.basename(folder), "; ".join(info.problems(folder))))
        except ValueError as error:
            print("{:<32} INVALID: {}".format(name, error))
            failed += 1
            continue

        row = {
            "submission": name, "gt": gt_name, "window": window,
            "spell_f1_mean": round(float(result["spell_f1_mean"]), 4),
            "macro_f1": round(float(result["macro_f1"]), 4),
            "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"],
            "accuracy": round(float(result["accuracy"]), 4),
            "n_graded": result["n_graded"],
            "n_abstain_on_graded": result["n_abstain_on_graded"],
            "date": datetime.date.today().isoformat(),
        }
        line = "{:<32} spellF1 {:.4f}  macroF1 {:.4f}  possessions {} (gt {})".format(
            name, row["spell_f1_mean"], row["macro_f1"], row["n_pred_spells"], row["n_gt_spells"])
        if args.bootstrap:
            row["spell_bootstrap_low"] = round(result["spell_bootstrap_low"], 4)
            row["spell_bootstrap_high"] = round(result["spell_bootstrap_high"], 4)
            line += "  [{:.3f}, {:.3f}]".format(row["spell_bootstrap_low"], row["spell_bootstrap_high"])
        if reference is not None:
            low, high = np.percentile(spell_draws(gt_codes, pred_codes, args.bootstrap) - reference,
                                      pm.BOOTSTRAP_PERCENTILES)
            line += "  vs ref [{:+.3f}, {:+.3f}]{}".format(low, high, "" if low <= 0 <= high else " *")
        info.upsert_result(folder, row)
        print(line)

    if reference is not None:
        print("\nvs ref = 5-95% interval of (spellF1 - spellF1 of {}) over {} paired draws; "
              "* = excludes zero".format(os.path.basename(args.versus), args.bootstrap))
    print("\nResults in solutions/*/info.json  (gt={}, window={})".format(gt_name, window))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
