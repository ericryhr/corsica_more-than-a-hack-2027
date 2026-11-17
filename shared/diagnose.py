"""Why a submission loses spellF1: sort the true and predicted possessions into the ways they fail.

    python shared/diagnose.py submissions/nc01-context-nearest-width-w125.csv
    python shared/diagnose.py submissions/*-w125.csv --gt labels/fg_30_11175-56199.csv --window 11185 56199

Prints counts only, no frames or labels, so the output can be shared. Same `--gt` and `--window`
as score.py. Possessions are spells as the scorer builds them (X forward-filled), and "matched"
means matched by the scorer's own one-to-one greedy matching at `--tau` (0.5 by default, the
middle of the 0.3-0.7 range spellF1 averages over).

True possessions not matched are sorted into the first case that applies:

  no overlap  no predicted possession of the same team touches it: wrong team, or D, all along
  merged      its best predicted possession also overlaps another true possession of the same
              team, across a stoppage or a short possession of the other team
  split       two or more predicted possessions of its team cover it, none well enough
  boundary    one predicted possession covers it, but starts or ends too far away

Predicted possessions not matched are sorted by what the truth says on their frames: mostly a
stoppage (D), mostly the other team, or mostly the same team (a piece of a split or a merge).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.challenge import GAME, GAMES, PUBLIC_GT, pm  # noqa: E402
from shared.score import load_codes, restrict  # noqa: E402

FPS = 25
DURATION_BUCKETS = ((0, 2), (2, 5), (5, 15), (15, None))  # seconds


def bucket_name(low, high):
    return "{}+ s".format(low) if high is None else "{}-{} s".format(low, high)


def bucket_of(frames):
    seconds = frames / float(FPS)
    for low, high in DURATION_BUCKETS:
        if high is None or seconds < high:
            return bucket_name(low, high)


def diagnose(gt_codes, pred_codes, tau):
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), tau)

    # the prediction each true spell was matched to, replaying the same greedy matching
    pred_of_gt, used = {}, set()
    for iou, i, j in overlaps:
        if iou < tau:
            break
        if i in pred_of_gt or j in used:
            continue
        pred_of_gt[i] = j
        used.add(j)

    by_gt = {i: [] for i in range(len(gt_spells))}
    by_pred = {j: [] for j in range(len(pred_spells))}
    for iou, i, j in overlaps:
        by_gt[i].append((iou, j))
        by_pred[j].append(i)

    cases, offsets = {}, []
    by_bucket = {bucket_name(*b): [0, 0] for b in DURATION_BUCKETS}
    for i, (start, stop, _) in enumerate(gt_spells):
        bucket = by_bucket[bucket_of(stop - start)]
        bucket[0] += 1
        candidates = sorted(by_gt[i], reverse=True)
        if candidates:
            best = pred_spells[candidates[0][1]]
            offsets.append((best[0] - start, best[1] - stop))
        if matched[i]:
            bucket[1] += 1
            continue
        if not candidates:
            case = "no overlap"
        elif len(by_pred[candidates[0][1]]) > 1:
            case = "merged"
        elif len(candidates) > 1:
            case = "split"
        else:
            case = "boundary"
        cases[case] = cases.get(case, 0) + 1

    spurious = {}
    for j, (start, stop, code) in enumerate(pred_spells):
        if j in used:
            continue
        truth = gt_codes[start:stop]
        shares = {"stoppage (D)": np.mean(truth == pm.D),
                  "other team": np.mean((truth != code) & (truth != pm.D) & (truth != pm.X)),
                  "same team": np.mean(truth == code)}
        kind = max(shares, key=shares.get)
        key = (kind, bucket_of(stop - start))
        spurious[key] = spurious.get(key, 0) + 1

    stoppages = [(a, b) for a, b, k in pm.segments(gt_codes, runs) if k == pm.D]
    bridged = sum(1 for a, b in stoppages if np.mean(filled[a:b] == pm.D) < 0.1)

    return {"n_gt": len(gt_spells), "n_pred": len(pred_spells), "n_matched": int(matched.sum()),
            "cases": cases, "by_bucket": by_bucket, "offsets": np.array(offsets, dtype=float),
            "spurious": spurious, "n_stoppages": len(stoppages), "bridged": bridged}


def report(name, d, tau):
    print("\n=== {}  (tau {})".format(name, tau))
    print("true possessions {}, predicted {}, matched {}".format(d["n_gt"], d["n_pred"], d["n_matched"]))

    print("\ntrue possessions NOT matched, by cause:")
    for case in ("no overlap", "merged", "split", "boundary"):
        print("  {:<12} {:>4}".format(case, d["cases"].get(case, 0)))

    print("\ntrue possessions matched, by length:")
    for bucket, (total, hit) in d["by_bucket"].items():
        print("  {:<8} {:>3} of {:>3}".format(bucket, hit, total))

    if len(d["offsets"]):
        start, stop = d["offsets"][:, 0] / FPS, d["offsets"][:, 1] / FPS
        print("\nboundary offset of the best predicted possession, seconds (+ = late):")
        print("  start  median {:+.1f}   25-75% [{:+.1f}, {:+.1f}]".format(
            np.median(start), *np.percentile(start, (25, 75))))
        print("  end    median {:+.1f}   25-75% [{:+.1f}, {:+.1f}]".format(
            np.median(stop), *np.percentile(stop, (25, 75))))

    print("\npredicted possessions NOT matched, by what the truth says there:")
    for kind in ("stoppage (D)", "other team", "same team"):
        row = [d["spurious"].get((kind, bucket_name(*b)), 0) for b in DURATION_BUCKETS]
        print("  {:<13} {:>4}   ({})".format(kind, sum(row), ", ".join(
            "{} {}".format(n, bucket_name(*b)) for n, b in zip(row, DURATION_BUCKETS))))

    print("\ntrue stoppages {}, bridged by the prediction (under 10% D): {}".format(
        d["n_stoppages"], d["bridged"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+", help="frame,label CSVs to diagnose")
    parser.add_argument("--gt", default="public", help="'public' or a path to a frame,label CSV")
    parser.add_argument("--window", type=int, nargs=2, metavar=("START", "END"),
                        help="only this frame range, both ends inclusive")
    parser.add_argument("--tau", type=float, default=0.5, help="IoU threshold for a match")
    args = parser.parse_args()

    n_frames = GAMES[GAME]["n_video_frames"]
    gt_path = PUBLIC_GT if args.gt == "public" else args.gt
    gt_codes = restrict(load_codes(gt_path, n_frames), args.window)
    for path in args.submissions:
        name = os.path.splitext(os.path.basename(path))[0]
        report(name, diagnose(gt_codes, load_codes(path, n_frames), args.tau), args.tau)


if __name__ == "__main__":
    main()
