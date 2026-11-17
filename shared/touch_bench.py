"""Ball touches on Metrica's 3 sample games: what a touch looks like in the tracking, and a scorer.

    python shared/touch_bench.py                           # profile of the true touches
    python shared/touch_bench.py --reference               # score the naive reference detector
    python shared/touch_bench.py <solution> [--tolerance 0.4] [--settings blurred default]

The labels' owner changes at events, so the true touches are the events' (shared/sample.py
touches_from_events): kicks, receptions and ball wins, with the team. A solution takes part by
having `solutions/<name>/src/bench.py` with

    def detect_touches(match, config) -> DataFrame with columns frame, team ('H'/'A')

on the same `match` dict as shared/sample_bench.py, so the same function can run on the challenge
tracking. Scores, per game and setting:

  recall      share of true touches with a predicted touch within the tolerance
  precision   share of predicted touches with a true touch within the tolerance
  team        of the matched pairs, share with the right team
  changes     share of true touches where possession can pass to the other team (`change`)
              found with the right team: the ones the owner models miss

Matching is one to one, nearest in time first.

Caution: the sample ball looks rebuilt from the events. Between touches it runs almost straight
(0.2 m off the line) at almost constant speed (6% variation), with ~1 mm of jitter, and at each
touch it sits on the player (median 0.1 m) at the event frame (+-1 frame). The naive reference
finds 80% of the touches at 96% precision here; on real tracking it will not.
So detectors are scored by default under `blurred`: the sample ball blurred over 1 s, 0.75 m of
drifting error and 6 frames early, which matches the ATD ball's totals (shared/sample.py BALL_ATD).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.sample import FPS, GAMES_SAMPLE, SCORED_KINDS, load_match, load_touches  # noqa: E402
from shared.sample_bench import SETTINGS, load_solution, matches  # noqa: E402

KICK_FRAMES = 3                 # velocity before and after a frame, over this many frames
PEAK_WINDOW = 12                # frames either side of an event to look for the velocity change


def velocity_change(ball, k=KICK_FRAMES):
    """|v after - v before| per frame (m/s), the ball's velocity over k frames each side."""
    n = len(ball)
    dv = np.full(n, np.nan)
    before = (ball[k:n - k] - ball[:n - 2 * k]) * FPS / k
    after = (ball[2 * k:] - ball[k:n - k]) * FPS / k
    dv[k:n - k] = np.linalg.norm(after - before, axis=1)
    return dv


def nearest(players, team, ball, side):
    """Distance from the ball to the nearest player of `side` (0 home, 1 away), per frame."""
    d = np.linalg.norm(players[:, team == side] - ball[:, None, :], axis=2)
    with np.errstate(all="ignore"):
        return np.where(np.all(np.isnan(d), axis=1), np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))


def match_touches(true_frames, pred_frames, tolerance):
    """One-to-one pairs (i_true, i_pred) within `tolerance` frames, nearest first."""
    pairs = []
    for i, f in enumerate(true_frames):
        lo, hi = np.searchsorted(pred_frames, [f - tolerance, f + tolerance + 1])
        pairs += [(abs(int(pred_frames[j]) - int(f)), i, j) for j in range(lo, hi)]
    used_t, used_p, out = set(), set(), []
    for _, i, j in sorted(pairs):
        if i not in used_t and j not in used_p:
            used_t.add(i), used_p.add(j)
            out.append((i, j))
    return out


def score(true, pred, tolerance):
    """recall, precision, team, changes (see the module docstring) of `pred` against `true`."""
    true = true[true.kind.isin(SCORED_KINDS)].reset_index(drop=True)
    pred = pred.sort_values("frame").reset_index(drop=True)
    pairs = match_touches(true.frame.to_numpy(), pred.frame.to_numpy(), tolerance)
    right = [(i, j) for i, j in pairs if true.team[i] == pred.team[j]]
    found_change = {i for i, _ in right} & set(np.flatnonzero(true.change.to_numpy()))
    return {"recall": len(pairs) / max(len(true), 1), "precision": len(pairs) / max(len(pred), 1),
            "team": len(right) / max(len(pairs), 1), "changes": len(found_change) / max(int(true.change.sum()), 1),
            "true": len(true), "predicted": len(pred)}


def naive_touches(match, threshold=4.0, radius=2.0):
    """Reference detector: local peaks of the ball's velocity change above `threshold` m/s with a
    player within `radius` m; the touch goes to the nearest player's team."""
    ball, players, team = match["ball"], match["players"], match["team"]
    dv = velocity_change(ball)
    peak = np.zeros(len(dv), bool)
    peak[1:-1] = (dv[1:-1] >= threshold) & (dv[1:-1] >= dv[:-2]) & (dv[1:-1] > dv[2:])
    home, away = nearest(players, team, ball, 0), nearest(players, team, ball, 1)
    with np.errstate(invalid="ignore"):
        close = np.fmin(home, away) <= radius
    frames = np.flatnonzero(peak & close)
    return pd.DataFrame({"frame": frames, "team": np.where(home[frames] <= away[frames], "H", "A")})


def profile():
    """What a true touch looks like in the clean sample tracking, and how often the degraded
    tracking still sees the ball there."""
    rows, background = [], []
    for game in GAMES_SAMPLE:
        match = load_match(game)
        ball, players, team, labels = match["ball"], match["players"], match["team"], match["labels"]
        dv = velocity_change(ball)
        home, away = nearest(players, team, ball, 0), nearest(players, team, ball, 1)
        seen = {name: ~np.isnan(degrade_ball(match, params)[:, 0]) for name, params in SETTINGS.items() if params}
        touches = load_touches(game)
        touches = touches[touches.kind.isin(SCORED_KINDS)]
        n = len(ball)
        for t in touches.itertuples():
            lo, hi = max(0, t.frame - PEAK_WINDOW), min(n, t.frame + PEAK_WINDOW + 1)
            window = dv[lo:hi]
            own, other = (home, away) if t.team == "H" else (away, home)
            row = {"game": game, "kind": t.kind, "change": t.change, "ball seen": not np.isnan(ball[t.frame, 0]),
                   "own player m": own[t.frame], "other player m": other[t.frame],
                   "own nearest": own[t.frame] < other[t.frame],
                   "own min ±0.5 s m": np.nanmin(own[lo:hi]) if np.any(~np.isnan(own[lo:hi])) else np.nan,
                   "dv at frame": dv[t.frame],
                   "dv peak ±0.5 s": np.nanmax(window) if np.any(~np.isnan(window)) else np.nan,
                   "peak offset frames": (lo + int(np.nanargmax(window)) - t.frame) if np.any(~np.isnan(window)) else np.nan}
            row.update({"seen " + name: bool(np.any(s[lo:hi])) for name, s in seen.items()})
            rows.append(row)
        live = np.flatnonzero(((labels == "H") | (labels == "A")) & ~np.isnan(dv))
        background.append(dv[np.random.default_rng(game).choice(live, 5000, replace=False)])
    t = pd.DataFrame(rows)
    bg = np.concatenate(background)
    pd.set_option("display.width", 200)

    print("true touches (kick, receive, win) on the 3 sample games: {}, {:.1f} per minute of live play, "
          "{} where possession can change".format(len(t), len(t) / (sum(np.sum(np.isin(load_match(g)["labels"],
                                                                            ["H", "A"])) for g in GAMES_SAMPLE) / FPS / 60),
                                                  int(t.change.sum())))
    print("\nball seen at the touch: clean {:.0%}; within ±0.5 s after degrading: {}".format(
        t["ball seen"].mean(), ", ".join("{} {:.0%}".format(c[5:], t[c].mean()) for c in t if c.startswith("seen "))))

    seen = t[t["ball seen"]]
    print("\ndistance ball - nearest player of the touching team (m), ball seen:")
    q = [0.25, 0.5, 0.75, 0.9]
    print(seen.groupby("kind")[["own player m", "own min ±0.5 s m", "other player m"]].quantile(q).unstack().round(1).to_string())
    print("\nthe touching team's player is the nearest one at the event frame:",
          ", ".join("{} {:.0%}".format(k, v) for k, v in seen.groupby("kind")["own nearest"].mean().items()),
          "| at a change: {:.0%}".format(seen[seen.change]["own nearest"].mean()))

    print("\nball velocity change |v after - v before| (m/s, over {} frames each side):".format(KICK_FRAMES))
    print("  live frames at random: median {:.1f}, 90% {:.1f}, 99% {:.1f}".format(*np.percentile(bg, [50, 90, 99])))
    for kind, g in seen.groupby("kind"):
        print("  {:<8} peak within ±0.5 s: median {:.1f}, 25% {:.1f}, 10% {:.1f}".format(
            kind, *np.nanpercentile(g["dv peak ±0.5 s"], [50, 25, 10])))
    offset = seen["peak offset frames"].dropna()
    print("\nwhere that peak sits from the event frame (frames; + = after): median {:+.0f}, "
          "50% within ±{:.0f}, 80% within ±{:.0f}".format(offset.median(), np.percentile(np.abs(offset), 50),
                                                         np.percentile(np.abs(offset), 80)))


def degrade_ball(match, params):
    from shared.sample import degrade
    return degrade(match, seed=match["game"], **params)["ball"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("solution", nargs="?", help="folder name in solutions/ with src/bench.py detect_touches")
    parser.add_argument("--reference", action="store_true", help="score the naive reference detector")
    parser.add_argument("--tolerance", type=float, default=0.4, help="seconds (default 0.4)")
    parser.add_argument("--settings", nargs="+", default=["blurred"], choices=sorted(SETTINGS))
    args = parser.parse_args()

    if not args.solution and not args.reference:
        profile()
        return
    if args.solution:
        import importlib
        load_solution(args.solution)
        bench = importlib.import_module("bench_{}.bench".format(args.solution.replace("-", "_")))
        config = load_solution(args.solution)[1]
        detect, name = (lambda m: bench.detect_touches(m, config)), args.solution
    else:
        detect, name = naive_touches, "reference (naive_touches)"

    rows = []
    for setting in args.settings:
        for game, match in matches(setting, GAMES_SAMPLE):
            print("  {} game {} ...".format(setting, game), file=sys.stderr, flush=True)
            rows.append(dict(setting=setting, game=game,
                             **score(load_touches(game), detect(match), int(round(args.tolerance * FPS)))))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("{} on the sample games, tolerance {} s\n".format(name, args.tolerance))
    print(table.round(3).to_string(index=False))
    print("\nmean over games:")
    print(table.groupby("setting")[["recall", "precision", "team", "changes"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
