"""er02-metrica-supervised: possession cues, a dead-ball detector and the decoder, all trained and
validated on Metrica's three public sample games (labels derived from their events).

    python solutions/er02-metrica-supervised/run.py                 # all stages
    python solutions/er02-metrica-supervised/run.py fetch           # clone the sample data (pinned)
    python solutions/er02-metrica-supervised/run.py prepare         # cues for 3 games + BAR-ATH
    python solutions/er02-metrica-supervised/run.py train           # leave-one-game-out + final fit
    python solutions/er02-metrica-supervised/run.py predict         # BAR-ATH -> submissions

Writes submissions/er02-metrica-supervised-<variant>.csv. Stages communicate through
artifacts/er02-metrica-supervised/. Nothing here reads a label of BAR-ATH: every label comes from
other matches (RULES.md allows hand labels from other games). The public ground truth is only
scored afterwards by shared/score.py as a smoke test.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from shared.challenge import (GAMES, SUBMISSIONS_DIR, artifacts_dir, require,  # noqa: E402
                              source_frames, write_submission)
from games import half_video_blocks  # noqa: E402  (on sys.path through shared.challenge)
from src import cues, deadball, decode, metrica, model, scoring, tracking  # noqa: E402

NAME = os.path.basename(HERE)
STAGES = ("fetch", "prepare", "train", "predict")
METRICA_DIR = (os.path.join(ROOT, "repo", "sample-data")
               if os.path.isdir(os.path.join(ROOT, "repo", "sample-data", "data"))
               else os.path.join(ROOT, "data", "external", "metrica-sample-data"))
ER01_ARTIFACTS = os.path.join(ROOT, "artifacts", "er01-weighted-ensemble")


def load_config():
    with open(os.path.join(HERE, "config.json")) as handle:
        return json.load(handle)


def path(name):
    return os.path.join(artifacts_dir(NAME), name)


def save_json(name, data):
    with open(path(name), "w") as handle:
        json.dump(data, handle, indent=2)


def load_json(name):
    with open(path(name)) as handle:
        return json.load(handle)


def need(name, stage):
    if not os.path.isfile(path(name)):
        sys.exit("{} is missing. Run first: python solutions/{}/run.py {}".format(
            os.path.relpath(path(name)), NAME, stage))


# -- fetch / prepare ---------------------------------------------------------------------------

def fetch(config):
    metrica.fetch(config["metrica_repo"], config["metrica_commit"], METRICA_DIR)
    print("Metrica sample data at {} (commit {})".format(os.path.relpath(METRICA_DIR),
                                                       config["metrica_commit"][:7]))


def _bar_ath(config):
    """BAR-ATH tracking (er01's cache if present), teams, sign, graded, Smart Tagging restarts."""
    game = config["game"]
    meta_path, raw_path, tagging_path = require(["atd_meta", "atd_raw", "smart_tagging"], game)
    n_frames = source_frames("tactical", game)
    blocks = half_video_blocks(game)
    team, names, first_sign = tracking.read_meta(meta_path)
    cache = os.path.join(ER01_ARTIFACTS, "tracking.npz")
    if not os.path.isfile(cache):
        cache = path("bar_tracking.npz")
    if os.path.isfile(cache):
        data = np.load(cache)
        players, ball = data["players"], data["ball"]
        print("BAR-ATH tracking from cache {}".format(os.path.relpath(cache)))
    else:
        print("reading {} ...".format(os.path.basename(raw_path)))
        players, ball = tracking.read_tracking(raw_path, n_frames, len(team), GAMES[game]["pitch"])
        np.savez(cache, players=players, ball=ball)
    sign = tracking.home_sign_per_frame(n_frames, blocks, first_sign)
    graded = tracking.in_halves(n_frames, blocks)
    tags = tracking.read_smart_tagging(tagging_path, names)
    restarts = np.array([x["start"] for x in tags if x["code"] == "SET PIECES"], dtype=np.int64)
    return players, ball, team, sign, graded, restarts


def _game_arrays(players, ball, team, sign, graded, config, pitch):
    raw = cues.compute_all(players, ball, team, sign, [], pitch, config)
    raw.pop("tagging")
    dead = deadball.raw_features(players, ball, pitch, config["dead_lag"], tracking.FPS)
    y_pl, keep_pl = model.pseudo_labels(players, ball, team, config["pl_max_dist"],
                                        config["pl_margin"])
    out = {"cue_" + k: v for k, v in raw.items()}
    out.update({"dead_" + k: v for k, v in dead.items()})
    out.update(y_pl=y_pl, keep_pl=keep_pl & graded, graded=graded, team=team, sign=sign)
    return out


def prepare(config):
    if not os.path.isdir(os.path.join(METRICA_DIR, "data")):
        sys.exit("Metrica data missing. Run first: python solutions/{}/run.py fetch".format(NAME))
    pitch = tuple(config["metrica_pitch"])
    bar_players, bar_ball, bar_team, bar_sign, bar_graded, restarts = _bar_ath(config)
    counts = metrica.visible_counts(bar_players, bar_team, bar_graded)
    print("BAR-ATH visible players per team: mean {:.1f} / {:.1f}".format(*counts.mean(0)))

    print("BAR-ATH cues ...")
    arrays = _game_arrays(bar_players, bar_ball, bar_team, bar_sign, bar_graded, config,
                          GAMES[config["game"]]["pitch"])
    np.savez(path("bar_ath.npz"), restarts=restarts, **arrays)

    rng = np.random.default_rng(config["seed"])
    for g in config["metrica_games"]:
        d = metrica.load_game(METRICA_DIR, g, pitch)
        graded = d["labels"] != "X"
        restart_frames = d["events"].loc[metrica.is_restart(d["events"]), "start"].to_numpy()
        variants = {"full": d["players"]}
        if config["occlusion"]:
            variants["occ"] = metrica.apply_occlusion(
                d["players"], d["team"], counts, rng,
                int(config["occlusion_block_seconds"] * tracking.FPS))
        for tag, players in variants.items():
            print("game {} ({}) cues ...".format(g, tag))
            arrays = _game_arrays(players, d["ball"], d["team"], d["sign"], graded, config, pitch)
            np.savez(path("game{}_{}.npz".format(g, tag)), labels=d["labels"],
                     restarts=restart_frames, **arrays)
        live = np.isin(d["labels"], ["H", "A"])
        print("  labels: {} graded frames, D {:.3f}, H share of live {:.3f}".format(
            int(graded.sum()), float(np.mean(d["labels"][graded] == "D")),
            float(np.mean(d["labels"][live] == "H"))))


# -- train -------------------------------------------------------------------------------------

def _load(name):
    data = dict(np.load(path(name)))
    return data


def _features(data, config):
    """Ball-free cue matrix (standardised per match), and the ball-term inputs."""
    raw = {k[4:]: v for k, v in data.items() if k.startswith("cue_")}
    X, _ = cues.feature_matrix(raw, config["cues"], data["graded"], config["smooth_frames"])
    X = np.nan_to_num(X, nan=0.0)
    return X, np.nan_to_num(raw["ball_code"]), raw["ball_gap"]


def _with_ball(X, code, gap, tau):
    return np.hstack([X, (code * np.exp(-gap / tau))[:, None]])


def _dead_matrix(data, config):
    raw = {k[5:]: v for k, v in data.items() if k.startswith("dead_")}
    return deadball.feature_matrix(raw, data["graded"], config["dead_windows"])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _dead_mask(p_dead, graded, threshold, min_run, mode, restarts, config):
    """D frames for one game: the detector (runs shorter than min_run merged away), optionally
    combined with a window around each restart. Always D outside the graded frames."""
    det = np.zeros(len(p_dead), dtype=bool)
    for start, end in decode._runs(graded):
        code = (p_dead[start:end] > threshold).astype(np.int8)
        det[start:end] = decode.merge_short_runs(code, min_run) == 1
    window = np.zeros(len(p_dead), dtype=bool)
    if mode != "det":
        before, after = config["restart_before"], config["restart_after"]
        for s in restarts:
            window[max(0, s - before):s + after] = True
    if mode == "det_and_restart":
        det &= window
    elif mode == "det_or_restart":
        det |= window
    elif mode == "restart":
        det = window
    return det | ~graded


def train(config):
    tag = "occ" if config["occlusion"] else "full"
    games = config["metrica_games"]
    data = {g: _load("game{}_{}.npz".format(g, tag)) for g in games}
    full = {g: _load("game{}_full.npz".format(g)) for g in games}
    taus = config["ball_taus"]
    feats = {g: _features(data[g], config) for g in games}
    feats_full = {g: _features(full[g], config) for g in games}
    live = {g: np.isin(data[g]["labels"], ["H", "A"]) for g in games}
    y = {g: (data[g]["labels"] == "H").astype(np.int8) for g in games}
    nob = {g: live[g] & (feats[g][2] > 0) for g in games}   # ball not tracked on this frame
    l2 = config["l2"]
    report = {"games": {}, "h_vs_a": {}, "dead": {}, "decoder": {}}

    for g in games:
        lab = data[g]["labels"]
        gr = lab != "X"
        report["games"][g] = {"graded": int(gr.sum()), "D": float(np.mean(lab[gr] == "D")),
                              "pseudo_labels": int(data[g]["keep_pl"].sum()),
                              "pseudo_label_agreement": float(np.mean(
                                  data[g]["y_pl"][data[g]["keep_pl"] & live[g]] ==
                                  y[g][data[g]["keep_pl"] & live[g]]))}
    print("pseudo-label agreement with event labels: " + ", ".join(
        "G{} {:.3f} on {} frames".format(g, r["pseudo_label_agreement"], r["pseudo_labels"])
        for g, r in report["games"].items()))

    # ---- H vs A, leave one game out ---------------------------------------------------------
    def fit(train_games, cols_fn, mask_fn, target_fn):
        Xs = np.vstack([cols_fn(t)[mask_fn(t)] for t in train_games])
        ys = np.concatenate([target_fn(t)[mask_fn(t)] for t in train_games])
        return model.fit_logistic(Xs, ys, l2)

    def evaluate(w, b, Xh, g):
        z = Xh @ w + b
        acc_live = float(np.mean((z[live[g]] > 0) == (y[g][live[g]] == 1)))
        acc_nob = float(np.mean((z[nob[g]] > 0) == (y[g][nob[g]] == 1)))
        loss = model._log_loss(z[live[g]], y[g][live[g]])
        return {"acc_live": acc_live, "acc_ball_missing": acc_nob, "log_loss": loss}

    variants = {
        "event_labels": (lambda t: feats[t][0], lambda t: live[t], lambda t: y[t]),
        "pseudo_labels": (lambda t: feats[t][0], lambda t: data[t]["keep_pl"],
                          lambda t: data[t]["y_pl"]),
        "event_labels_no_occlusion": (lambda t: feats_full[t][0], lambda t: live[t],
                                      lambda t: y[t]),
    }
    for tau in taus:
        variants["event_labels+ball_tau{}".format(tau)] = (
            lambda t, tau=tau: _with_ball(*feats[t], tau), lambda t: live[t], lambda t: y[t])
    er01_w = None
    if os.path.isfile(os.path.join(ER01_ARTIFACTS, "weights.json")):
        with open(os.path.join(ER01_ARTIFACTS, "weights.json")) as handle:
            w01 = json.load(handle)
        er01_w = (np.array([w01["weights"][w01["cues"].index(c)] for c in config["cues"]]),
                  w01["bias"], w01["ball_weight"], w01["ball_tau"])

    print("H vs A, leave-one-game-out (accuracy on all live frames / where the ball is missing):")
    for vname, (cols_fn, mask_fn, target_fn) in variants.items():
        rows = []
        for g in games:
            w, b = fit([t for t in games if t != g], cols_fn, mask_fn, target_fn)
            test_cols = feats[g][0] if not vname.startswith("event_labels+ball") else \
                _with_ball(*feats[g], int(vname.split("tau")[1]))
            rows.append(evaluate(w, b, test_cols, g))
        report["h_vs_a"][vname] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        report["h_vs_a"][vname]["per_game"] = rows
        r = report["h_vs_a"][vname]
        print("  {:<30} {:.3f} / {:.3f}   log-loss {:.3f}".format(
            vname, r["acc_live"], r["acc_ball_missing"], r["log_loss"]))
    if er01_w is not None:
        rows = []
        for g in games:
            X, code, gap = feats[g]
            rows.append(evaluate(np.append(er01_w[0], er01_w[2]), er01_w[1],
                                 _with_ball(X, code, gap, er01_w[3]), g))
        report["h_vs_a"]["er01_weights+ball"] = {k: float(np.mean([r[k] for r in rows]))
                                                 for k in rows[0]}
        r = report["h_vs_a"]["er01_weights+ball"]
        print("  {:<30} {:.3f} / {:.3f}   log-loss {:.3f}  (er01 as shipped, no tagging)".format(
            "er01_weights+ball", r["acc_live"], r["acc_ball_missing"], r["log_loss"]))
    tau = min(taus, key=lambda t: report["h_vs_a"]["event_labels+ball_tau{}".format(t)]["log_loss"])
    print("ball term: tau {} frames (lowest LOGO log-loss)".format(tau))

    # single-cue and drop-one on event labels
    single = {}
    for i, c in enumerate(config["cues"]):
        accs = []
        for g in games:
            w, b = fit([t for t in games if t != g], lambda t: feats[t][0][:, [i]],
                       lambda t: live[t], lambda t: y[t])
            accs.append(evaluate(w, b, feats[g][0][:, [i]], g)["acc_live"])
        single[c] = float(np.mean(accs))
    report["h_vs_a"]["single_cue"] = single
    print("  alone: " + ", ".join("{} {:.3f}".format(c, a) for c, a in single.items()))

    # ---- dead-ball detector, leave one game out ---------------------------------------------
    dmat = {g: _dead_matrix(data[g], config) for g in games}
    gr = {g: data[g]["labels"] != "X" for g in games}
    isd = {g: (data[g]["labels"] == "D").astype(np.int8) for g in games}
    p_dead, prob = {}, {}
    for g in games:
        others = [t for t in games if t != g]
        wd, bd = fit(others, lambda t: dmat[t], lambda t: gr[t], lambda t: isd[t])
        p_dead[g] = _sigmoid(dmat[g] @ wd + bd)
        w, b = fit(others, lambda t: _with_ball(*feats[t], tau), lambda t: live[t],
                   lambda t: y[t])
        prob[g] = _sigmoid(_with_ball(*feats[g], tau) @ w + b)
        pd_, t_ = p_dead[g][gr[g]] > 0.5, isd[g][gr[g]] == 1
        tp = np.sum(pd_ & t_)
        f1 = 2 * tp / max(1, pd_.sum() + t_.sum())
        report["dead"][g] = {"frame_f1_at_0.5": float(f1), "accuracy": float(np.mean(pd_ == t_))}
    print("dead-ball detector, LOGO frame F1 at 0.5: " + ", ".join(
        "G{} {:.3f}".format(g, r["frame_f1_at_0.5"]) for g, r in report["dead"].items()))

    # ---- decoder grid on the real metric ----------------------------------------------------
    grid = [s for s in itertools.product(config["dead_modes"], config["dead_thresholds"],
                                         config["min_dead_runs"], config["min_spells"],
                                         config["x_bands"])
            # the restart window alone ignores the detector settings: one of each is enough
            if s[0] != "restart" or (s[1] == config["dead_thresholds"][0] and
                                     s[2] == config["min_dead_runs"][0])]
    print("decoder grid: {} settings x {} games ...".format(len(grid), len(games)))
    results = []
    dead_cache = {}
    for mode, thr, mdr, ms, band in grid:
        rows = []
        for g in games:
            key = (g, mode, thr, mdr)
            if key not in dead_cache:
                dead_cache[key] = _dead_mask(p_dead[g], gr[g], thr, mdr, mode,
                                             data[g]["restarts"], config)
            labels = decode.decode(prob[g], dead_cache[key], gr[g], band, ms)
            rows.append(scoring.score(data[g]["labels"], labels))
        results.append({"dead_mode": mode, "dead_threshold": thr, "min_dead_run": mdr,
                        "min_spell": ms, "band": band,
                        "spell_f1": float(np.mean([r["spell_f1"] for r in rows])),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                        "per_game": rows})
    results.sort(key=lambda r: (-r["spell_f1"], -r["macro_f1"]))
    report["decoder"]["grid"] = results

    def best_of(mode):
        return next(r for r in results if r["dead_mode"] == mode)

    print("best per D mode (mean LOGO over {} games):".format(len(games)))
    for mode in config["dead_modes"]:
        r = best_of(mode)
        print("  {:<16} spellF1 {:.3f}  macroF1 {:.3f}  (thr {}, min D {}, min spell {}, band {})"
              "  possessions pred/gt {}".format(
                  mode, r["spell_f1"], r["macro_f1"], r["dead_threshold"], r["min_dead_run"],
                  r["min_spell"], r["band"],
                  "/".join("{}/{}".format(x["n_pred"], x["n_gt"]) for x in r["per_game"])))

    # reference points: er01 as shipped, and the oracle-D ceiling of the chosen H/A decoder
    chosen = results[0]
    refs = {}
    if er01_w is not None:
        rows = []
        for g in games:
            X, code, gap = feats[g]
            z = X @ er01_w[0] + er01_w[1] + er01_w[2] * code * np.exp(-gap / er01_w[3])
            dead = _dead_mask(p_dead[g], gr[g], 0.5, 1, "restart", data[g]["restarts"], config)
            rows.append(scoring.score(data[g]["labels"],
                                      decode.decode(_sigmoid(z), dead, gr[g], 0.0, 75)))
        refs["er01_as_shipped_s75"] = rows
    for ms in sorted({chosen["min_spell"], 75}):
        rows = []
        for g in games:
            dead = (data[g]["labels"] == "D") | ~gr[g]
            rows.append(scoring.score(data[g]["labels"],
                                      decode.decode(prob[g], dead, gr[g], 0.0, ms)))
        refs["oracle_D_s{}".format(ms)] = rows
    rows = [scoring.score(data[g]["labels"], data[g]["labels"]) for g in games]
    refs["perfect"] = rows
    report["decoder"]["references"] = {
        k: {"spell_f1": float(np.mean([r["spell_f1"] for r in v])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in v])), "per_game": v}
        for k, v in refs.items()}
    for k, v in report["decoder"]["references"].items():
        print("  {:<22} spellF1 {:.3f}  macroF1 {:.3f}".format(k, v["spell_f1"], v["macro_f1"]))

    # ---- final fit on all games -------------------------------------------------------------
    w, b = fit(games, lambda t: _with_ball(*feats[t], tau), lambda t: live[t], lambda t: y[t])
    wd, bd = fit(games, lambda t: dmat[t], lambda t: gr[t], lambda t: isd[t])
    print("final H/A weights: " + ", ".join("{} {:+.2f}".format(c, x) for c, x in
                                            zip(config["cues"] + ["ball"], w)) +
          ", bias {:+.2f}".format(b))
    save_json("model.json", {
        "cues": config["cues"], "weights": w.tolist(), "bias": b, "ball_tau": tau,
        "dead_features": deadball.column_names(config["dead_windows"]),
        "dead_weights": wd.tolist(), "dead_bias": bd,
        "decoder": {k: chosen[k] for k in ("dead_mode", "dead_threshold", "min_dead_run",
                                           "min_spell", "band")},
        "decoder_det": {k: best_of("det")[k] for k in ("dead_mode", "dead_threshold",
                                                       "min_dead_run", "min_spell", "band")},
    })
    save_json("logo.json", report)
    print("  -> {}".format(os.path.relpath(artifacts_dir(NAME))))


# -- predict -----------------------------------------------------------------------------------

def predict(config, out_dir):
    need("model.json", "train")
    need("bar_ath.npz", "prepare")
    m = load_json("model.json")
    data = _load("bar_ath.npz")
    graded = data["graded"]
    X, code, gap = _features(data, config)
    prob = _sigmoid(_with_ball(X, code, gap, m["ball_tau"]) @ np.asarray(m["weights"]) + m["bias"])
    p_dead = _sigmoid(_dead_matrix(data, config) @ np.asarray(m["dead_weights"]) + m["dead_bias"])

    settings = {"best": m["decoder"], "det": m["decoder_det"]}
    done = set()
    for label, s in settings.items():
        key = tuple(sorted(s.items()))
        if key in done:
            continue
        done.add(key)
        dead = _dead_mask(p_dead, graded, s["dead_threshold"], s["min_dead_run"], s["dead_mode"],
                          data["restarts"], config)
        labels = decode.decode(prob, dead, graded, s["band"], s["min_spell"])
        name = "{}-{}-t{:g}-s{}".format(NAME, s["dead_mode"].replace("_", ""),
                                        s["dead_threshold"], s["min_spell"])
        out = write_submission(os.path.join(out_dir, name + ".csv"), labels)
        counts = {k: int(np.sum(labels[graded] == k)) for k in "HADX"}
        print("{:<40} H {:>6}  A {:>6}  D {:>6}  X {:>6}  possessions {:>4}  -> {}".format(
            name[len(NAME) + 1:], counts["H"], counts["A"], counts["D"], counts["X"],
            decode.count_spells(labels, graded), os.path.relpath(out)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES + ("all",))
    parser.add_argument("--out", default=SUBMISSIONS_DIR, help="output folder")
    args = parser.parse_args()

    config = load_config()
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print("== {} ==".format(stage))
        if stage == "predict":
            predict(config, args.out)
        else:
            {"fetch": fetch, "prepare": prepare, "train": train}[stage](config)
    print("\nNext: python shared/score.py submissions/{}-*.csv".format(NAME))


if __name__ == "__main__":
    main()
