"""Tables that relate the per-frame features to the labels and to the label boundaries.

Labels are the codes of possession_metrics: H=0, A=1, D=2, X=3. A boundary is a frame whose label
differs from the previous frame's, both graded. Boundaries come in three classes:

  team       H -> A or A -> H. Signed features are flipped so that + favours the new owner.
  to_dead    H or A -> D. Signed features are flipped so that + favours the team that had it.
  from_dead  D -> H or A. Signed features are flipped so that + favours the new owner.

Offsets are always event frame minus boundary frame: negative means the event comes before the
labelled boundary.
"""

import numpy as np
import pandas as pd

from . import stats
from .features import FPS, GROUPS, SIGNED, TOUCH_RADIUS, rolling_mean

H, A, D, X = 0, 1, 2, 3
CLASSES = ("team", "to_dead", "from_dead")
FEATURES = [name for names in GROUPS.values() for name in names]


def group_of(name):
    return next(group for group, names in GROUPS.items() if name in names)


def window_codes(codes, start, end):
    """Copy of codes with every frame outside [start, end] set to X."""
    out = np.full_like(codes, X)
    out[start:end + 1] = codes[start:end + 1]
    return out


def boundaries(codes):
    """DataFrame of boundaries: frame, prev, next, cls, sign (+1 H, -1 A, the team the class is
    oriented toward)."""
    prev, nxt = codes[:-1], codes[1:]
    at = np.flatnonzero((prev != nxt) & (prev != X) & (nxt != X)) + 1
    rows = []
    for frame in at:
        p, n = codes[frame - 1], codes[frame]
        if p != D and n != D:
            cls, team = "team", n
        elif n == D:
            cls, team = "to_dead", p
        else:
            cls, team = "from_dead", n
        rows.append((frame, p, n, cls, 1.0 if team == H else -1.0))
    return pd.DataFrame(rows, columns=["frame", "prev", "next", "cls", "sign"])


# --- agreement between two label sets -----------------------------------------------------------

def agreement(reference, other):
    """Frame confusion and boundary offsets of `other` against `reference`, on frames graded in
    both."""
    both = (reference != X) & (other != X)
    confusion = pd.crosstab(pd.Series(reference[both], name="reference").map("HADX".__getitem__),
                            pd.Series(other[both], name="other").map("HADX".__getitem__))
    ref_b = boundaries(np.where(both, reference, X))
    oth_b = boundaries(np.where(both, other, X))
    rows = []
    for _, b in ref_b.iterrows():
        same = oth_b[(oth_b.prev == b.prev) & (oth_b.next == b.next)]
        offset = np.nan
        if len(same):
            delta = same.frame.to_numpy() - b.frame
            offset = delta[np.argmin(np.abs(delta))]
        rows.append({"frame": b.frame, "cls": b.cls, "type": "HAD"[b.prev] + ">" + "HAD"[b.next],
                     "offset": offset})
    offsets = pd.DataFrame(rows)
    return confusion, offsets, len(ref_b), len(oth_b)


# --- univariate ---------------------------------------------------------------------------------

def univariate(features, codes):
    """One row per feature: coverage, mean per label, AUC H vs A (overall, with ball, without
    ball) and AUC D vs alive."""
    graded = codes != X
    alive = (codes == H) | (codes == A)
    ball = features["ball_seen"] > 0
    rows = []
    for name in FEATURES:
        f = features[name]
        rows.append({
            "feature": name, "group": group_of(name), "signed": name in SIGNED,
            "coverage": np.mean(~np.isnan(f[graded])),
            "mean_H": np.nanmean(f[codes == H]), "mean_A": np.nanmean(f[codes == A]),
            "mean_D": np.nanmean(f[codes == D]),
            "auc_HA": stats.auc(f[codes == H], f[codes == A]),
            "auc_HA_ball": stats.auc(f[(codes == H) & ball], f[(codes == A) & ball]),
            "auc_HA_noball": stats.auc(f[(codes == H) & ~ball], f[(codes == A) & ~ball]),
            "auc_dead": stats.auc(f[codes == D], f[alive]),
        })
    return pd.DataFrame(rows)


# --- boundary-locked averages -------------------------------------------------------------------

def locked(features, bounds, half):
    """Mean of each feature at every offset in [-half, half] around the boundaries of each class,
    oriented by the class sign for signed features. Long table: cls, feature, offset, mean, n."""
    offsets = np.arange(-half, half + 1)
    n_frames = len(features["ball_seen"])
    rows = []
    for cls in CLASSES:
        b = bounds[bounds.cls == cls]
        if b.empty:
            continue
        index = np.clip(b.frame.to_numpy()[:, None] + offsets[None, :], 0, n_frames - 1)
        sign = b.sign.to_numpy()[:, None]
        for name in FEATURES:
            values = features[name][index]
            if name in SIGNED:
                values = values * sign
            counts = np.sum(~np.isnan(values), axis=0)
            means = np.nansum(values, axis=0) / np.where(counts > 0, counts, np.nan)
            rows.append(pd.DataFrame({"cls": cls, "feature": name, "offset": offsets,
                                      "mean": means, "n": counts}))
    return pd.concat(rows, ignore_index=True)


def locked_summary(table, features, codes, level_span):
    """Per class and feature: level before (offsets up to -level_span), level after (from
    +level_span), the jump in units of the feature's std on graded frames, and the offset where
    the mean curve, smoothed over 9 frames, crosses the midpoint between the two levels in the
    direction of the jump. Of several crossings, the one closest to the boundary."""
    graded = codes != X
    rows = []
    for (cls, name), part in table.groupby(["cls", "feature"], sort=False):
        part = part.sort_values("offset")
        off = part.offset.to_numpy()
        mean = pd.Series(part["mean"].to_numpy()).rolling(9, center=True, min_periods=1).mean()
        mean = mean.to_numpy()
        before = np.nanmean(part["mean"].to_numpy()[off <= -level_span])
        after = np.nanmean(part["mean"].to_numpy()[off >= level_span])
        std = np.nanstd(features[name][graded])
        jump = (after - before) / std if std > 0 else np.nan
        crossing = np.nan
        if np.isfinite(jump) and abs(jump) > 0:
            side = (mean - (before + after) / 2) * np.sign(after - before) > 0
            up = np.flatnonzero(side[1:] & ~side[:-1]) + 1
            if len(up):
                crossing = off[up[np.argmin(np.abs(off[up]))]]
        rows.append({"cls": cls, "feature": name, "before": before, "after": after,
                     "jump_std": jump, "crossing": crossing})
    return pd.DataFrame(rows)


def locked_profile(table, names, offsets):
    """Wide table of the boundary-locked mean at a few offsets, one row per class and feature."""
    part = table[table.feature.isin(names) & table.offset.isin(offsets)]
    wide = part.pivot_table(index=["cls", "feature"], columns="offset", values="mean")
    wide.columns = ["{:+d}".format(c) for c in wide.columns]
    return wide.reset_index()


# --- event lags ---------------------------------------------------------------------------------

def onsets(mask):
    """Frames where mask turns True."""
    mask = np.asarray(mask, dtype=bool)
    return np.flatnonzero(mask[1:] & ~mask[:-1]) + 1


def nearest_offset(events, frame, reach):
    """Signed offset of the event closest to frame within reach, NaN if none."""
    if len(events) == 0:
        return np.nan
    i = np.searchsorted(events, frame)
    candidates = events[max(i - 1, 0):i + 1] - frame
    candidates = candidates[np.abs(candidates) <= reach]
    return candidates[np.argmin(np.abs(candidates))] if len(candidates) else np.nan


def touches(features, radius):
    """(frames, team) of every frame where the nearest player is within radius of the ball;
    team is +1 for H and -1 for A."""
    near = features["near_dist"] < radius
    frames = np.flatnonzero(near)
    return frames, features["near_team"][frames]


def touch_switch(frames, teams, boundary, new_sign, reach):
    """Around a team boundary, the pair (last old-team touch, first new-team touch) whose gap is
    closest to the boundary. Returns their offsets, NaN if there is none within reach."""
    lo, hi = np.searchsorted(frames, [boundary - reach, boundary + reach + 1])
    f, t = frames[lo:hi], teams[lo:hi]
    switches = np.flatnonzero((t[:-1] == -new_sign) & (t[1:] == new_sign))
    if len(switches) == 0:
        return np.nan, np.nan
    old, new = f[switches], f[switches + 1]
    distance = np.where(boundary < old, old - boundary, np.where(boundary > new, boundary - new, 0))
    best = np.argmin(distance)
    return old[best] - boundary, new[best] - boundary


def lags(features, tags, codes_of_tags, bounds, reach):
    """One row per boundary with the offsets of the events that might mark it."""
    speed_r = rolling_mean(features["ball_speed"], 0.4)
    seen = features["ball_seen"] > 0
    # A missing stretch counts when it lasts at least one second.
    gone = onsets(~seen)
    gone = gone[[np.all(~seen[g:g + FPS]) for g in gone]]
    back = onsets(seen)
    back = back[[np.all(~seen[max(b - FPS, 0):b]) for b in back]]
    out_on = onsets(features["ball_out"] > 0)
    reliable = np.isin(np.array(codes_of_tags)[tags["code"]], ["BUILD UP", "PROGRESSION", "LONG BALL"])
    set_piece = np.array(codes_of_tags)[tags["code"]] == "SET PIECES"
    goals = np.array(codes_of_tags)[tags["code"]] == "GOALS"
    sp_start, sp_end = np.sort(tags["start"][set_piece]), np.sort(tags["end"][set_piece])
    goal_start = np.sort(tags["start"][goals])
    touch_frames = {r: touches(features, r) for r in (1.0, TOUCH_RADIUS, 2.0, 3.0)}
    near_r = features["near_team_r1"]

    rows = []
    for _, b in bounds.iterrows():
        f, s = int(b.frame), b.sign
        row = {"frame": f, "cls": b.cls, "type": "HAD"[b.prev] + ">" + "HAD"[b.next]}
        if b.cls in ("team", "null_owner", "null_other"):
            for r, (tf, tt) in touch_frames.items():
                row["touch_new_r{}".format(r)] = nearest_offset(tf[tt == s], f, reach)
            last_old, first_new = touch_switch(*touch_frames[TOUCH_RADIUS], f, s, reach)
            row["switch_last_old"], row["switch_first_new"] = last_old, first_new
            oriented = near_r * s
            row["near_r1_cross"] = nearest_offset(onsets(oriented > 0), f, reach)
            starts = np.sort(tags["start"][reliable & (tags["owner"] == (0 if s > 0 else 1))])
            row["tag_start_new"] = nearest_offset(starts, f, reach)
        if b.cls == "to_dead":
            row["ball_out_onset"] = nearest_offset(out_on, f, reach)
            row["ball_gone"] = nearest_offset(gone, f, reach)
            row["slow_onset"] = nearest_offset(onsets(speed_r < 1.0), f, reach)
            row["set_piece_start"] = nearest_offset(sp_start, f, reach)
            row["goal_start"] = nearest_offset(goal_start, f, reach)
        if b.cls == "from_dead":
            row["ball_back"] = nearest_offset(back, f, reach)
            row["fast_onset"] = nearest_offset(onsets(speed_r > 3.0), f, reach)
            row["set_piece_end"] = nearest_offset(sp_end, f, reach)
            tf, tt = touch_frames[TOUCH_RADIUS]
            row["touch_new_r1.5"] = nearest_offset(tf[tt == s], f, reach)
        rows.append(row)
    return pd.DataFrame(rows)


def null_boundaries(codes, bounds, n, margin, seed):
    """Random alive frames at least `margin` frames from any boundary, as fake team boundaries.
    Two copies: sign = the team that has the ball there ("null_owner") and the other team
    ("null_other"). They give the chance level of the team-boundary events."""
    alive = (codes == H) | (codes == A)
    far = ~near_boundary(bounds, list(CLASSES), len(codes), margin)
    candidates = np.flatnonzero(alive & far)
    frames = np.sort(np.random.default_rng(seed).choice(candidates, size=min(n, len(candidates)),
                                                        replace=False))
    owner = np.where(codes[frames] == H, 1.0, -1.0)
    rows = [pd.DataFrame({"frame": frames, "prev": codes[frames], "next": codes[frames],
                          "cls": cls, "sign": sign})
            for cls, sign in (("null_owner", owner), ("null_other", -owner))]
    return pd.concat(rows, ignore_index=True)


def lag_summary(table):
    rows = []
    for cls, part in table.groupby("cls", sort=False):
        for column in part.columns.drop(["frame", "cls", "type"]):
            values = part[column].to_numpy(dtype=float)
            found = values[~np.isnan(values)]
            if part[column].isna().all():
                continue
            rows.append({
                "cls": cls, "event": column, "n": len(values), "found": len(found),
                "median": np.median(found) if len(found) else np.nan,
                "p25": np.percentile(found, 25) if len(found) else np.nan,
                "p75": np.percentile(found, 75) if len(found) else np.nan,
                # Shares over every boundary: an event not found counts as a miss.
                "exact_0": np.sum(found == 0) / len(values),
                "within_12": np.sum(np.abs(found) <= 12) / len(values),
                "within_25": np.sum(np.abs(found) <= 25) / len(values),
            })
    return pd.DataFrame(rows)


# --- boundary detection -------------------------------------------------------------------------

def deltas(features, seconds):
    """|change| of each feature's 0.5 s mean between `seconds` before and after each frame."""
    k = int(seconds * FPS)
    out = {}
    for name in FEATURES:
        smooth = rolling_mean(features[name], 0.5)
        d = np.full_like(smooth, np.nan)
        d[k:-k] = np.abs(smooth[2 * k:] - smooth[:-2 * k])
        out[name] = d
    return out


def near_boundary(bounds, cls, n_frames, radius):
    mask = np.zeros(n_frames, dtype=bool)
    for frame in bounds.frame[bounds.cls.isin(cls)]:
        mask[max(frame - radius, 0):frame + radius + 1] = True
    return mask


def boundary_auc(features, change, codes, bounds, radius):
    """AUC of each feature level and each feature change for frames within `radius` of a
    boundary against the other graded frames."""
    graded = codes != X
    n = len(codes)
    targets = {"team": near_boundary(bounds, ["team"], n, radius),
               "dead": near_boundary(bounds, ["to_dead", "from_dead"], n, radius)}
    rows = []
    for name in FEATURES:
        row = {"feature": name, "group": group_of(name)}
        for target, mask in targets.items():
            pos, neg = graded & mask, graded & ~mask
            level = features[name] if name not in SIGNED else np.abs(features[name])
            row["level_" + target] = stats.auc(level[pos], level[neg])
            row["change_" + target] = stats.auc(change[name][pos], change[name][neg])
        rows.append(row)
    return pd.DataFrame(rows), {key: int((graded & mask).sum()) for key, mask in targets.items()}


# --- linear models ------------------------------------------------------------------------------

def design(features, names, frames):
    return np.column_stack([features[name][frames] for name in names]).astype(np.float64)


def fit_eval(features, names, train, test, y_train, y_test, l2):
    scaler = stats.Standardizer().fit(design(features, names, train))
    w = stats.logistic(scaler.transform(design(features, names, train)), y_train, l2=l2)
    p = stats.predict_proba(w, scaler.transform(design(features, names, test)))
    return stats.auc(p[y_test == 1], p[y_test == 0]), np.mean((p > 0.5) == (y_test == 1)), w


def models(features, change, datasets, splits, l2, radius):
    """Logistic regressions on three tasks and several feature sets. Each split trains on one
    dataset and tests on another. Returns (scores, coefficients of the full model fitted on the
    largest dataset)."""
    sets = {group: names for group, names in GROUPS.items()}
    sets["ball+proximity"] = GROUPS["ball"] + GROUPS["proximity"]
    sets["no tags"] = [n for n in FEATURES if n not in GROUPS["tags"]]
    sets["all"] = FEATURES
    sets["change all"] = FEATURES  # levels plus their changes, for the boundary task only

    def task(codes, name):
        if name == "owner":
            frames = np.flatnonzero((codes == H) | (codes == A))
            return frames, (codes[frames] == H).astype(float)
        if name == "dead":
            frames = np.flatnonzero(codes != X)
            return frames, (codes[frames] == D).astype(float)
        bounds = boundaries(codes)
        mask = near_boundary(bounds, ["team"], len(codes), radius)
        frames = np.flatnonzero((codes == H) | (codes == A))
        return frames, mask[frames].astype(float)

    feats = dict(features)
    feats.update({"chg_" + k: v for k, v in change.items()})
    rows, coefs = [], {}
    for task_name in ("owner", "dead", "boundary"):
        for set_name, names in sets.items():
            if set_name == "change all":
                if task_name != "boundary":
                    continue
                names = FEATURES + ["chg_" + n for n in FEATURES]
            for train_name, test_name in splits:
                tr, ytr = task(datasets[train_name], task_name)
                te, yte = task(datasets[test_name], task_name)
                score, acc, _ = fit_eval(feats, names, tr, te, ytr, yte, l2)
                rows.append({"task": task_name, "features": set_name, "train": train_name,
                             "test": test_name, "auc": score, "accuracy": acc,
                             "base_rate": yte.mean()})
        names = FEATURES + (["chg_" + n for n in FEATURES] if task_name == "boundary" else [])
        largest = max(datasets, key=lambda k: np.sum(datasets[k] != X))
        frames, y = task(datasets[largest], task_name)
        scaler = stats.Standardizer().fit(design(feats, names, frames))
        w = stats.logistic(scaler.transform(design(feats, names, frames)), y, l2=l2)
        coefs[task_name] = pd.DataFrame({"feature": names, "coef": w[1:]}).sort_values(
            "coef", key=np.abs, ascending=False)
    return pd.DataFrame(rows), coefs


# --- unsupervised -------------------------------------------------------------------------------

def unsupervised(features, codes, n_components, ks, stride, seed):
    """PCA and k-means on the standardised features of graded frames, every `stride`-th frame.
    Returns (pca table, loadings, cluster table)."""
    frames = np.flatnonzero(codes != X)[::stride]
    x = stats.Standardizer().fit(design(features, FEATURES, frames))
    z = x.transform(design(features, FEATURES, frames))
    labels = codes[frames]
    components, ratio = stats.pca(z, n_components)
    scores = (z - z.mean(axis=0)) @ components.T
    pca_rows, loadings = [], []
    for i in range(n_components):
        s = scores[:, i]
        top = np.argsort(-np.abs(components[i]))[:6]
        pca_rows.append({"pc": i + 1, "variance": ratio[i],
                         "auc_HA": stats.auc(s[labels == H], s[labels == A]),
                         "auc_dead": stats.auc(s[labels == D], s[labels != D]),
                         "top": ", ".join("{}{:+.2f}".format(FEATURES[j], components[i, j])
                                          for j in top)})
        loadings.append(components[i])
    cluster_rows = []
    for k in ks:
        assign = stats.kmeans(z, k, seed)
        alive = labels != D
        cluster_rows.append({"k": k, "nmi_HAD": stats.nmi(assign, labels),
                             "nmi_dead": stats.nmi(assign, labels == D),
                             "nmi_owner_alive": stats.nmi(assign[alive], labels[alive])})
        if k == max(ks):
            composition = []
            for j in range(k):
                member = assign == j
                centre = z[member].mean(axis=0)
                top = np.argsort(-np.abs(centre))[:4]
                composition.append({
                    "cluster": j, "share": member.mean(),
                    "H": np.mean(labels[member] == H), "A": np.mean(labels[member] == A),
                    "D": np.mean(labels[member] == D),
                    "profile": ", ".join("{}{:+.1f}".format(FEATURES[i], centre[i]) for i in top)})
    return (pd.DataFrame(pca_rows), pd.DataFrame(cluster_rows),
            pd.DataFrame(composition).sort_values("share", ascending=False))
