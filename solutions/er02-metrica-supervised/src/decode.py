"""From the combined probability p(H) to the challenge labels H / A / D / X.

D and X answer different questions, so they come from different places:

  D  dead ball. It comes from a detector, never from an uncertain score: a 50/50 frame is a frame
     we cannot read, not a frame where play has stopped. Version 1 uses the Smart Tagging restarts
     (SET PIECES), see `dead_ball_mask`.
  X  abstain. The scorer replaces every X with the previous label of the half before spellF1 is
     computed, so an X is "keep what you had". Here it is written only on uncertain frames
     (|p - 0.5| < band) whose final label already equals the previous frame's H/A label. That makes
     spellF1 identical with or without the X; the band only trades macroF1 (an X is a miss, but
     unlike a wrong label it adds no false positive) and acts as hysteresis: an uncertain frame
     never flips the possession.

Then every H/A run shorter than `min_spell` frames is merged into its neighbours, shortest first.
"""

import numpy as np

H, A, D, X = "H", "A", "D", "X"


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def dead_ball_mask(n, restarts, graded, config):
    """True on frames labelled D: around each Smart Tagging restart, [start - before, start +
    after), and everywhere outside the graded halves (never scored)."""
    dead = ~graded
    before, after = config["dead_before"], config["dead_after"]
    for start, _ in restarts:
        dead[max(0, start - before):min(n, start + after)] = True
    return dead


def merge_short_runs(codes, min_spell):
    """Merge runs shorter than `min_spell` into their neighbours, shortest first. codes is a 1-D
    int array (one segment, no D inside). A run between two runs of the same value takes that
    value; otherwise it joins the longer neighbour."""
    if min_spell <= 1 or len(codes) == 0:
        return codes
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = list(np.concatenate(([0], change)))
    values = [int(codes[s]) for s in starts]
    lengths = list(np.diff(np.concatenate((starts, [len(codes)]))))
    while len(lengths) > 1:
        i = int(np.argmin(lengths))
        if lengths[i] >= min_spell:
            break
        left = i - 1 if i > 0 else None
        right = i + 1 if i + 1 < len(lengths) else None
        if left is not None and right is not None and values[left] == values[right]:
            lengths[left] += lengths[i] + lengths[right]
            del values[i:i + 2], lengths[i:i + 2]
        else:
            target = left if right is None or (left is not None and lengths[left] >= lengths[right]) \
                else right
            lengths[target] += lengths[i]
            del values[i], lengths[i]
            # Neighbours may now share a value: fuse them.
            j = min(i, len(values) - 1)
            if 0 < j < len(values) and values[j - 1] == values[j]:
                lengths[j - 1] += lengths[j]
                del values[j], lengths[j]
    return np.repeat(values, lengths)


def decode(prob, dead, graded, band, min_spell):
    """Labels for every frame: D on `dead`, H/A elsewhere with hysteresis and run merging, X on
    uncertain frames that repeat the previous label."""
    n = len(prob)
    labels = np.full(n, D, dtype="<U1")
    uncertain = np.abs(prob - 0.5) < band
    for start, end in _runs(~dead):
        p, unc = prob[start:end], uncertain[start:end]
        code = (p > 0.5).astype(np.int8)
        if unc.any() and not unc.all():
            # Hysteresis: uncertain frames keep the last certain decision (the first certain one
            # at the head of the segment).
            idx = np.where(~unc, np.arange(len(p)), -1)
            idx = np.maximum.accumulate(idx)
            first = int(np.flatnonzero(~unc)[0])
            idx[idx < 0] = first
            code = code[idx]
        code = merge_short_runs(code, min_spell)
        seg = np.where(code == 1, H, A)
        same_as_prev = np.concatenate(([False], seg[1:] == seg[:-1]))
        seg = np.where(unc & same_as_prev, X, seg)
        labels[start:end] = seg
    return labels


def count_spells(labels, graded):
    """Possessions (maximal H or A runs, after the scorer's forward fill of X) on graded frames."""
    total = 0
    for start, end in _runs(graded):
        seg = labels[start:end].copy()
        last = None
        for i, value in enumerate(seg):
            if value == X:
                seg[i] = last if last is not None else X
            else:
                last = value
        alive = (seg == H) | (seg == A)
        change = np.concatenate(([True], seg[1:] != seg[:-1]))
        total += int(np.sum(alive & change))
    return total
