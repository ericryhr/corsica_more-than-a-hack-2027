"""spellF1 and macroF1 with the challenge's own scorer (evaluation/possession_metrics.py), for
label arrays held in memory (the Metrica games have no CSV ground truth)."""

import possession_metrics as pm  # on sys.path through run.py


def score(gt_labels, pred_labels):
    gt = pm.label_codes(gt_labels)
    pred = pm.label_codes(pred_labels)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spells = pm.spell_scores(gt, filled, runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return {"spell_f1": spells["mean_f1"], "macro_f1": macro,
            "n_gt": spells["n_gt"], "n_pred": spells["n_pred"]}
