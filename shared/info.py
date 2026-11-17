"""Read, write and validate `info.json`, the file every solution must have.

    solutions/NN-name/info.json
    {
      "name": "NN-name",
      "description": "What it does, in one or two sentences.",
      "results": [
        {"submission": "NN-name-v1", "gt": "public", "window": "all",
         "spell_f1_mean": 0.4242, "macro_f1": 0.5812, "n_pred_spells": 16, "n_gt_spells": 17,
         "date": "2026-10-21"}
      ]
    }

`name` and `description` are written by a person (or by `shared/new.py` when the solution is
created). `name` is free text and does not have to match the folder. `results` is touched only by
`shared/score.py`, and only in the `info.json` of the solution the submission belongs to. Each
solution owns its file, so two solutions never write the same one.
"""

import json
import os
import tempfile

PLACEHOLDER = "TODO"
RESULT_KEYS = ("submission", "gt", "window", "spell_f1_mean", "macro_f1")


def info_path(solution_dir):
    return os.path.join(solution_dir, "info.json")


def read(solution_dir):
    with open(info_path(solution_dir)) as handle:
        return json.load(handle)


def write(solution_dir, info):
    """Atomic write: the old file or the new one is left, never a half-written one."""
    fd, tmp = tempfile.mkstemp(dir=solution_dir, prefix=".info-", suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, info_path(solution_dir))


def problems(solution_dir):
    """Reasons why this folder's info.json is not valid. Empty if it is."""
    if not os.path.isfile(info_path(solution_dir)):
        return ["info.json is missing"]
    try:
        info = read(solution_dir)
    except ValueError as error:
        return ["info.json is not valid JSON: {}".format(error)]

    out = []
    if not (info.get("name") or "").strip():
        out.append("name is empty")
    description = (info.get("description") or "").strip()
    if not description or description == PLACEHOLDER:
        out.append("description is empty or still '{}'".format(PLACEHOLDER))
    if not isinstance(info.get("results"), list):
        out.append("results must be a list")
    else:
        for index, row in enumerate(info["results"]):
            missing = [key for key in RESULT_KEYS if key not in row]
            if missing:
                out.append("results[{}] is missing {}".format(index, ", ".join(missing)))
    return out


def upsert_result(solution_dir, row):
    """Add or replace the row with the same (submission, gt, window) key."""
    info = read(solution_dir)
    key = (row["submission"], row["gt"], row["window"])
    rows = [r for r in info["results"] if (r["submission"], r["gt"], r["window"]) != key]
    rows.append(row)
    info["results"] = sorted(rows, key=lambda r: (r["gt"], r["window"], r["submission"]))
    write(solution_dir, info)
