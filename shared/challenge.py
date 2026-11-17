"""Single access point to the challenge repo (games.py, evaluation/, starter/).

Solutions do not touch `sys.path` or build challenge paths on their own. They import from here:

    from shared.challenge import GAME, GAMES, require, source_frames, pm, write_submission

The challenge repo is looked up in `repo/more-than-a-hack-2027` (a git submodule). Set
`CORSICA_CHALLENGE_DIR` to point at another copy. The match data lives outside the submodule, in
`data/` at the root of this repo. `MTAH_DATA_DIR` overrides it, and is what `games.py` reads.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
CHALLENGE_DIR = os.environ.get("CORSICA_CHALLENGE_DIR") or os.path.join(
    ROOT, "repo", "more-than-a-hack-2027")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")
SOLUTIONS_DIR = os.path.join(ROOT, "solutions")
PUBLIC_GT = os.path.join(CHALLENGE_DIR, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

for path in (CHALLENGE_DIR, os.path.join(CHALLENGE_DIR, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

from games import GAME, GAMES, halves, require, source_frames  # noqa: E402,F401
import possession_metrics as pm  # noqa: E402,F401


def write_submission(path, labels):
    """`frame,label` CSV with one row per video frame. Returns the path written."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def artifacts_dir(solution):
    """Cache folder of a solution, created if missing."""
    path = os.path.join(ARTIFACTS_DIR, solution)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
