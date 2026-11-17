"""Regenerate the submission file end to end.

    python regenerate.py

Runs, in order:
1. solutions/er03-windowed-viterbi   prepare, train, predict  (the base model, ~25 min)
2. solutions/ns03-kickoff-outside    prepare, predict         (the two laws-based rules, ~2 min)

and prints the path and SHA-256 of the final file,
submissions/ns03-kickoff-outside-combined-m0-f25.csv.

Requires the match data in data/ (or MTAH_DATA_DIR) and the sample cache from
`python shared/sample.py`. See README.md.
"""

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ER03 = ["solutions", "er03-windowed-viterbi", "run.py"]
NS03 = ["solutions", "ns03-kickoff-outside", "run.py"]
FINAL = os.path.join("submissions", "ns03-kickoff-outside-combined-m0-f25.csv")


def run(script, stages):
    for stage in stages:
        print("\n=== {} {} ===".format("/".join(script[:-1]), stage), flush=True)
        done = subprocess.run([sys.executable, os.path.join(ROOT, *script), stage])
        if done.returncode != 0:
            sys.exit("stage {} of {} failed".format(stage, "/".join(script[:-1])))


def main():
    run(ER03, ["prepare", "train", "predict"])
    run(NS03, ["prepare", "predict"])

    path = os.path.join(ROOT, FINAL)
    if not os.path.isfile(path):
        sys.exit("{} was not written".format(FINAL))
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("\nsubmission : {}".format(FINAL))
    print("sha256     : {}".format(digest))
    print("\nvalidate it with:")
    print("  python repo/more-than-a-hack-2027/evaluation/validate_submission.py {}".format(FINAL))


if __name__ == "__main__":
    main()
