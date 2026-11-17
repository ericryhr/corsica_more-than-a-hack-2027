# Possession labelling for More than a Hack 2027

Label every frame of the provided match (BAR-ATH) with the team in possession (`H`/`A`), a dead
ball (`D`), or nothing (`X`). The presented submission is **ns03-kickoff-outside**: a supervised
base model improved by two corrections that follow directly from the Laws of the Game.

## Approach in one paragraph

The base model (er03-windowed-viterbi) reads the tactical tracking through three time windows
([-4,-1) s, [-1,+1) s, [+1,+4) s) around each frame, computes geometric features of ball, proximity,
team shape and touch events, and fits two logistic regressions — the probability the home team
owns the ball, and the probability play is dead. A 3-state Viterbi decode with a cost per change of
team and per stoppage turns the per-frame probabilities into consistent spells. Both models train
exclusively on Metrica's three public sample games, whose labels are rebuilt from their published
event data by an algorithm, with the tracking degraded to match the provided ATD's measured
statistics. On top of that, ns03 enforces two rules the base model can violate: at a kickoff only
the serving team may hold the ball (identified by who stands in the centre circle), and a ball
tracked outside the pitch for at least a second is out of play.

## Repository layout

```
repo/more-than-a-hack-2027/   the challenge repository, a git submodule (public, MIT)
repo/sample-data/             Metrica's public sample games, a git submodule (training + validation data)
shared/                       access to the challenge data and code, sample-game caching, bench and scorer
solutions/er03-windowed-viterbi/   the base model (trains on the sample games)
solutions/ns03-kickoff-outside/    the kickoff and out-of-bounds rules (the presented file)
solutions/fg01, fg02, fg03, fg07, er02, 00-baseline, 01-ball-drift/
                                   vendored code the base model reuses through src/peers.py
```

## Requirements

Python 3.11 and:

```
pip install -r requirements.txt
```

The evaluation itself needs only numpy and pandas; scikit-learn trains the base model and kloppy
parses one sample game. Versions are pinned because the deliverable is a byte-identical file.

## Reproduce the submission

1. Fetch the two public submodules and put the match data in `data/` (the bundle's `broadcast/`
   and `tactical/` folders; or point `MTAH_DATA_DIR` at them). The match data is **not** in this
   repository and must not be committed to it.

   ```
   git submodule update --init
   mkdir data
   # copy the bundle's broadcast/ and tactical/ into data/
   ```

2. Cache the sample games once (downloads nothing; reads `repo/sample-data`):

   ```
   python shared/sample.py
   ```

3. Regenerate the file end to end (~30 min on one machine; the training step fits the logistic
   models on the three sample games under two degradations):

   ```
   python regenerate.py
   ```

   This runs `solutions/er03-windowed-viterbi` (prepare, train, predict) and then
   `solutions/ns03-kickoff-outside` (prepare, predict), and prints the path of
   `submissions/ns03-kickoff-outside-combined-m0-f25.csv` with its SHA-256.

4. Validate the file with the organisers' checker:

   ```
   python repo/more-than-a-hack-2027/evaluation/validate_submission.py submissions/ns03-kickoff-outside-combined-m0-f25.csv
   ```

## Validation

- **Metrica sample bench** (`python shared/sample_bench.py er03-windowed-viterbi`, then
  `... nc05-style rules`): 3 games, 867 possessions, four degradations, leave-one-game-out. The
  base model's mean spellF1 is 0.676; the two rules are neutral there (±0.005) because the
  sample's event-derived labels already respect the laws — they were accepted because they never
  hurt and correct violations the base model shows on the provided match.
- **Public 5 minutes** (`python shared/score.py submissions/ns03-kickoff-outside-combined-m0-f25.csv`):
  0.6667 spellF1 / 0.8620 macroF1. Used only as a correctness check of the pipeline, per the
  challenge's EVALUATION.md — not for tuning, which is why every rule constant comes from the
  Laws of the Game or a fixed prior in `config.json`.

## Data provenance (rule compliance)

- No hand labels of the provided match exist in this repository, and none were used to train,
  tune or validate anything presented here.
- Training and validation labels come from Metrica's three public sample games
  (`repo/sample-data`), rebuilt from their event data by `shared/sample.py` (possession events
  become owner changes, stoppages become `D`, flickers under 2 s are merged).
- Degradation parameters of the sample tracking were calibrated to label-free statistics of the
  provided ATD (share and length of missing-ball gaps, players per frame, ball blur profile).
- The public 5-minute ground truth shipped with the challenge was read for pipeline checks only.

## Limitations

- The base model predicts ~250 possessions against an expected ~280 for a full match (its Viterbi
  pays for every change of team), so short possessions are under-detected.
- The start of a possession is its weakest point: ~80% of owner errors sit within 3 s of a true
  change of team. A touch detector is the obvious next step.
- The kickoff rule acts on 4 restarts in the match; its contribution is real but small by design.
