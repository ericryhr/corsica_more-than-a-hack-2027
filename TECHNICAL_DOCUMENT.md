# Possession labelling with a supervised base model and laws-based corrections

Technical document — More than a Hack 2027, team **\<TEAM_NAME\>**

## 1. Approach

We label every frame of the provided match with the team in possession (`H`/`A`), dead ball
(`D`), or nothing (`X`). The submission combines:

1. **A supervised base model** (`er03-windowed-viterbi`). The tactical tracking is read through
   three time windows around each frame — [-4,-1) s, [-1,+1) s, [+1,+4) s — because team shape
   reacts about 2 s after a change of possession. Per frame we compute geometric features: ball
   position and speed, distances of the nearest players to the ball, team shape (width, depth,
   centroid distances), off-ball context, and ball-touch indicators. Two L2 logistic regressions
   estimate P(home owns | alive) and P(dead ball); a 3-state Viterbi decode with a fixed cost per
   change of team (25) and per stoppage (50) produces consistent spells.
2. **Two laws-based corrections** (`ns03-kickoff-outside`) on top of the base output:
   - **Kickoff serving rule.** At a kickoff only the serving team may have players in the centre
     circle, and the ball is struck from the centre spot after a stoppage. When the base model
     transitions from dead to live near the centre spot (after ≥ 20 s of dead ball; near the start
     of the video the requirement relaxes, because the opening kickoff follows the pre-match
     assembly, not 20 s of play), the rule identifies the serving team as the team whose player is
     closest to the ball — or, if the ball is never seen near a player, the team whose player is
     closest to the centre spot — forces `D` on premature pre-strike labels, and gives the serving
     team the first 1.4 s after the strike.
   - **Out-of-bounds rule.** A ball tracked outside the exact pitch rectangle (105.3 × 68.0 m,
     margin 0, as the Laws define the field of play) for ≥ 1 s (25 frames at 25 fps) is out of
     play: those frames become `D`.

We chose this combination over end-to-end learning because our validation showed the marginal
gains were not where bigger models look: owner accuracy saturates around 0.90 with any model
family we tried, and the errors concentrate at possession boundaries (see §5).

## 2. What we tried and dropped

- **Boosted trees for the two base models** (HistGradientBoosting on the same features): +0.04
  spellF1 on our sample testbed, but the gain came almost entirely from the dead-ball model,
  which called live play through true stoppages on the provided match — it scored *worse* there
  (public smoke check 0.58 vs 0.63) while keeping the sample gain. We kept the linear models.
- **A passing-network feature**: a per-frame passing network between shape clusters, fused into
  the owner model. No gain over the geometry features on the testbed.
- **Team-width heuristics** for possession (the wider team is less likely to have the ball):
  neutral to negative on the testbed.
- **A long-ball filter** (block possession flips during a pass unless an opponent reaches the
  ball or it turns): our base model predicts too *few* possessions — its errors are missed
  switches, so a filter that removes switches has nothing to fix. Neutral on the testbed.
- **Kickoff rule parameters swept beyond the priors**: kept at the prior values (§4).

## 3. Training data

All training and validation labels come from **Metrica's three public sample games**
(github.com/metrica-sports/sample-data, ~280 minutes, 867 possessions):

- Labels are rebuilt from the games' published event data by an algorithm
  (`shared/sample.py`): possession events (PASS, CARRY, SET PIECE, RECOVERY, SHOT, BALL LOST,
  BALL OUT, FAULT RECEIVED) define owner changes; stoppages define `D`; flickers shorter than 2 s
  are merged into the previous spell. No person decided any label.
- The sample tracking is degraded to match the provided ATD's **label-free** statistics: share
  and length of missing-ball gaps, players visible per frame, and the ball's blur/imprecision/lead
  profile measured on the provided data. Two degradation levels are used (a calibrated one and a
  harsher one) because the dead-ball visibility during stoppages cannot be pinned down from
  statistics alone; models that must be robust to both are preferred.
- **No hand labels of the provided match were made, and none were used** — not for training, not
  for validation, not for tuning. The public 5-minute ground truth shipped with the challenge was
  read twice as a pipeline correctness check (file well-formed, labels aligned), per EVALUATION.md.

## 4. The possession rules we implemented

- **Loose balls / ownership within a spell:** inside live play the owner is whatever the decoded
  state says; the Viterbi's per-change cost is what keeps 1–2 s flickers from splitting spells.
- **Restarts:** every transition out of a stoppage is free (no change-of-team cost), so kickoffs,
  throw-ins and goal kicks always start a new spell.
- **Kickoff serving rule and out-of-bounds rule:** as §1.2. Every constant is either from the
  Laws of the Game (pitch rectangle with margin 0; only the serving team inside the centre
  circle; kickoff order of play) or a fixed prior set before looking at any score: 20 s minimum
  prior dead time, 1.4 s serving-team hold, 1.5 m centre-spot verification radius.
- **Robustness of the constants.** None of them was tuned on the provided match:
  - The opening-window constant (the only frame-relative value) was swept from 3 to 20 minutes of
    video time: the produced submission file is **byte-identical for every value in that range**.
    Its role is structural (the opening kickoff follows the pre-match period), not informational.
  - On the three sample games, sweeping the rule parameters across
    [\<SWEEP_RESULTS\>] leaves spellF1 within ±\<X\> of the base model — the rules are
    insensitive to their constants and never hurt where the base model already obeys the laws.
  - The rule changed 2 frames at 2 of the 4 kickoffs it detected in the provided match.

## 5. Validation

**Metrica sample testbed** (3 games, 867 possessions, four degradation settings; models scored
leave-one-game-out, mean spellF1):

| pipeline | default | harsh | blurred | harsh-blurred | mean |
|---|---|---|---|---|---|
| base model (`dead02-controlled`) | 0.689 | 0.676 | 0.674 | 0.663 | 0.676 |
| + kickoff rule | 0.689 | 0.676 | 0.674 | 0.663 | 0.676 |
| + kickoff and out-of-bounds (submitted) | 0.687 | 0.674 | 0.667 | 0.658 | 0.671 |

The rules are neutral on the sample games because the sample's event-derived labels already
respect the laws — there is nothing for them to correct. They were kept because they never hurt
there, and because on the provided match the base model demonstrably violates both laws at
kickoffs and after the ball leaves the pitch. Paired bootstrap (1,000 draws) of the base model
against simpler pipelines (nearest-player baselines, ball-drift heuristics, rule systems) shows
+0.15 to +0.37 under every setting with intervals excluding zero.

**Public 5 minutes** (correctness check only): 0.6667 spellF1 / 0.8620 macroF1
(base model alone: 0.6323 / 0.8567). We report it because the organisers publish it, not because
we selected on it: no variant choice in the final pipeline was made on this window — the rules
are neutral on our validation data, so the selection criterion ("never hurts") does not involve
it.

**Where it fails.** An error breakdown of the base model on the sample games: ~7% of live frames
carry the wrong team; ~80% of those sit within 3 s of a true change of possession (the model is
blind to the touch event itself); possessions shorter than 3 s are found only 15% of the time
(over 6 s: 84%). With the true dead-ball labels given, spellF1 rises from 0.67 to 0.74, so
stoppage detection is the second lever. The obvious next step is a touch detector to place
possession boundaries.

## 6. Reproduction

The repository (link in the submission form) regenerates the file with:

```
git submodule update --init          # challenge repo + Metrica sample data (both public)
pip install -r requirements.txt      # numpy, pandas, scikit-learn, kloppy (pinned)
mkdir data && cp -r <bundle>/broadcast <bundle>/tactical data/   # match data, not in the repo
python shared/sample.py              # cache the sample games (~5 min, once)
python regenerate.py                 # base model + rules → submissions/ns03-...-combined-m0-f25.csv
python repo/more-than-a-hack-2027/evaluation/validate_submission.py submissions/ns03-kickoff-outside-combined-m0-f25.csv
```

`regenerate.py` is ~30 min end to end on one machine and prints the file's SHA-256. The staged
run reproduced the submitted file byte-identically from a fresh clone.

## 7. Assets

- Challenge bundle: tactical ATD + XML (used), broadcast video and homography (not used).
- Metrica sample data: the 3 public games (training/validation labels and tracking).
- Software: numpy, pandas, scikit-learn, kloppy. No external models, APIs or commercial services.
