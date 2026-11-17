# er03-windowed-viterbi

**Hypothesis.** er02 showed that a tracking-based dead-ball (D) detector is the big lever, and that
H vs A is then the bottleneck. On the `ferran` branch, fg03 and fg07 read each feature through
three time windows (before, around and after the frame), because team shape reacts about 2 s after
a change of team, and decode with a 3-state Viterbi. A scratch test of those windows on er02's own
cues raised H/A accuracy from 0.807 to 0.870 and spellF1 from 0.487 to 0.572. er03 puts this
together and tests the label proposals:
1. **One shared testbed.** Everything is trained and scored on `shared/sample.py` /
   `sample_bench.py` (canals), the same truth as every other solution.
2. **Labels for controlled possession.** These follow CHALLENGE.md's own proxies instead of the
   2 s flicker merge.
3. **A blurred ball for training.** The sample ball is rebuilt from the events, so the models train
   with the ball degraded like the ATD's (`blurred`, `harsh-blurred`).
4. **Touch features.** fg03's `touch_diff` and `touch_any` are in the models. A separately trained
   touch detector is left for later.

**Method.** Code from fg07, fg03 and er02 is loaded by path (`src/peers.py`). Nothing is copied.

- **Features.**
  - fg07's `base` features: fg02's ball, proximity, shape and context features, plus fg03's touch
    features, on coordinates turned so home attacks +x.
  - er02's cues (motion, marking, shape and height, standardised per match, plus the ball vote)
    and its dead-ball features.
  - Each feature is read through the windows [-4, -1) s, [-1, +1) s and [+1, +4) s.
- **Models.** fg03's L2 logistic owner model, P(H | alive), and dead model, P(D), are decoded by
  its Viterbi with a cost per change of team and per stoppage.
- **Variants** are a feature set crossed with a set of training labels:

  | feature set | owner model | dead model |
  |---|---|---|
  | `fg07` | fg07 `linear` lists (control) | fg07 `linear` lists |
  | `owner02` | + er02 cues | fg07 |
  | `dead02` | fg07 | + er02 dead-ball features |
  | `full` | + er02 cues | + er02 dead-ball features |

  - **`bench` labels:** shared/sample.py. The owner comes from events, D runs from stoppage to
    restart, and flickers under 2 s are merged.
  - **`controlled` labels:** a change of team counts only if the new team then has at least 2
    on-ball events or keeps the ball for at least 2 s. Otherwise the previous owner continues. A
    restart always counts. This gives 1,004 possessions (median 5.8 s) against the bench labels'
    867 (7.7 s).
- **Validation.**
  - Leave one game out (LOGO): the models that score a game never saw it, under either degradation.
  - The costs are the best mean spellF1, and the LOGO figure also picks the costs on the other two
    games.
  - Every variant is **scored against the bench truth**.
  - `src/bench.py` gives each sample game its LOGO model.

**Results** (`run.py train`, blurred and harsh-blurred, mean of 3 games):

| variant | owner accuracy | spellF1 | spellF1 LOGO | macroF1 | costs (team, dead) |
|---|---|---|---|---|---|
| `fg07-bench` (= fg07 `linear`, which reports LOGO 0.656) | 0.894 | 0.664 | 0.656 | 0.862 | 25, 100 |
| `fg07-controlled` | 0.895 | 0.668 | 0.661 | 0.862 | 25, 100 |
| **`dead02-controlled` (chosen)** | 0.895 | **0.669** | **0.665** | 0.863 | 25, 50 |
| `dead02-bench` | 0.894 | 0.665 | 0.659 | 0.862 | 25, 50 |
| `owner02-controlled` | 0.891 | 0.658 | 0.655 | 0.858 | 10, 100 |
| `owner02-bench` | 0.889 | 0.642 | 0.632 | 0.856 | 10, 100 |
| `full-bench` | 0.889 | 0.645 | 0.632 | 0.857 | 10, 50 |
| `full-bench` with er02's merge decoder, s50 / s75 | | 0.628 / 0.618 | | 0.852 | |

**Reading.**
- The control reproduces fg07 `linear` exactly (LOGO 0.656, same costs), so the comparison is fair.
- **The windows and the Viterbi are the gain.**
  - The same features with er02's decoder lose 0.017.
  - er02's cues **hurt the owner model** (−0.024 with bench labels). They are redundant with fg02's
    shape and proximity features and add columns. Its dead-ball features help by only +0.003.
- **Controlled labels:** +0.004 to +0.023 against the bench truth. They help most where the model
  is weaker (`owner02`), and they predict a few more possessions. This is within noise, but it
  costs nothing and is defensible from CHALLENGE.md rather than from the public truth's shape.
- So er03 ≈ fg07 `linear`. Its best variant is +0.009 LOGO above it, which is under the team's
  0.04 bar. **What raised spellF1 from er02's 0.49 to 0.66 is fg03/fg07's windowed model.**

## Global comparison on the shared bench

`shared/sample_bench.py`, 3 Metrica games, 867 true possessions. Rows that need training use
models that never saw the scored game (er02, er03, nc03/nc04's detector). Untrained rows use
parameters chosen on all 3 games, so they read slightly high. Bench scores read 0.04–0.09 above
the leaderboard, so compare rows with each other. Public is the 5 BAR-ATH minutes (17 possessions),
a smoke check only.

| solution / variant | default | harsh | blurred | harsh-blurred | mean | macroF1 (blurred) | public spellF1 |
|---|---|---|---|---|---|---|---|
| **er03 `dead02-controlled`** | **0.689** | **0.676** | **0.674** | **0.663** | **0.676** | 0.865 | 0.632 |
| er03 `fg07-bench` (= fg07 `linear`) | 0.671 | 0.673 | 0.673 | 0.655 | 0.668 | 0.866 | 0.563 |
| fg07 `linear-net`* | | | LOGO 0.650 (both settings) | | | | 0.613 |
| nc04 (er01 owner + nc03 D) | 0.557 | 0.571 | 0.503 | 0.506 | 0.534 | 0.824 | 0.425 (`-nc03`) |
| nc03 `s0-w125` | 0.519 | 0.525 | 0.493 | 0.484 | 0.505 | 0.833 | 0.500 |
| fg07 `rules` | 0.595 | 0.439 | 0.541 | 0.395 | 0.493 | 0.816 | 0.465 |
| er02 `det-s50` | 0.563 | 0.428 | 0.536 | 0.407 | 0.484 | 0.839 | 0.424 |
| er01 `b0-s75` (no D) | 0.345 | 0.336 | 0.323 | 0.334 | 0.334 | 0.418 | 0.453 |
| 00-baseline `w50` (= nc01 `nearest-w50`, fg07 `baseline`) | 0.352 | 0.294 | 0.321 | 0.266 | 0.308 | 0.744 | 0.326 |
| nc02 / nc01 best (`s0-w50`, `nearest-width-w50`) | 0.314 | 0.325 | 0.281 | 0.286 | 0.302 | 0.412 | 0.400 |
| fg07 `drift` (01-ball-drift tuned) | 0.310 | 0.298 | 0.298 | 0.286 | 0.298 | 0.429 | 0.469 |

\* fg07 `linear-net` and `netfuse` (LOGO 0.472) have no bench adapter. Their figures come from
`fg07 run.py report`, which uses the same truth and the blurred settings.

- **Paired bootstrap** (1,000 draws) against nc03 `s0-w125`: er03 `dead02-controlled` is
  +0.150 to +0.181 under every setting, and every interval excludes zero, so it is **BETTER** under
  the team rule.
- `fg07-bench` and `full-bench` are also BETTER than nc03.
- er03 against fg07 `linear` (+0.008 on the bench mean) is no clear difference.
- **Not on the bench:**
  - fg01–fg05 were fitted or tuned on the hand labels `labels/fg_30_*`, which RULES.md now
    forbids, so their public scores (fg03 `full-oof` 0.703, fg05 0.648–0.703) are not comparable.
  - fg06 is a degrader, not a predictor.
- er02's `harsh` drop (0.43) comes from its dead-ball detector. It was trained without stoppages
  that keep the ball visible, which `harsh` adds. nc03 and er03 train with the ball blurred and are
  robust to it.

**BAR-ATH.** `dead02-controlled` predicts H 38.3%, A 20.3% and D 41.4% of graded frames, with 244
possessions.
- The smoke check on the public 5 minutes gives 0.6323 / 0.8567 with 14 possessions for 17 true.
  It was not used to choose.
- `fg07-bench` and `full-bench` are written too, as references.

**Caveats.**
- The truth comes from events, not from Metrica's corrected-tracking flags. The bench reads
  0.04–0.09 above the leaderboard.
- Three training games. The degradation was matched to label-free statistics of the ATD. fg07
  raises the open question of whether that counts as tuning on the provided match, and it should be
  asked to the organisers.
- The costs are chosen on the same 3 games as the bench reports, so the LOGO column is the honest
  one.

**Next.**
- A dedicated touch detector (canals' `shared/touch_bench.py`) to place the boundaries. The
  windows make the owner right inside a possession, and its start is now the limit.
- A duration-aware decoder (HSMM) for the short possessions (23–28% are under 3 s).

**Reproduce.**

```
export MTAH_DATA_DIR=$PWD/repo/more-than-a-hack-2027/data
uv run --no-project python shared/sample.py                         # sample cache (kloppy for game 3)
uv run --no-project python solutions/fg07-sample-tuned/run.py prepare   # optional: er03 reuses its feature cache
uv run --no-project python solutions/er03-windowed-viterbi/run.py   # prepare ~10 min, train ~15 min, predict
uv run --no-project python -m pytest solutions/er03-windowed-viterbi/tests
uv run --no-project python shared/sample_bench.py er03-windowed-viterbi --settings default harsh blurred harsh-blurred \
    --variants dead02-controlled fg07-bench --versus nc03-dead-ball:s0-w125
uv run --no-project python shared/score.py submissions/er03-windowed-viterbi-*.csv
```
