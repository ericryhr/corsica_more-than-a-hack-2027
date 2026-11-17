# ns03-kickoff-outside

Two laws-based corrections applied on top of the base model
(`er03-windowed-viterbi`, `dead02-controlled` variant). Every parameter is either a constant of
the Laws of the Game (pitch size, centre circle, kickoff order of play) or a fixed prior listed in
`config.json`; none is fitted to the provided match. The rules were additionally checked for
neutrality on Metrica's three public sample games (they change no score there, see
`shared/sample_bench.py` and the top-level README), so they only act where the base model
violates the laws.

## 1. Kickoff serving rule

At a kickoff, only the serving team may have players inside the centre circle, and the ball is
struck from the centre spot after a stoppage. The base model sometimes labels the pre-strike
gather as live possession and sometimes gives the first seconds after the strike to the wrong
team. The rule:

1. **Trigger.** The base model transitions from `D`/`X` to `H`/`A` after at least
   `min_dead_frames` (20 s) of dead ball; within the first `opening_window_frames` of the video the
   requirement is relaxed, because the opening kickoff follows the pre-match period rather than
   20 s of dead ball. The window is a structural prior about any match video: the produced file is
   identical for any value in [3 min, 20 min] (verified frame by frame).
2. **Verification.** The ball is seen within `center_radius_m` (1.5 m) of the centre spot in a
   window around the transition.
3. **Serving team.** The team whose player stands closest to the ball in that window; if the ball
   is never seen near a player, the team whose player stands closest to the centre spot (by the
   Laws, only the serving team can be in the circle). If no player of either team is visible near
   either, the rule abstains and the base label stands.
4. **Pre-strike.** Premature live labels while the ball waits at the centre spot become `D`.
5. **Post-strike.** The serving team holds possession for at least `kickoff_hold_frames`
   (1.4 s), covering the backward pass into open space that proximity models misattribute.

## 2. Out-of-bounds rule

A ball tracked outside the pitch (`105.3 x 68.0 m`, margin 0 = the exact touchline, as the Laws
define the field of play) for at least `outside_min_frames` (1 s at 25 fps) is out of play: those
frames become `D` instead of a nearest-player possession.

## Results

Validation on Metrica's three public sample games (867 possessions, `shared/sample_bench.py`,
leave-one-game-out where training applies): both rules are neutral there (kickoff ±0.000, outside
−0.002 to −0.007 spellF1 against the base model) — the sample's labels already respect the laws,
so the rules have nothing to correct. On the provided match the rules change 2 frames at 2 of the
4 detected kickoffs. The public 5-minute window (a correctness check only, not tuning evidence)
reads 0.6667 spellF1 / 0.8620 macroF1 against the base model's 0.6323 / 0.8567.

## Reproduce

```
python solutions/ns03-kickoff-outside/run.py prepare    # reads the tactical ATD
python solutions/ns03-kickoff-outside/run.py predict    # writes submissions/ns03-kickoff-outside-combined-m0-f25.csv
```

`predict` needs the base submission from er03 (see the top-level README for the full chain).
Parameters are in `config.json`; the `min_dead_frames`, `center_radius_m` and
`kickoff_hold_frames` CLI overrides exist for experimentation only, the submitted file uses the
config values.
