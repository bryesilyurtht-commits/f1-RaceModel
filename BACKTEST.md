# Backtest — v2.6

What the model's predictions are worth, measured rather than asserted.

This is a progress report, not a verdict. The measuring instrument is built and
tested, the baselines it has to beat are measured on a defined sample, and the
model itself is **not yet scored** — one input stands in the way, and the last
section says which and why.

## The question

The README describes a model. It does not say whether the model forecasts races
well, and until v2.6 nothing in the project did. 271 tests check that the code
does what it was written to do; none of them check that what it was written to
do resembles a Grand Prix.

## What is fixed before anything is measured

**The cutoff.** Qualifying is over, the race has not started. This matches what
the model already needs — it reads a pole time and a grid — and it is the only
moment at which the model has a well-defined input. It is not a claim about
predicting before qualifying, which the model cannot do.

**The sample.** 52 races, 2023 round 19 through 2025 round 24. The rules that
chose them were written before any score was computed, and they run off the
archive alone, so no race can be dropped for scoring badly:

| | races | why |
|---|---|---|
| included | 52 | 18-inch era, 40+ same-era races of history behind it |
| excluded | 80 | 2018–2021: the 13-inch tyre, a regime the curve does not cross |
| excluded | 40 | not enough same-era history yet |
| **archive** | **172** | |

The full table, per race, is `output/backtest_universe.csv`.

**Ground truth.** Everyone who took the start, retirements included. A model
that predicts a whole finishing order is answerable for the whole order;
scoring only the finishers would grade it on the easy half of its own claim. A
car that started and was not classified is placed behind every car that was.

"Points" is position ≤ 10 — checked, not assumed. Across all 3438 rows of the
archive, scoring a point and finishing in the top ten are the same event, with
no exceptions in any of the eight seasons.

## The baselines

Two, because they answer different questions and neither answers both.

**Grid order** — the starting grid, unchanged, as the predicted finish. The
reference for position error. It has no probabilities and is not given any: a
grid slot is not a claim about how often something happens.

**Grid history** — what a car starting in this slot has historically done, from
races before the target only, shrunk toward the field-wide rate so that a slot
with four observations cannot assert a 67% win rate. This is the reference for
*probability* quality, which grid order cannot be.

### Results, 52 races

| | MAE (expected) | MAE (ranked) | Brier win | Brier podium | Brier points |
|---|---|---|---|---|---|
| grid order | **3.141** | 3.145 | — | — | — |
| grid history | 3.348 | 3.172 | 0.032 | 0.067 | 0.161 |

Grid order MAE, 95% interval over races: **[2.867, 3.421]**. Per-race scores
are in `output/backtest_baseline_scores.csv`; per-race MAE runs from 0.90 to
5.95, so a single race says very little.

Two position errors are reported because they are different measurements. The
*expected* error scores the mean of a distribution — a number a driver whose
race is bimodal will almost never finish in. The *ranked* error scores the
single order those means imply. Quoting whichever came out lower is the easiest
way to flatter a model, so both are always shown.

**On the 3.45 the status document reports.** It is not reused here. Measured on
this sample, with this ground-truth rule, grid order scores 3.14. Whether that
disagrees with 3.45 or merely describes a different set of races cannot be
settled without the sample the 3.45 came from, so the two are not compared —
3.14 is what applies to the 52 races scored here, and nothing more.

### Are the grid-history probabilities calibrated?

Roughly, and it is worth seeing what "roughly" costs, because the model will be
held to the same table.

| band | n | predicted | observed |
|---|---|---|---|
| **win** | | | |
| 0–5% | 882 | 0.016 | 0.006 |
| 30–50% | 52 | 0.431 | 0.558 |
| **points** | | | |
| 20–30% | 260 | 0.245 | 0.177 |
| 30–50% | 158 | 0.426 | 0.259 |
| 70–100% | 306 | 0.792 | 0.902 |

It is under-confident at the top and over-confident in the middle. The full
tables, thin bands marked, come out of `backtest.pooled_calibration`.

## Uncertainty

Two kinds, and more simulations only fixes one.

**Monte Carlo error** comes from running 10,000 races instead of infinitely
many. It shrinks with more runs.

**Performance uncertainty** comes from having watched 52 races. It does not
shrink with more runs, and it is the one that decides whether a difference
means anything. Every interval here resamples whole races, never drivers —
twenty drivers in one race share a safety car and a shower, and resampling them
individually would report an interval several times too narrow. There is a test
that fails if that is ever done.

On this sample, the two baselines differ by +0.206 [+0.037, +0.377] on expected
error, and by +0.027 [−0.025, +0.077] on ranked error — the second interval
covers zero, so on ranked error the two are indistinguishable here.

## Leakage: what was actually done about it

The walk-forward rule lives in exactly one function, `history_before`, so there
is one line to audit rather than one per feature. Three tests assert it never
returns the target race or anything after it.

The inputs a past race is predicted from are assembled into their own
directory, and the model is pointed at that directory. This is deliberate: a
date filter that is missing from one table fails silently — the prediction
still appears, slightly too good, with nothing to show for it. A file that is
not written cannot be read, so an unfiltered table is a table the model does
not get, and the result is a named fallback rather than a quiet improvement.

**Historical qualifying had to be downloaded.** The archive held race laps and
results for 2018–2025 but qualifying for 2026 alone, and the model's pace is a
delta to the pole time. The two cheap ways out were both rejected: using the
race's own fastest lap reads the race being predicted, and dropping the
reference compares Monaco against Monza. 51 of the 52 sessions are now in
`data/quali_history.csv` (Miami 2025 has no timed laps on record, and is
reported as excluded rather than patched).

**What this still cannot claim.** These are today's copies of those sessions,
not the copies that existed those weekends. Timing data gets corrected. This is
a reconstruction from the present archive, not a replay of what was knowable at
the time — the same limit `pipeline.py` states about every other file here.

## Why the model is not scored yet

The per-circuit tables as committed — overtaking rates, safety-car rates, pit
loss, the strategy menu, tyre curves — are fitted on the whole 2018–2025
archive. For a 2024 target, that archive contains 2025 **and the target race
itself**. Handing them over would be precisely the pooling leak this exercise
exists to rule out.

Most of them the model can do without: it has documented fallbacks, and
withholding them measures a weaker model than the one that runs on a current
race. That understates its strength, which is the direction an unproven claim
should err in.

**The tyre curve is the exception.** `tyre_curve_params.json` has no fallback —
the store raises rather than guessing — so the model will not start without it.
It cannot be filtered either: the file carries no per-cell record of which
seasons fed it, so there is no way to subset it to a cutoff. Making it
leak-free means refitting the chain behind it — `tyre_age_profile.csv`,
`stint_index.csv`, `stint_limits.csv`, `pit_summary.csv` — once per cutoff.

That is the remaining work, and it is a real piece of work rather than a
loose end. Everything upstream of it is built and verified:

- the scoring instrument, 33 tests, including the brief's own checks — a
  perfect call scores 0, a coin flip scores 0.25, a perfect distribution
  scores 0 RPS
- the race universe and ground truth
- both baselines, measured
- historical qualifying, downloaded
- the pre-race input builder, which produces sensible pace (before Monza 2024
  it ranks NOR fastest, then HAM, VER, LEC — which is what mid-2024 looked
  like) and is verified to write nothing but the four files it should
- `F1_DATA_DIR`, `F1_SEASON` and `F1_TARGET_RACE`, so the engine can be aimed
  at a past race without restructuring it. Unset, every normal run is
  bit-identical to before: seed 42 still returns LEC 0.360, HAM 0.280.

## What may not be concluded from any of this

The model has not been shown to beat the grid. It has not been shown to lose to
it either. Nothing in this document is evidence about the model's accuracy,
because the model has not yet been scored — what is established is the sample,
the ground truth, the baselines those scores will be read against, and that the
machinery doing the reading is correct.
