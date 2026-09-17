# Changelog

## Team orders: no double stacking, no teammate dicing on the same plan

Two teammates were racing each other exactly like two rivals. Real teams do
not: the pit wall will not queue both cars through one jack, and it will not
let the rear car dice past the front one on track while both are running the
same plan.

**No double stacking.** If both cars from a team would pit on the same lap,
whichever is behind on the road is held out one lap - its own decision
(due, opportunistic, reactive, or a weather change) still stands, it is just
not taken this lap. A cap-forced stop (`ENFORCE_STINT_CAP`) is never
delayed: the cap exists because the curve was never fitted past that age,
and asking a tyre to run one lap longer than the data covers is the thing
the cap was built to prevent. Checked directly: across a full field and a
real strategy mix, zero same-lap stops survive between two teammates who are
both still racing.

**No teammate dicing on the same plan.** Two cars nose-to-tail, same team,
on the same tyre compound within three laps of age, no longer get a dice
roll for the place between them - the wheel-to-wheel pass model that
decides every other fight on the grid is switched off for that one pair,
that one lap. Nothing about a genuine strategy difference is touched: a
teammate on a different compound, or one that just pitted or retired, still
changes the order exactly as before. An undercut against your own teammate
still works; racing past them for no reason does not.

Both mechanisms are pure functions - `apply_double_stack_delay` and
`team_order_hold` in `simulate.py` - independently unit-tested against
synthetic arrays, plus whole-race checks in `test_team_orders.py` (13
tests). Both are runtime flags (`DOUBLE_STACK_ENABLED`, `TEAM_ORDERS_ENABLED`),
on by default, visible in the Parameters tab under Racing.

## Stint chart: a gap at every pit, not just a border

The previous fix split same-compound pit stops into two bars correctly, but
left them touching - a dark 1.4px border and a small triangle were the only
things saying "pit here", and at the zoom level the chart actually renders
at, two HARD bars in a row still read as one stint by eye. Each bar is now
inset from its own boundaries (up to 0.16 laps, capped at a third of a very
short stint), so there is a visible strip of page background at every real
stint boundary regardless of colour. One more test in `test_frontend.py`
checks the gap is there and sits on the pit lap.

## Stint chart: a same-compound pit stop is two stints, not one

`build_stints` accepted `pits_by_lap` and never read it - a stint ended only
when the compound changed, so a stop that fitted the same compound again (a
splash-and-dash, or `ENFORCE_STINT_CAP` forcing a fresh set because nothing
else was left to spend) drew as one uninterrupted bar. At the Spanish Grand
Prix this showed a driver on a 49-lap MEDIUM stint against a curve fitted for
32 - the tyre model was never wrong, the chart was silently merging two real
stints (25 and 24 laps, a stop at lap 25) into one rectangle. Verified against
the model's own cap: no stint in a fresh run now exceeds what its compound was
measured for.

Tyre colours moved to the broadcast convention - soft red, medium yellow, hard
white - from the page's own accent ramp. Pit stops get an explicit marker
rather than relying on the seam between two bars, which was not a reliable
boundary when both bars were the same fill colour.

Eight tests in `test_diagnostics.py`, plus two more in `test_frontend.py` for
the colours and the marker.

## v2.7 - the start, measured

The grid no longer simply holds through lap one. Cars draw a start, the draw is
resolved into an order, and the order is one the archive supports.

**What the lock was.** `LOCK_START_ORDER` gave every car the time gap its slot
implied and then sorted those times back onto the starting order, so the field
left the line realistically spread and nobody was ever promoted. It is now
derived from `START_MODEL` rather than set, and it is kept because "the start
does nothing" is the baseline the new start has to be read against.

**What it was replaced with.** One time drawn per car, sorted into an order.
Not a per-driver "gains three places": two cars cannot be promoted into the
same slot, a place gained is a place someone else lost, and every active car
appears exactly once because the result is a sort rather than a set of
independent draws needing repair afterwards.

**Calibration.** 1313 racing starts from 72 races, 2022-25. The measurement is
in `starts.py`, and the cleaning is most of it: a driver who starts eighth and
runs sixth has not passed two cars if two cars ahead retired at turn one,
pitted, or started from the pit lane. Places are counted among the cars that
were actually racing, so 513 car-races were classified out of the sample rather
than silently counted as successful starts.

- `START_SPREAD` = 2.60 positions, pooled. Per-circuit spread runs 1.21 to 3.00
  across the fifteen circuits with enough data, which looks like circuit
  character until it is tested: permuting race labels between circuits
  reproduces it 49% of the time (p = 0.489). Three races cannot tell a fast
  start from a lucky one.
- `START_CHAOS_PROFILE` replaces the linear ramp from `START_FRONT_STABILITY`
  to 1.0. The measured exposure is not a ramp - it is flat at the front, peaks
  around slot thirteen and falls again at the back, where there is less room to
  lose places than to gain them. `START_FRONT_STABILITY` survives as the first
  entry, 0.34, so the constant the brief names still has a value and a source.
  It was 0.35, chosen, which was close by accident and applied through a shape
  that gave pole the midfield's exposure.

**Grid side: no coefficient.** The first test scored each car against its own
slot's mean and compared odd against even. It returned exactly zero, and it had
to - parity is a deterministic function of the slot, so the test could not have
found anything. Rebuilt to test what is actually identifiable, per-circuit
parity effects that disagree with each other, it gives p = 0.248 over 25
circuits. The model carries no side term. That is a simplification, not a
finding that the sides are equal.

**Lap one is not double counted, and not zeroed either.** `order` is carried
between laps and moved by the overtaking sweep, not rebuilt by sorting the
clock, so that sweep is the mechanism that turns the start's times into
positions. Switching it off on lap one - the obvious reading of "do not apply
overtaking twice" - stops the start happening at all: every slot's hold rate
went to 97-100% and field-wide change SD to 0.50 against an observed 1.81. The
start owns lap one by owning the times, and `START_SPREAD` is fitted against
the lap-one order the loop actually produces.

**What it does to the race: almost nothing.** Over 10,000 runs, mean absolute
change in win probability is 0.0005 and in expected finishing position 0.008
places. Across 53 laps the lap-one reshuffle washes out as cars find the order
their pace implies. This is worth stating plainly: v2.7 makes lap one resemble
lap one, and it is not evidence of a better race prediction.

**Where it does not fit.** The model reproduces the field-wide spread (1.93
against 1.81 measured) but not the retention profile. Pole leads at the end of
lap one 38% of the time in the model against 78% in the archive. Some of that
gap is a mismatch in the comparison - one circuit with one 2026 grid against 72
races where pole is usually the quickest car - and some is real. It is not
patched with a hand-set front-row protection, which would be the double
counting §8 of the brief warns about.

**Restarts stay separate.** A red-flag restart already used the running order
rather than the original grid and shares the standing-start core, which is
correct. A safety-car restart never called it and still does not.

**New.** `starts.py` and the two tables it writes, 17 tests in `test_starts.py`,
and `lap_one_order` on the diagnostic so the calibration stays checkable.

## v2.6 - the measuring instrument, and what it has measured so far

No model behaviour changed. `simulate.py` and `target_race.py` gained three
environment overrides - `F1_DATA_DIR`, `F1_SEASON`, `F1_TARGET_RACE` - so the
engine can be aimed at a past race without restructuring it. Unset, which is
every normal run, nothing behaves differently: seed 42 still returns LEC 0.360,
HAM 0.280 on 300 simulations, and all 271 earlier tests pass unchanged.

**What is measured.** The two naive baselines, over 52 races from 2023 round 19
to 2025 round 24. Grid order scores 3.141 mean absolute position error, 95%
interval [2.867, 3.421] resampling whole races. The grid-history baseline
scores 0.032 / 0.067 / 0.161 Brier on win / podium / points. `BACKTEST.md` has
the sample rules, the calibration tables and the caveats.

The 3.45 the status document reports is not reused. It is not compared against
either, because the sample behind it is not known.

**What is not measured.** The model itself. `tyre_curve_params.json` has no
fallback and no per-cell record of which seasons fed it, so it can neither be
withheld nor filtered to a cutoff; scoring the model leak-free needs that chain
refitted per cutoff. Everything upstream of that is built and tested.

**New.** `backtest.py` (scoring, baselines, universe, bootstrap), 33 tests in
`test_backtest.py`, `backtest_inputs.py` (pre-race input assembly),
`quali_history.py` and the 51 sessions it downloaded into
`data/quali_history.csv` - historical qualifying, which the archive did not
have and the model's delta-to-pole needs.

## v2.5 - packaging and a verified install

The work in this release is engineering, not modelling. One import moved; no
coefficient did. Predictions from v2.4 and v2.5 at the same seed are the same
predictions.

**Installing.** `pip install -r requirements.txt` was checked by building an
empty virtual environment from it and nothing else: all 271 tests pass there
and the interface serves. Before this, one of them did not.

`Simülasyon/track_features.py` imported `fastf1` at the top of the module, and
`team_affinity.py` reads two pure definitions out of it - `FEATURE_SET` and
`standardise`, neither of which touches the network. So a clone that installed
only `requirements.txt`, exactly as the README said to, could not run
`test_team_affinity`. The import now happens inside `collect()`, which is the
only function that downloads anything. Nothing else about the module changed.

**Reproducing a run.** Seed 42 was run twice in one environment and then in a
second built a major library version apart - pandas 2.3.3 with numpy 2.5.2,
against pandas 3.0.5 with numpy 2.5.3. All four finishing distributions are
identical to the digit. Python 3.13.12 on Windows 11 in both cases; other
platforms are not claimed, because they were not run.

**README.** It said the committed data comes to about 250 KB. It is 871 KB
across 46 files, and now says so. The install step was missing from the top of
the file and is now the line above `streamlit run app.py`.

### Also in this release

v2.5 is the first published version since v1.6, so the v2.0-v2.4 work arrives
with it. That work *did* change model behaviour:

- **v2.0 reactive strategy.** Pit timing is decided lap by lap inside a window
  around the planned stop, priced against the old tyre, the new one, the flag
  that is out and up to four rivals' traffic. The plan itself does not move.
- **v2.1 weather.** Rain and a wet track as separate states, with a circuit
  that takes laps to flood and laps to dry. Off by default.
- **v2.2 retirements.** Accidents and mechanical failures as separate causes,
  measured per driver and per team, with accidents able to bring out a flag.
- **v2.3 parameters.** Every constant inventoried with its evidence: measured,
  derived, donor, shrunk, or hand-set. Three safety-car constants were changed
  against held-out data; the parameter set is versioned `v2.3-measured` and
  reverting is one line.
- **v2.4 pipeline.** `python -m Simülasyon.pipeline` as a single entry point
  for what data exists, what is stale and why, and running the race. Staleness
  is content-hashed and also tracks the producing module's own source.

Twenty-one runtime flags and one scenario choice are recorded with every
result, alongside the seed, the parameter set version and a content hash of
each data file read. That record is what makes two results comparable.

### Not in this release

Prediction accuracy is still not measured. The model is tested for internal
correctness - 271 of those - and reproducibility, and neither of those is
evidence that it forecasts races well. A backtest against held-out races is
v2.6, and until it exists the honest summary is that the probabilities are a
carefully-built model's opinion, not a validated forecast.

Files were not reorganised. The layout is documented in the README and the
import graph works; moving 95 files to look tidier is risk without a reader.
