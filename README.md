# F1 Race Model

A lap-by-lap Monte Carlo of a Formula 1 race. Ten thousand races are run,
each one simulated lap by lap with tyre wear, pit strategy, safety cars, red
flags, overtaking and retirements, and the finishing order is read off the
distribution rather than guessed at.

The interface exists mostly to answer one question: **where did that number
come from.** Every parameter carries a label saying whether it was measured
on this circuit, derived from another cell, borrowed from a donor circuit,
shrunk toward the calendar because the sample was thin, or simply chosen by a
person and still owed a measurement.

```
pip install -r requirements.txt
streamlit run app.py
```

Nothing is downloaded at startup and nothing is fetched on first run. Every
number the interface reads is committed - 46 files, 871 KB - so a fresh clone
opens on a finished prediction rather than on a progress bar.

That install line is the whole of it. `requirements.txt` holds five packages
and was checked by building an empty virtual environment from it alone: all
271 tests pass and the interface serves, with no `fastf1` and no network. The
download half of the project is a separate file, `requirements-pipeline.txt`,
and is only needed to rebuild data rather than to use it.

## What the model does

**Tyres.** A two-segment degradation curve per compound per circuit,
`D(a) = b1*a + b2*a^2 + g*max(0, a-tau)^2`, fitted on 2018-2025 stints with
the cliff identified by right-censored survival analysis. Each curve carries
a cap: the longest stint it was actually measured for, past which it refuses
to extrapolate rather than inventing a number.

**Strategy.** The menu of one- and two-stop plans is priced against this
circuit's own curves and the measured pit loss, then sampled by cost.

When to stop is then decided lap by lap. Inside a window around the planned
stop the car prices every pit lap still open to it - what the old tyre costs,
what the new one costs, what the stop costs with whatever flag is out, and up
to four rivals' worth of traffic - and takes the cheapest. Every candidate is
priced to the same horizon, so a short window is never compared against a
long one.

The plan itself does not move: same number of stops, same compound order,
only the lap. And the car decides on what a pit wall could actually see. It
does not know a rival's strategy, the lap that rival will stop on, or the
safety car schedule the simulator drew before the race - which is a property
of the arrays the decision is handed, not a promise about how they are read.

**Overtaking.** Pass probability per lap depends on how much quicker the
following car is *and* on how close it actually is. The second term matters
more than the first - within a second, the measured rate runs from 0.69
inside a quarter second to 0.10 approaching a full one - and it was missing
until v1.5. A team term from measured wheel-to-wheel strength sits on top,
shrunk to what a permutation test supports.

**The start.** Cars draw a start time and the order falls out of sorting them,
so a place gained is a place someone else lost and nobody is promoted into an
occupied slot. It is calibrated on 1313 racing starts - places counted only
among the cars that were actually racing, because a driver who inherits two
places from a retirement in front has passed nobody. Pole is exposed to about a
third of the spread the midfield sees, which is measured rather than assumed.

It changes lap one and it barely touches the finish: over 10,000 runs, turning
it off moves a win probability by 0.0005 on average. Across a race distance the
first lap washes out.

**Neutralisation.** Safety cars, virtual safety cars and red flags, with
rates measured per circuit and shrunk toward the calendar where a circuit has
too few races to speak for itself. The field bunches behind a safety car and
a red flag hands out a free tyre change. In the wet they start 2.24x as often
per eligible lap - measured, against the 1.8x the project used to assume.

**Rain.** Optional, and off by default. Rain and a wet track are separate:
the circuit takes laps to flood and laps to dry, so a shower that stops is
not the same as a track that is dry. Intermediates and wets are real
categories with their own pace, spread and wear, and the car decides when to
change by pricing the alternatives the same way it prices a dry stop.

Half of this is measured and half is not, and the interface says which:

| | |
|---|---|
| measured | inters cost +13.5% of the lap against the same race's own dry median, over 3,167 green laps and 13 races |
| measured | full wets cost +25.2%, over 237 laps and 3 races |
| measured | lap-to-lap spread runs 3.1x and 4.5x the dry figure |
| measured | neutralisations start 2.24x as often per wet lap |
| thin | an inter on a dry track costs about 13.8% of a lap, from 5 lap-instants where cars were out on both at once |
| **scenario** | where the crossover actually sits. 32 lap-instants over 11 races is not a curve |
| **scenario** | the wetness index itself. There is no water measurement in the data |
| **assumption** | how a wet tyre wears. A drying track makes an inter quicker with age, which is the opposite sign to wear |

A wet result is conditional on a scenario somebody chose. Nobody has measured
how likely any of them is, so there is no single combined win probability on
offer and the interface does not pretend to one.

**Why cars stop.** Retirements split into accidents and mechanical
failures, both measured per lap at risk rather than per race - a driver who
crashed twice in five races did not crash on 40% of laps, and a car that
stopped on lap 6 was not exposed on lap 7. Accident risk follows the driver,
mechanical risk follows the car, and both are pulled toward the field where
the sample is thin, because zero observed crashes is not zero risk.

An accident can bring out a flag, and the background safety-car rate is
scaled down when it does, so the same incidents are not counted twice.

Three of this version's starting assumptions did not survive the measurement:

| assumed | measured |
|---|---|
| failures peak mid-race (a bell) | flat, slightly rising; the bell loses on AIC |
| ~90% of accident retirements bring out a safety car | 49.6%, over 123 deduplicated incidents |
| 6% of cars retire, cause unknown | 32% accidents, 34% mechanical, **34% unexplained** |

That last row is a gap, not a third cause. A third of retirements in eight
seasons say only "Retired", and they are reported rather than spread over the
two causes the model can produce.

**Affinity.** How well each circuit suits each car, from 2026 qualifying
rather than the 2022-25 record - the older cars measure +0.6% skill against
2026 results, which is nothing.

## Where the numbers come from

Every constant the model runs on is inventoried in
`Simülasyon/parameters.py`: what kind of number it is, what it was measured
against, how much the output moves when it moves, and whether it was changed.
The set is versioned and reverting is one line.

The audit found four constants that nothing reads - `POOLED_THRESHOLD`, the
`MIN_GAP` import, the whole of `Tyre_model/tyre_cliff.py`, and the Pirelli
rating chain, which ends at a number the interface displays and the lap loop
never reads. Three were changed against held-out data: safety cars start
earlier than assumed (43% of race distance, not 60%) and per-circuit rates
need four times more pooling than they were getting.

One finding outweighs the rest. `NOISE_AUTOCORR` is 0.60 and measured
lap-to-lap persistence is 0.155, with 93% of a driver's lap-time variance
sitting inside a stint rather than between stints. The 0.60 is not a
correlation anyone measured - it was raised until the favourite stopped
winning 98% of simulations, and it stands in for race-level variance the model
has no other source for. Setting the measured value takes the leading car from
32.5% to 38.1%. It is left as it is and labelled hand-set rather than quietly
corrected in either direction; it is the largest open assumption in the model.

## Layout

```
app.py                      the interface
Simülasyon/                 the model, and the one entry point to it
  pipeline.py               what data exists, what is stale, and run the race
  target_race.py            which Grand Prix is being predicted
  simulate.py               the race model, and run() for the interface
  reactive_strategy.py      when to pit, priced against rivals and traffic
  starts.py                 the start and lap one, measured
  race_select.py            predicting any 2026 round that has qualified
  weather.py                rain, a drying track, and the tyre that suits it
  dnf.py                    accidents, failures, and the flags they bring out
  tracks.py                 per-circuit constants
  fuel_effect.py            what a lighter car is worth per lap
  parameters.py             every constant, its kind, and what it is worth
  calibrate.py              measuring the ones that were only ever chosen
  backtest.py               scoring a prediction against what happened
  backtest_inputs.py        rebuilding what a past race could have known
  diagnostics.py            the per-lap report
  profile_run.py            where the time and the memory actually go
  data_prep/                everything that writes into data/
    fetch.py                downloads the season, withholds the target race
    dataset.py              the 2018-2025 lap cache
    clean.py                driver pace and lap-time sigma
    pit_analysis.py         pit loss and strategy costs
    overtaking.py           pass rates, per circuit
    overtake_speed.py       per-team overtaking and straight-line speed
    neutralization.py       safety car and red flag rates
    retirements.py          causes, exposure and timing, measured
    wet_conditions.py       what the lap data can and cannot say about rain
    team_affinity.py        which 2026 circuits suit which 2026 team
    track_features.py       circuit character from telemetry
    track_affinity.py       the 2022-25 record, per driver and team
    quali_history.py        historical qualifying, for the backtest
  tests/                    every test, run one file at a time
  Tyre_model/               curve fitting, the store, and its HTTP API
data/                       what the model reads
output/                     predictions, distribution, diagnostics
  run_summary.json          what the committed sample prediction actually is
docs/
  CHANGELOG.md              what changed, and whether it changed predictions
  BACKTEST.md               how good the predictions actually are
  STATUS.md                 what is done, part-done, and open
  tyre-model.md             the degradation curve and how it is fitted
```

`docs/CHANGELOG.md` is the place to look before comparing a result against an
older one: it says which releases moved a coefficient and which only moved
code. `docs/STATUS.md` is the shortest honest answer to "is this finished".

## Changing the race

Pick it in the interface. Every 2026 round whose qualifying is on file can be
predicted, and driver pace is rebuilt from the rounds *before* the one chosen -
so a prediction never reads the race it is predicting. Rebuilding the round the
pipeline last fetched for reproduces its committed pace file exactly, which is
what says the two paths are the same path.

Two things differ from the fetched race. The official grid, with penalties
applied, is only on file for that one; the others start from the qualifying
order, and the run says which it used. And an early round has fewer races of
pace behind it, which is why rounds without at least two are not offered rather
than answered from fallback constants.

To move the pipeline itself - which race gets its laps withheld and its grid
downloaded - is still one line, in `Simülasyon/target_race.py`:

```python
TARGET_RACE = 'italian'      # any key in RACE_REGISTRY
```

Then ask the pipeline what that broke:

```
python -m Simülasyon.pipeline                     what is ready, stale, missing
python -m Simülasyon.pipeline --update --network  rebuild only what moved
python -m Simülasyon.pipeline --run               check, then predict
```

It knows which steps feed which, so it rebuilds what actually depends on the
change and leaves the rest alone. Staleness is decided by content, not by
date - and a step is also stale when *its own source* has changed, which is
the case no input hash can see: a cleaning rule moves, every raw byte stays
where it was, and the derived file is now built by code that no longer exists.

The interface runs the same check before every prediction and refuses rather
than building a confident answer on another circuit's grid.

What it is not: a way to reconstruct the past. One copy of each file is kept
and overwritten, so a record downloaded today is today's version of it,
whatever date the event has.

## Rebuilding the data

Not needed to run the model. To do it anyway:

```
pip install -r requirements.txt -r requirements-pipeline.txt
python -m Simülasyon.data_prep.dataset --rebuild     # 2018-2025 laps, slow
python -m Simülasyon.data_prep.fetch                 # the current season
python -m Simülasyon.data_prep.clean
python -m Simülasyon.Tyre_model.fit_tyre_curve
python -m Simülasyon.data_prep.overtaking
python -m Simülasyon.data_prep.neutralization
python -m Simülasyon.data_prep.pit_analysis
python -m Simülasyon.data_prep.team_affinity
python -m Simülasyon.data_prep.wet_conditions
python -m Simülasyon.data_prep.retirements
python -m Simülasyon.calibrate
python -m Simülasyon.parameters
```

The raw lap tables are not committed - 40 MB of FastF1 download that these
commands put back.

## Tests

```
python -m Simülasyon.tests.test_pass_model            # 23
python -m Simülasyon.tests.test_team_affinity         # 37
python -m Simülasyon.tests.test_reactive_strategy     # 32
python -m Simülasyon.tests.test_weather               # 37
python -m Simülasyon.tests.test_dnf                   # 37
python -m Simülasyon.tests.test_parameters            # 25
python -m Simülasyon.tests.test_pipeline              # 22
python -m Simülasyon.tests.test_backtest              # 33
python -m Simülasyon.tests.test_starts                # 17
python -m Simülasyon.tests.test_frontend              # 33
python -m Simülasyon.tests.test_race_select           # 21
python -m Simülasyon.Tyre_model.test_tyre_curve # 58
```

Most of them exist because something was wrong once. The gap term produced
118 overtakes at a circuit whose history says forty, because a coefficient
fitted on laps with real pace in hand was being applied to laps without any;
there is a test for that. A retiring car used to be scored where it was
classified rather than where it was actually running, which turned circuit
suitability into a reliability ranking; there is a test for that too.

## Reproducing a run

`RANDOM_SEED` is 42 and a run is a pure function of it, the committed data and
the parameter set. Two runs at the same seed and the same simulation count
return the same table, to the digit.

That survives a change of library, which is the part worth stating: the same
seed and the same 300 simulations were run under pandas 2.3.3 with numpy
2.5.2, and again under pandas 3.0.5 with numpy 2.5.3 - a major version apart -
and the finishing distribution came back identical rather than merely close.
Nothing in the lap loop depends on a floating-point detail those versions
changed.

Both of those were Python 3.13.12 on Windows 11. Linux and macOS are not
claimed, because they were not run.

The result carries what it needs to be compared against another: the race, the
seed, the parameter set version, the flags, the number of simulations that
actually completed, and a content hash of every data file it read. Two results
with the same hashes and the same seed are the same experiment; two without
are not, and the run summary is what says so.

The committed `output/predictions.csv` is one such run, and
`output/run_summary.json` beside it is its label: Italian Grand Prix 2026,
10,000 simulations, seed 42, parameter set `v2.3-measured`, dry scenario,
23.5 seconds, and the hash of all nineteen inputs. It is there so the
interface opens on a finished prediction instead of an empty screen - and so
that a number read off it can be traced to the run that produced it rather
than to an unlabelled table of percentages.

## Speed

Measured before anything was changed, which turned out to matter: the
simulation core is 99.8% of a run, and everything else - loading inputs,
pricing strategies, collecting results, drawing the representative race -
comes to about a fifth of a second at 10,000 simulations. Peak memory is
433 MB, of which 73 MB is the per-lap traces kept so one race can be replayed.

Two optimisations were tried and neither shipped:

- Cutting redundant arithmetic out of the pit decision's traffic loops. Proven
  bit-identical, strictly less work, and worth 1.4% across three interleaved
  pairs of 10,000-run measurements - against a run-to-run spread of a third on
  this machine. That is not a measured gain.
- Running the 10,000 simulations in chunks, which the brief suggests for
  memory. It is *slower*: ten chunks of 1,000 take 1.43x as long as one run of
  10,000, because the lap loop's per-lap overhead does not depend on how many
  races are in flight and small chunks pay it ten times over.

Parallelism and compilation were not attempted, on the strength of that second
result: if splitting the work across chunks in one process already costs 43%,
splitting it across processes - with 433 MB to copy - is not the next thing to
try.

## Notes on what is not modelled

Damage a car carries on with. Penalties and disqualification. Multi-car
collisions - a first-corner pile-up is modelled as separate independent
retirements, and no blame is assigned.
Team orders - a team mate is priced as ordinary traffic, with no pit
priority and no stacking. Fuel saving and lift-and-coast. Per-driver tyre
management - the v1.2 curve has no driver term, so the per-driver strategy
flag is left switched off rather than pretending to a difference of zero.

**Where the rain falls.** One condition for the whole circuit. A shower over
one sector, a dry line that forms on the racing line, and standing water at a
single corner are all outside this version.

**Position on the road.** The model carries cumulative race time, not where a
car is around the lap. Two cars a lap apart have a large time gap and can be
side by side on the road, so pit-exit traffic is found in time and any rival
further away than one lap time is dropped rather than guessed at. Where the
leaders lap the tail this under-counts traffic for the cars being lapped.
