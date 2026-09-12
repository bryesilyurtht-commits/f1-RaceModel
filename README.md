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
streamlit run app.py
```

Nothing is downloaded at startup. Everything the interface reads is committed
and comes to about 250 KB.

## What the model does

**Tyres.** A two-segment degradation curve per compound per circuit,
`D(a) = b1*a + b2*a^2 + g*max(0, a-tau)^2`, fitted on 2018-2025 stints with
the cliff identified by right-censored survival analysis. Each curve carries
a cap: the longest stint it was actually measured for, past which it refuses
to extrapolate rather than inventing a number.

**Strategy.** The menu of one- and two-stop plans is priced against this
circuit's own curves and the measured pit loss, then sampled by cost. Cars
pit reactively, early or late, as the tyre actually goes off.

**Overtaking.** Pass probability per lap depends on how much quicker the
following car is *and* on how close it actually is. The second term matters
more than the first - within a second, the measured rate runs from 0.69
inside a quarter second to 0.10 approaching a full one - and it was missing
until v1.5. A team term from measured wheel-to-wheel strength sits on top,
shrunk to what a permutation test supports.

**Neutralisation.** Safety cars, virtual safety cars and red flags, with
rates measured per circuit and shrunk toward the calendar where a circuit has
too few races to speak for itself. The field bunches behind a safety car and
a red flag hands out a free tyre change.

**Affinity.** How well each circuit suits each car, from 2026 qualifying
rather than the 2022-25 record - the older cars measure +0.6% skill against
2026 results, which is nothing.

## Layout

```
app.py                      the interface
Simülasyon/
  simulate.py               the race model, and run() for the interface
  target_race.py            which Grand Prix is being predicted
  fetch.py                  downloads the season, withholds the target race
  clean.py                  driver pace and lap-time sigma
  dataset.py                the 2018-2025 lap cache
  overtaking.py             pass rates, per circuit
  overtake_speed.py         per-team overtaking and straight-line speed
  team_affinity.py          which 2026 circuits suit which 2026 team
  track_features.py         circuit character from telemetry
  track_affinity.py         the 2022-25 record, per driver and team
  neutralization.py         safety car and red flag rates
  pit_analysis.py           pit loss and strategy costs
  diagnostics.py            the per-lap report
  Tyre_model/               curve fitting, the store, and its HTTP API
data/                       what the model reads
output/                     predictions, distribution, diagnostics
```

## Changing the race

One line, in `Simülasyon/target_race.py`:

```python
TARGET_RACE = 'italian'      # any key in RACE_REGISTRY
```

Then rebuild what depends on the split, because `fetch.py` withholds the
target race's laps and downloads its grid instead:

```
python -m Simülasyon.fetch
python -m Simülasyon.clean
python -m Simülasyon.team_affinity
```

The interface says so itself if the data on disk belongs to a different race.

## Rebuilding the data

Not needed to run the model. To do it anyway:

```
pip install -r requirements.txt -r requirements-pipeline.txt
python -m Simülasyon.dataset --rebuild     # 2018-2025 laps, slow
python -m Simülasyon.fetch                 # the current season
python -m Simülasyon.clean
python -m Simülasyon.Tyre_model.fit_tyre_curve
python -m Simülasyon.overtaking
python -m Simülasyon.neutralization
python -m Simülasyon.pit_analysis
python -m Simülasyon.team_affinity
```

The raw lap tables are not committed - 40 MB of FastF1 download that these
commands put back.

## Tests

```
python -m Simülasyon.test_pass_model            # 23
python -m Simülasyon.test_team_affinity         # 37
python -m Simülasyon.Tyre_model.test_tyre_curve # 58
```

Most of them exist because something was wrong once. The gap term produced
118 overtakes at a circuit whose history says forty, because a coefficient
fitted on laps with real pace in hand was being applied to laps without any;
there is a test for that. A retiring car used to be scored where it was
classified rather than where it was actually running, which turned circuit
suitability into a reliability ranking; there is a test for that too.

## Notes on what is not modelled

Weather changes mid-race. Driver error as anything but a retirement. Damage.
Team orders. Fuel saving and lift-and-coast. Per-driver tyre management -
the v1.2 curve has no driver term, so the per-driver strategy flag is left
switched off rather than pretending to a difference of zero.
