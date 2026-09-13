# Changelog

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
