# Status

Where the project actually stands, as of v2.7. A roadmap having been written is
not a version having been delivered, so this separates what is in the source
from what is not.

## Done, and checked in the source

| | what it is | how it is checked |
|---|---|---|
| **v2.0** reactive strategy | pit lap chosen by pricing candidates against rivals and traffic | 32 tests |
| **v2.1** weather | rain and track wetness as separate states, off by default | 37 tests |
| **v2.2** retirements | accidents and failures as separate causes, per driver and team | 37 tests |
| **v2.3** parameters | every constant inventoried with its evidence; set versioned | 25 tests |
| **v2.4** pipeline | one entry point; staleness by content hash and producer source | 22 tests |
| **v2.5** install | `requirements.txt` verified in an empty virtualenv | 321 tests pass there |
| **v2.6** instrument | scoring, baselines, universe, bootstrap | 33 tests |
| **v2.7** start | grid lock replaced by a start calibrated on 1313 observations | 17 tests |

321 tests, passing in the development environment and in a clean one built from
`requirements.txt` alone. Seed 42 reproduces to the digit across pandas 2.3.3 /
numpy 2.5.2 and pandas 3.0.5 / numpy 2.5.3.

## Partly done

**v2.6 backtest.** The instrument is finished and the baselines are measured —
grid order scores 3.141 mean absolute position error over 52 races, 95%
interval [2.867, 3.421]. **The model itself is not scored.**
`tyre_curve_params.json` has no fallback and carries no record of which seasons
fed each cell, so it can neither be withheld from a historical run nor cut to a
cutoff. Scoring the model leak-free needs that chain — `tyre_age_profile.csv`,
`stint_index.csv`, `stint_limits.csv`, `pit_summary.csv` — refitted per cutoff.
Everything upstream of it is built: `BACKTEST.md` has the detail.

**v2.7 retention.** The start reproduces the field-wide spread (1.93 against
1.81 measured) but not the shape of the retention curve: pole leads at the end
of lap one 38% of the time in the model against 78% in the archive. Part of
that gap is the comparison itself — one circuit and one 2026 grid against 72
races where pole is usually the quickest car — and part is real. It is left
visible rather than patched with a hand-set front-row protection.

## Open

- **Prediction accuracy is unmeasured.** Nothing in this project yet shows the
  model forecasts races better than the grid does, or worse. The tests check
  internal correctness and reproducibility, neither of which is evidence about
  forecasting.
- **No licence.** The repository has none, which is a decision for its author
  rather than something to be chosen on their behalf. Until one exists the
  default applies: no permissions are granted to anyone else.
- **No screenshot.** `README.md` describes the interface without showing it.
- **Not published.** Everything here is committed locally. Nothing has been
  pushed, and the remote still holds v1.6.

## Deliberately not done

- **Files were not reorganised.** The layout is documented, the import graph
  works, and moving 95 files to look tidier is risk without a reader.
- **No grid-side coefficient.** Tested at p = 0.248 over 25 circuits and not
  separable from grid position. Zero here is a simplification, not a finding
  that the sides are equal.
- **No per-circuit start scale.** Tested at p = 0.489; three races a circuit
  cannot tell a fast start from a lucky one.
- **No separate reaction, slipstream or braking coefficients.** The archive
  records a position at the end of lap one and nothing inside it, so three
  coefficients would be three names for one observation.
