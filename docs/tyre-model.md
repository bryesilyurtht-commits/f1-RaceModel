# Tyre degradation model

Per track and compound, the time a tyre loses to age:

```
D(a) = b1*a + b2*a^2 + gamma * max(0, a - tau)^2
```

Seconds, against a fresh tyre of the same compound. Deterministic — same
track, compound and age, same answer, every time. Driver, car, temperature
and traffic are out of scope in this version, so every car on the same
compound at the same age loses the same time.

`b0` is fixed at zero. The SOFT/MEDIUM/HARD pace difference is **not** in this
curve; the simulation adds it separately from `compound_scaling.csv`.

## The age `a`

`a = FastF1 TyreLife - 2`.

The profiles this is fitted to are expressed relative to the median of each
stint's opening laps, which puts their zero at TyreLife 2. So `D(0)` is a tyre
that has already completed two laps, not one out of the blanket. The offset is
published as `age_offset` in the parameter file and lives in one constant,
`tyre_curve.AGE_OFFSET`.

Age carries over: a used set continues at its own age, and a pit stop does not
reset it by itself.

## Files

| File | What it does |
|---|---|
| `tyre_curve.py` | The formula. No I/O, no state. |
| `fit_tyre_curve.py` | Fits the parameters, writes `data/tyre_curve_params.json`. |
| `tyre_store.py` | Loads that file, resolves track names, caches. |
| `tyre_api.py` | HTTP access. Standard library only. |
| `test_tyre_curve.py` | 58 checks, API included. |

## Rebuilding the parameters

```
python Simülasyon/Tyre_model/fit_tyre_curve.py
```

Reads `tyre_age_profile_track.csv`, `stint_index.csv`, `stint_limits.csv` and
`pit_summary.csv`. Writes the parameter file plus `tyre_curve_report.csv`,
which records where every number came from and why any cliff was or was not
fitted.

Two stages, because lap times alone cannot see a cliff:

1. **`b1`, `b2`** from the lap-time profiles, inside the observed age range.
2. **`tau`, `gamma`** from when teams actually stop. A stint that ended in a
   pit decision is an event; one that ran to the flag or ended in a
   retirement is censored. If teams stop earlier than the cliff-free curve's
   economics predict, the difference is the cliff. If they do not, there is no
   cliff evidence and `gamma = 0`, `tau = null`.

Every curve carries `max_age`, the range it was measured over. Past it the
curve refuses to answer rather than extrapolating.

## Using it in code

```python
from tyre_store import load

store = load()
curve = store.get('Netherlands', 'SOFT')   # aliases resolve to 'Dutch Grand Prix'

curve.loss(12)               # total loss at age 12, seconds
curve.incremental_loss(12)   # what lap 12 cost over lap 11
curve.curve()                # ages 0..max_age with losses
```

`loss` and `slope` are different quantities and the API keeps them apart:
`incremental_loss` is the discrete step `D(a) - D(a-1)`, `slope` is `D'(a)`.

## The API

```
python Simülasyon/Tyre_model/tyre_api.py --port 8077
```

| Endpoint | Body |
|---|---|
| `POST /v1/degradation` | `{track, compound, tyre_age}` |
| `POST /v1/curve` | `{track, compound, max_age?}` |
| `POST /v1/curves` | `{queries: [{track, compound, max_age?}]}` |
| `GET /v1/catalog` | – |

The HTTP layer does no arithmetic: it calls the same functions the simulation
calls, so the two cannot drift apart. There is no seed and no sample count —
the curves are deterministic and one is reused rather than sampled.

An unknown track or compound is a 404, never a default profile. An invalid or
out-of-range age is a 400. Unrecognised fields are rejected rather than
ignored.

## In the simulation

`simulate.py` loads the curves once in `load_track()`, before the first of the
10,000 races, and reads them from an in-memory table thereafter — no HTTP call
and no fitting inside the lap loop. Each lap adds the total `D(a)` for that
tyre's current age; it does not accumulate on top of the previous lap.

`ENFORCE_STINT_CAP` is load-bearing rather than a safety net. The curves are
only fitted over ages that were actually run, so it is what keeps the
simulation inside the range the model can answer for.

## Known limits

- Where SOFT cannot be measured, or measures kinder than MEDIUM at the same
  circuit, it takes MEDIUM's wear curve and keeps a much shorter cap. Combined
  with the soft's pace offset this can leave MEDIUM strictly dominated at that
  circuit — see the note in `tyre_curve_report.csv`.
- `tau` and `gamma` assume stints end for tyre reasons. Undercuts, safety cars
  and traffic all end stints too, and the fitted cliff absorbs some of that.
- Cars running in traffic show flatter stints because they were never pushing.
  That needs gap-to-car-ahead and is not fixed here.
