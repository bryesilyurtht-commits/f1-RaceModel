"""
F1 Prediction Simulation - v2.6 - quali_history.py
Historical qualifying, which the archive did not have.

Why this exists
---------------
The model's driver pace is a delta to the pole time. The committed archive
holds race laps and classified results for 2018-2025, and qualifying for 2026
alone - so a backtest of a 2024 race had no pole to measure a delta against.

The two wrong ways out were both available and both rejected. Taking the
fastest lap of the race as the reference reads the race being predicted, which
is the exact leak v2.6 exists to rule out. Dropping the reference and scoring
pace in absolute seconds compares Monaco against Monza. So qualifying is
downloaded instead, once, and committed.

What it is allowed to be
------------------------
Qualifying happens before the race. Reading it at a cutoff set after qualifying
is not leakage - it is the definition of the cutoff. What is downloaded here is
the session result and nothing from Sunday.

What it cannot claim
--------------------
This is today's copy of a 2024 session, not the copy that existed that weekend.
Timing data gets corrected, and a session re-published after a stewards'
decision comes back in its corrected form. The backtest is a reconstruction
from the present archive, not a replay of what was knowable at the time, and
`pipeline.py` says the same thing about every other file here.

    python -m Simülasyon.data_prep.quali_history            fetch what is missing
    python -m Simülasyon.data_prep.quali_history --all      refetch everything
"""

import argparse
import os
import warnings

import pandas as pd

# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
OUTPUT = os.path.join(DATA_DIR, 'quali_history.csv')

COLUMNS = ['Season', 'Round', 'Race', 'Driver', 'Team', 'quali_pos',
           'quali_time', 'pole_time']


def _best_lap(row):
    """
    A driver's best qualifying lap, in seconds.

    Q3 if they reached it, else Q2, else Q1. A driver knocked out in Q1 has no
    Q3 time, and treating that as missing pace rather than as a slower lap
    would drop the back of the grid out of the table entirely.
    """
    for session in ('Q3', 'Q2', 'Q1'):
        value = row.get(session)
        if pd.notna(value):
            seconds = pd.to_timedelta(value).total_seconds()
            if seconds > 0:
                return seconds
    return None


def fetch_session(season, event):
    """One qualifying session, or None with a reason."""
    import fastf1

    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(season, event, 'Q')
    session.load(telemetry=False, weather=False, messages=False)
    results = session.results
    if results is None or results.empty:
        return None, 'no results in session'

    rows = []
    for _, row in results.iterrows():
        best = _best_lap(row)
        rows.append(dict(Driver=str(row['Abbreviation']),
                         Team=str(row.get('TeamName', '')),
                         quali_pos=float(row['Position'])
                                   if pd.notna(row['Position']) else None,
                         quali_time=best))
    frame = pd.DataFrame(rows)
    timed = frame['quali_time'].dropna()
    if timed.empty:
        return None, 'no timed laps'
    frame['pole_time'] = float(timed.min())
    return frame, None


def load():
    """What has been fetched so far, or an empty table."""
    if not os.path.exists(OUTPUT):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(OUTPUT)


def fetch(races, existing=None, verbose=True):
    """
    Fetch each (season, round, event name) that is not already on file.

    Failures are collected and reported rather than raising: one session
    missing from the source should not cost the other fifty-one, and a race
    with no qualifying on record has to be visible in the backtest's excluded
    count rather than silently absent.
    """
    warnings.filterwarnings('ignore')
    existing = load() if existing is None else existing
    have = set(zip(existing.get('Season', []), existing.get('Round', [])))

    collected, failures = [], []
    for i, (season, rnd, event) in enumerate(races, 1):
        if (season, rnd) in have:
            continue
        try:
            frame, why = fetch_session(season, event)
        except Exception as exc:                         # noqa: BLE001
            frame, why = None, f'{type(exc).__name__}: {exc}'
        if frame is None:
            failures.append((season, rnd, event, why))
            if verbose:
                print(f'  [{i}/{len(races)}] {season} {event:34s} FAILED  {why}')
            continue
        frame['Season'], frame['Round'], frame['Race'] = season, rnd, event
        collected.append(frame[COLUMNS])
        if verbose:
            print(f'  [{i}/{len(races)}] {season} {event:34s} '
                  f'{len(frame):2d} drivers, pole {frame["pole_time"].iloc[0]:.3f}')

    if collected:
        existing = pd.concat([existing] + collected, ignore_index=True)
        existing = existing.sort_values(['Season', 'Round', 'quali_pos'])
        existing.to_csv(OUTPUT, index=False)
    return existing, failures


def evaluation_races():
    """The races the backtest scores, which is what needs qualifying."""
    from Simülasyon import backtest as B

    frame = B.universe()
    included = frame[frame.state == 'included']
    return list(included[['Season', 'Round', 'Race']].itertuples(index=False,
                                                                name=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true',
                        help='refetch races already on file')
    args = parser.parse_args()

    races = evaluation_races()
    print(f'{len(races)} races in the evaluation set')
    existing = pd.DataFrame(columns=COLUMNS) if args.all else load()
    if len(existing):
        print(f'{existing.groupby(["Season", "Round"]).ngroups} already on file')

    table, failures = fetch(races, existing)
    print(f'\n{table.groupby(["Season", "Round"]).ngroups} races, '
          f'{len(table)} driver rows -> {OUTPUT}')
    if failures:
        print(f'{len(failures)} failed:')
        for season, rnd, event, why in failures:
            print(f'  {season} R{rnd} {event}: {why}')


if __name__ == '__main__':
    main()
