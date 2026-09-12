"""
F1 Prediction Simulation - v0.2 - fetch.py
Downloads 2026 race laps, pole times and the target race's starting grid.

Two fixes over the first version.

The target race is matched on aliases, not one string. FastF1 calls the event
"Dutch Grand Prix" while the country is "Netherlands", so a single substring
test silently failed and the target race's own laps were being downloaded into
the training data. That is leakage: the model was fed the race it predicts.

The grid is now fetched. GridPosition comes from the race classification, so it
already includes penalties, unlike raw qualifying order. Qualifying runs before
the race, so using it is legitimate.

Outputs:
    data/f1_2026_laps.csv     raw laps, target race excluded
    data/f1_2026_poles.csv    pole lap time per race, target included
    data/f1_2026_grid.csv     starting grid of the target race
    data/f1_2026_quali.csv    qualifying order of every race, target included
"""

import os
import warnings
from datetime import datetime

import fastf1
import numpy as np
import pandas as pd

from Simülasyon.target_race import TRACK_ALIASES as TARGET_ALIASES

warnings.filterwarnings('ignore')

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASON = 2026

# which race is withheld from training and has its grid downloaded instead -
# set in target_race.py; change TARGET_RACE there, not here.

LAP_COLUMNS = [
    'Driver', 'Team', 'LapNumber', 'LapTime', 'Compound', 'TyreLife',
    'Stint', 'TrackStatus', 'Position', 'PitInTime', 'PitOutTime',
]

RESULT_COLUMNS = [
    'Abbreviation', 'TeamName', 'GridPosition', 'Position',
    'ClassifiedPosition', 'Status', 'Points',
]

# --- helpers ----------------------------------------------------------------


def is_target(event_name, country=''):
    text = f'{event_name} {country}'.lower()
    return any(alias in text for alias in TARGET_ALIASES)


def load_session(season, event_name, kind):
    try:
        s = fastf1.get_session(season, event_name, kind)
        s.load(telemetry=False, weather=(kind == 'R'), messages=False)
        return s
    except Exception as exc:
        print(f'    [{kind}] skipped: {type(exc).__name__} - {exc}')
        return None


def get_pole_time(session):
    if session is None:
        return None, None
    laps = session.laps
    if laps is None or laps.empty:
        return None, None

    laps = laps.copy()
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
    laps = laps[laps['LapTime_s'].notna()]
    if laps.empty:
        return None, None

    pole = laps.loc[laps['LapTime_s'].idxmin()]
    return float(pole['LapTime_s']), str(pole['Driver'])


def get_quali_order(session):
    """
    Qualifying classification. Falls back to ranking by fastest lap when the
    session has no results table.

    Qualifying order is a cleaner pace signal than the grid: it carries no
    penalties, and no strategy or first-lap luck the way a race result does.
    """
    if session is None:
        return None

    res = getattr(session, 'results', None)
    if res is not None and not res.empty and 'Position' in res.columns:
        cols = [c for c in ['Abbreviation', 'TeamName', 'Position'] if c in res.columns]
        q = res[cols].copy().rename(columns={'Abbreviation': 'Driver',
                                             'TeamName': 'Team',
                                             'Position': 'quali_pos'})
        q['quali_pos'] = pd.to_numeric(q['quali_pos'], errors='coerce')
        q = q[q['quali_pos'].notna()]
        if not q.empty:
            return q.sort_values('quali_pos').reset_index(drop=True)

    laps = session.laps
    if laps is None or laps.empty:
        return None
    laps = laps.copy()
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
    best = (laps[laps['LapTime_s'].notna()]
            .groupby('Driver')['LapTime_s'].min()
            .sort_values())
    if best.empty:
        return None
    return pd.DataFrame({'Driver': best.index,
                         'quali_pos': np.arange(1, len(best) + 1)})


def get_grid(session):
    """Starting grid from the race classification, penalties included."""
    if session is None:
        return None
    res = getattr(session, 'results', None)
    if res is None or res.empty or 'GridPosition' not in res.columns:
        return None

    cols = [c for c in ['Abbreviation', 'TeamName', 'GridPosition', 'Position']
            if c in res.columns]
    grid = res[cols].copy().rename(columns={'Abbreviation': 'Driver',
                                            'TeamName': 'Team'})
    grid['GridPosition'] = pd.to_numeric(grid['GridPosition'], errors='coerce')
    grid = grid[grid['GridPosition'].notna() & (grid['GridPosition'] > 0)]
    return grid.sort_values('GridPosition').reset_index(drop=True)


def get_results(session, event_name, rnd):
    """
    Race classification: who finished where, and who did not finish at all.

    The laps alone cannot answer the second question. A car that stops on lap
    30 and a car that is two laps down both end the race with fewer laps than
    the winner, and only Status separates them. Anything that wants to treat a
    retirement differently from a slow finish needs this table.

    The target race is never passed in here - its result is the thing being
    predicted.
    """
    if session is None:
        return None
    res = getattr(session, 'results', None)
    if res is None or res.empty:
        return None

    have = [c for c in RESULT_COLUMNS if c in res.columns]
    if 'Abbreviation' not in have:
        return None

    out = res[have].copy().rename(columns={'Abbreviation': 'Driver',
                                           'TeamName': 'Team'})
    for col in ('GridPosition', 'Position', 'Points'):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    out['Race'] = event_name
    out['Round'] = rnd
    return out


def get_race_laps(session, event_name, rnd):
    if session is None:
        return None
    laps = session.laps
    if laps is None or laps.empty:
        return None

    missing = [c for c in LAP_COLUMNS if c not in laps.columns]
    if missing:
        print(f'    [R] missing columns: {missing}')
        return None

    out = laps[LAP_COLUMNS].copy()
    out['Race'] = event_name
    out['Round'] = rnd
    return out


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    schedule = fastf1.get_event_schedule(SEASON, include_testing=False)
    now = pd.Timestamp(datetime.now())

    all_laps, pole_rows, quali_rows, result_rows = [], [], [], []
    grid_df = None
    skipped_future = 0

    for _, event in schedule.iterrows():
        name = event['EventName']
        rnd = event['RoundNumber']
        country = str(event.get('Country', ''))
        event_date = pd.Timestamp(event['EventDate'])
        target = is_target(name, country)

        if event_date > now:
            skipped_future += 1
            continue

        print(f'Round {rnd:2d} - {name}{"  [TARGET]" if target else ""}')

        quali = load_session(SEASON, name, 'Q')
        pole_time, pole_driver = get_pole_time(quali)

        q_order = get_quali_order(quali)
        if q_order is not None:
            q_order = q_order.copy()
            q_order['Race'] = name
            q_order['Round'] = rnd
            q_order['is_target'] = target
            quali_rows.append(q_order)
        if pole_time is not None:
            pole_rows.append({
                'Race': name, 'Round': rnd,
                'pole_time': round(pole_time, 3),
                'pole_driver': pole_driver,
                'is_target': target,
            })
            print(f'    pole: {pole_driver} {pole_time:.3f} s')
        else:
            print('    pole: not available')

        if target:
            # the grid is pre-race information, the laps are not
            race = load_session(SEASON, name, 'R')
            grid_df = get_grid(race)
            if grid_df is not None:
                order = ', '.join(grid_df['Driver'].head(10))
                print(f'    grid: {len(grid_df)} cars   front: {order}')
            else:
                print('    grid: not available')
            print('    race laps intentionally not downloaded')
            continue

        race = load_session(SEASON, name, 'R')
        laps = get_race_laps(race, name, rnd)
        if laps is not None:
            all_laps.append(laps)
            print(f'    laps: {len(laps)}')

        result = get_results(race, name, rnd)
        if result is not None:
            result_rows.append(result)
            # ClassifiedPosition, not Status: this feed marks a lapped car
            # "Lapped" rather than "+1 Lap", and a lapped car is a finisher.
            # Reading Status for this counts half the field as retired.
            cp = result.get('ClassifiedPosition')
            out = int((~cp.astype(str).str.fullmatch(r'\d+')).sum()) \
                if cp is not None else 0
            print(f'    result: {len(result)} entries, {out} unclassified')

    print(f'\nFuture events skipped: {skipped_future}')

    if not all_laps:
        print('No race data collected.')
        return

    laps_df = pd.concat(all_laps, ignore_index=True)
    poles_df = pd.DataFrame(pole_rows).sort_values('Round')

    # leakage check: the predicted race must not be in the training laps
    leaked = [r for r in laps_df['Race'].unique() if is_target(r)]
    if leaked:
        print(f'\n! LEAKAGE: target race still present in laps: {leaked}')
    else:
        print('\nLeakage check passed: target race is not in the lap data.')

    laps_path = os.path.join(DATA_DIR, f'f1_{SEASON}_laps.csv')
    poles_path = os.path.join(DATA_DIR, f'f1_{SEASON}_poles.csv')
    laps_df.to_csv(laps_path, index=False)
    poles_df.to_csv(poles_path, index=False)

    print(f'Races: {laps_df["Race"].nunique()}   Laps: {len(laps_df)}')
    print(f'Saved: {laps_path}')
    print(f'Saved: {poles_path}')

    if grid_df is not None:
        grid_path = os.path.join(DATA_DIR, f'f1_{SEASON}_grid.csv')
        grid_df.to_csv(grid_path, index=False)
        print(f'Saved: {grid_path}')

    if result_rows:
        results_df = pd.concat(result_rows, ignore_index=True)
        results_path = os.path.join(DATA_DIR, f'f1_{SEASON}_results.csv')
        results_df.to_csv(results_path, index=False)
        leaked = [r for r in results_df['Race'].unique() if is_target(r)]
        if leaked:
            print(f'\n! LEAKAGE: target race is in the results: {leaked}')
        print(f'Saved: {results_path}  '
              f'({results_df["Race"].nunique()} races, {len(results_df)} entries)')

    if quali_rows:
        quali_df = pd.concat(quali_rows, ignore_index=True)
        quali_path = os.path.join(DATA_DIR, f'f1_{SEASON}_quali.csv')
        quali_df.to_csv(quali_path, index=False)
        print(f'Saved: {quali_path}  '
              f'({quali_df["Race"].nunique()} sessions, {len(quali_df)} entries)')


if __name__ == '__main__':
    main()