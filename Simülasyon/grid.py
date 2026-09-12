"""
F1 Prediction Simulation - grid.py
Measures what a grid slot is worth, in seconds, and how much the start scrambles it.

Why the simulation needs this
-----------------------------
Until now every car started from zero, so lap 1 order was pure pace: the fastest
car was already leading and never had to pass anyone. At Zandvoort that erases
the whole point of an overtaking model. Grid position is known before the race,
so using it is not leakage - qualifying happens first.

What it measures
----------------
grid_gap        seconds per grid slot at the end of lap 1, from a regression of
                lap-1 cumulative time on grid position
start_sigma     how far drivers move from their grid slot on lap 1, in positions
grid_to_finish  correlation between starting and finishing position, the naive
                model the roadmap uses as a benchmark in v0.9

Outputs:
    data/grid_by_race.csv
    data/grid_stats.csv
"""

import os

import numpy as np
import pandas as pd

from Simülasyon.dataset import load_laps, load_results, iter_races

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]

MIN_DRIVERS = 12
MAX_LAP1_GAP = 60.0        # ignore cars that hit trouble on the opening lap

# --- per race ---------------------------------------------------------------


def lap1_gaps(race_df):
    """Cumulative time at the end of lap 1, as a gap to the leader."""
    lap1 = race_df[race_df['LapNumber'] == 1][['Driver', 'LapTime_s']]
    lap1 = lap1[lap1['LapTime_s'].notna()]
    if lap1.empty:
        return None
    lap1 = lap1.copy()
    lap1['gap'] = lap1['LapTime_s'] - lap1['LapTime_s'].min()
    return lap1[lap1['gap'] <= MAX_LAP1_GAP]


def lap1_order(race_df):
    """Running order at the end of lap 1, from Position when it is available."""
    lap1 = race_df[race_df['LapNumber'] == 1][['Driver', 'Position']]
    lap1 = lap1[lap1['Position'].notna()]
    return lap1 if not lap1.empty else None


def analyse_race(race_df, results_df):
    grid = results_df[['Driver', 'GridPosition', 'Position']].copy()
    grid = grid[grid['GridPosition'].notna() & (grid['GridPosition'] > 0)]
    if len(grid) < MIN_DRIVERS:
        return None

    gaps = lap1_gaps(race_df)
    if gaps is None or len(gaps) < MIN_DRIVERS:
        return None

    merged = grid.merge(gaps, on='Driver', how='inner')
    if len(merged) < MIN_DRIVERS:
        return None

    # seconds per grid slot: slope of lap-1 gap on grid position
    x = merged['GridPosition'].to_numpy(dtype=float)
    y = merged['gap'].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    order = lap1_order(race_df)
    start_sigma = np.nan
    if order is not None:
        m2 = grid.merge(order, on='Driver', how='inner', suffixes=('', '_lap1'))
        if len(m2) >= MIN_DRIVERS:
            moved = m2['Position_lap1'].astype(float) - m2['GridPosition'].astype(float)
            start_sigma = float(moved.std())

    finished = grid[grid['Position'].notna()]
    corr = (float(finished['GridPosition'].corr(finished['Position']))
            if len(finished) >= MIN_DRIVERS else np.nan)
    mae = (float((finished['Position'] - finished['GridPosition']).abs().mean())
           if len(finished) >= MIN_DRIVERS else np.nan)

    return {
        'grid_gap': float(slope),
        'lap1_intercept': float(intercept),
        'start_sigma': start_sigma,
        'grid_finish_corr': corr,
        'grid_finish_mae': mae,
        'n_drivers': len(merged),
    }


# --- main -------------------------------------------------------------------


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    laps = load_laps(SEASONS)
    results = load_results(SEASONS)
    print(f'Races in laps    : {laps.groupby(["Season", "Race"]).ngroups}')
    print(f'Results entries  : {len(results)}\n')

    rows = []
    for season, name, race_laps, race_df in iter_races(laps):
        res = results[(results['Season'] == season) & (results['Race'] == name)]
        if res.empty:
            continue
        out = analyse_race(race_df, res)
        if out is None:
            continue
        out.update({'Season': season, 'Race': name})
        rows.append(out)
        print(f"  {season} {name:32s} grid_gap {out['grid_gap']:5.2f} s/slot  "
              f"start_sigma {out['start_sigma']:4.1f}  "
              f"grid->finish MAE {out['grid_finish_mae']:4.1f}")

    if not rows:
        print('Nothing collected.')
        return

    hist = pd.DataFrame(rows)

    print('\n===== field-wide =====')
    print(f"grid_gap        : {hist['grid_gap'].median():.2f} s per grid slot")
    print(f"start_sigma     : {hist['start_sigma'].median():.2f} positions")
    print(f"grid->finish r  : {hist['grid_finish_corr'].median():.2f}")
    print(f"grid->finish MAE: {hist['grid_finish_mae'].median():.2f} positions")
    print('\nThe MAE above is the naive benchmark from v0.9: a model that cannot')
    print('beat "everyone finishes where they started" is not adding anything.')

    tracks = (hist.groupby('Race')
                  .agg(grid_gap=('grid_gap', 'median'),
                       start_sigma=('start_sigma', 'median'),
                       grid_finish_corr=('grid_finish_corr', 'median'),
                       grid_finish_mae=('grid_finish_mae', 'median'),
                       n_seasons=('grid_gap', 'size'))
                  .reset_index()
                  .sort_values('grid_finish_corr', ascending=False))

    print('\n===== per track (high corr = grid decides the race) =====')
    print(tracks.round(2).to_string(index=False))

    hist.to_csv(os.path.join(DATA_DIR, 'grid_by_race.csv'), index=False)
    tracks.to_csv(os.path.join(DATA_DIR, 'grid_stats.csv'), index=False)
    print(f'\nSaved 2 files to {DATA_DIR}')


if __name__ == '__main__':
    main()