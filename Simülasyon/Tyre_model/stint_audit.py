"""
F1 Prediction Simulation - stint_audit.py
v1.1-A / A2: sample audit. Decides whether a track x compound curve can be
measured at all, before any curve fitting is attempted.

Reads data/stint_index.csv (written by stint_table.py). Fits nothing.

Four questions:
    1. How many track x compound cells have enough stints?
    2. Did the degradation regime shift across 2022-2025?
    3. Are the long stints real or red-flag artefacts?
    4. Is SOFT - the compound the whole exercise is about - measurable?

Usage:
    python stint_audit.py
"""

import os

import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 300)

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

MIN_STINTS_PER_CELL = 8       # below this a track x compound cell is unmeasurable
MIN_LONG_STINTS = 3           # cells also need a few long stints, else no cliff
LONG_STINT_LAPS = 15          # what counts as long enough to reach a cliff

SUSPECT_STINT_LEN = 45        # anything past this gets inspected by hand


def load_index():
    path = os.path.join(DATA_DIR, 'stint_index.csv')
    df = pd.read_csv(path)
    return df[df['usable']].copy()


def cell_table(idx):
    """One row per track x compound, with the counts that decide measurability."""
    idx = idx.copy()
    idx['is_long'] = idx['n_laps_clean'] >= LONG_STINT_LAPS

    cells = (idx.groupby(['Race', 'Compound'])
                .agg(n_stints=('Stint', 'size'),
                     n_seasons=('Season', 'nunique'),
                     n_long=('is_long', 'sum'),
                     n_to_flag=('runs_to_flag', 'sum'),
                     median_laps=('n_laps_clean', 'median'),
                     max_laps=('n_laps_clean', 'max'))
                .reset_index())

    cells['measurable'] = ((cells['n_stints'] >= MIN_STINTS_PER_CELL)
                           & (cells['n_long'] >= MIN_LONG_STINTS))
    return cells


def main():
    idx = load_index()
    print(f'Usable stints : {len(idx)}')
    print(f'Tracks        : {idx["Race"].nunique()}')
    print(f'Seasons       : {sorted(idx["Season"].unique())}\n')

    # --- 1. cell coverage ---------------------------------------------------
    cells = cell_table(idx)
    tracks = idx['Race'].nunique()
    expected = tracks * 3
    present = len(cells)
    ok = int(cells['measurable'].sum())

    print('=' * 70)
    print('1. CELL COVERAGE')
    print('=' * 70)
    print(f'Cells expected (tracks x 3) : {expected}')
    print(f'Cells with any data         : {present}')
    print(f'Cells measurable            : {ok}  ({ok / expected:.1%})')
    print(f'  rule: >= {MIN_STINTS_PER_CELL} stints and '
          f'>= {MIN_LONG_STINTS} stints of >= {LONG_STINT_LAPS} clean laps\n')

    print('--- measurable cells per compound ---')
    print(cells.groupby('Compound')
              .agg(cells=('measurable', 'size'),
                   measurable=('measurable', 'sum'),
                   median_stints=('n_stints', 'median'))
              .to_string())

    print('\n--- thin cells (present but not measurable) ---')
    thin = cells[~cells['measurable']].sort_values(['Compound', 'n_stints'])
    if thin.empty:
        print('  none')
    else:
        print(thin[['Race', 'Compound', 'n_stints', 'n_long',
                    'median_laps']].to_string(index=False))

    print('\n--- tracks with no row at all for a compound ---')
    have = set(map(tuple, cells[['Race', 'Compound']].values))
    missing = [(t, c) for t in sorted(idx['Race'].unique())
               for c in ['SOFT', 'MEDIUM', 'HARD'] if (t, c) not in have]
    if missing:
        for t, c in missing:
            print(f'  {t:34s} {c}')
    else:
        print('  none')

    # --- 2. season regime ---------------------------------------------------
    print('\n' + '=' * 70)
    print('2. SEASON REGIME')
    print('=' * 70)
    print('Stint lengths grew 16 -> 22 across the window. If that is a tyre')
    print('regime change, pooling all four seasons blends two different cars.\n')

    print('--- stint length by season and compound ---')
    piv = idx.pivot_table(index='Season', columns='Compound',
                          values='n_laps_clean', aggfunc='median')
    print(piv.to_string())

    print('\n--- compound share of stints by season ---')
    share = (idx.groupby(['Season', 'Compound']).size()
                .unstack(fill_value=0))
    print(share.div(share.sum(axis=1), axis=0).round(3).to_string())

    print('\n--- stints per race by season (pit-stop count proxy) ---')
    per_race = (idx.groupby(['Season', 'Race']).size()
                   .groupby('Season').agg(['mean', 'median']).round(2))
    print(per_race.to_string())

    # tracks present in all four seasons: the only fair like-for-like set
    by_track_season = idx.groupby('Race')['Season'].nunique()
    stable = by_track_season[by_track_season == 4].index
    print(f'\n--- tracks run in all 4 seasons: {len(stable)} ---')
    sub = idx[idx['Race'].isin(stable)]
    print(sub.pivot_table(index='Season', columns='Compound',
                          values='n_laps_clean', aggfunc='median').to_string())
    print('If the trend holds on this fixed set, it is a regime change and')
    print('not just a calendar that swapped tracks around.')

    # --- 3. suspect stints --------------------------------------------------
    print('\n' + '=' * 70)
    print('3. SUSPECT LONG STINTS')
    print('=' * 70)
    susp = idx[idx['stint_length'] >= SUSPECT_STINT_LEN].sort_values(
        'stint_length', ascending=False)
    print(f'Stints with raw length >= {SUSPECT_STINT_LEN}: {len(susp)}\n')
    if not susp.empty:
        cols = ['Season', 'Race', 'Driver', 'Compound', 'Stint',
                'stint_length', 'n_laps_clean', 'race_laps',
                'tyre_life_start', 'tyre_life_end']
        print(susp[cols].head(25).to_string(index=False))
        print('\nCheck: stint_length close to race_laps means a genuine')
        print('no-stop run, which only Monaco-like races allow. A stint')
        print('longer than race_laps is a red-flag artefact and must go.')
        bad = susp[susp['stint_length'] > susp['race_laps']]
        print(f'\nStints longer than their own race: {len(bad)}')
        if not bad.empty:
            print(bad[cols].to_string(index=False))

    # --- 4. soft compound ---------------------------------------------------
    print('\n' + '=' * 70)
    print('4. SOFT - THE COMPOUND THE FIX IS ABOUT')
    print('=' * 70)
    soft = idx[idx['Compound'] == 'SOFT']
    print(f'SOFT stints : {len(soft)}  '
          f'({len(soft) / len(idx):.1%} of all)')
    print(f'Tracks with any SOFT stint : {soft["Race"].nunique()} / {tracks}')
    soft_cells = cells[cells['Compound'] == 'SOFT']
    print(f'Measurable SOFT cells      : {int(soft_cells["measurable"].sum())} '
          f'/ {len(soft_cells)}\n')

    print('--- SOFT stint length distribution ---')
    print(soft['n_laps_clean'].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).round(1).to_string())
    print(f'\nSOFT stints reaching {LONG_STINT_LAPS}+ clean laps: '
          f'{(soft["n_laps_clean"] >= LONG_STINT_LAPS).sum()}')
    print('These are the only stints that can show where SOFT falls off.')
    print('If this number is small the SOFT cliff cannot be measured per')
    print('track and will have to be pooled.\n')

    print('--- top SOFT tracks by stint count ---')
    print(soft.groupby('Race')
              .agg(stints=('Stint', 'size'),
                   long=('n_laps_clean', lambda s: (s >= LONG_STINT_LAPS).sum()),
                   median_laps=('n_laps_clean', 'median'),
                   max_laps=('n_laps_clean', 'max'))
              .sort_values('stints', ascending=False)
              .head(15).to_string())

    out = os.path.join(DATA_DIR, 'stint_cells.csv')
    cells.to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()