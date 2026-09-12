"""
F1 Prediction Simulation - stint_table.py
v1.1-A / A1: builds the stint-level table that the cliff measurement runs on.

This does NOT fit anything. It produces two files:

    data/stint_laps.csv   lap level, cleaned, only laps belonging to a usable stint
    data/stint_index.csv  one row per stint, with metadata and quality flags

Why a separate cleaning pass instead of reusing clean_race as-is:
dataset.clean_race clips laps above median * 1.07. A tyre falling off the
cliff is exactly the lap that clipping removes, so the margin is widened here
and both versions are reported. Whether the wider margin lets junk in is a
question for A2, not a decision made silently now.

Usage:
    python stint_table.py
    python stint_table.py --margin 1.12
"""

import os
import sys

import numpy as np
import pandas as pd

from Simülasyon.dataset import load_laps, DRY_COMPOUNDS, TYRE_SEASONS

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# The tyre model reaches back to 2018; the rest of the project does not. See
# dataset.TYRE_SEASONS for why, and for how the pre-2019 compound names are
# mapped onto SOFT/MEDIUM/HARD.
#
# 2018-2021 ran 13-inch tyres and 2022 onwards run 18-inch, which is a real
# break in how a tyre wears. Every stint carries its era in `tyre_era` so the
# fit can weigh the two rather than pooling them without looking.
SEASONS = TYRE_SEASONS

STRICT_MARGIN = 1.07        # the project's existing clean_lap cutoff
CLIFF_MARGIN = 1.07         # wider, so cliff laps survive to be measured

MIN_CLEAN_LAPS = 6          # below this a stint carries no curve shape at all
MIN_COVERAGE = 0.60         # clean laps / raw stint length


def clean_for_cliff(race_df, margin):
    """
    Same rules as dataset.clean_race except for the outlier margin, and the
    per-driver minimum is dropped: here the unit of analysis is the stint,
    not the driver.
    """
    mask = (race_df['LapTime_s'].notna()
            & (race_df['TrackStatus'] == '1')
            & race_df['PitInTime_s'].isna()
            & race_df['PitOutTime_s'].isna()
            & (race_df['LapNumber'] > 1)
            & (race_df['TyreLife'] > 1)
            & race_df['Compound'].isin(DRY_COMPOUNDS))

    clean = race_df[mask].copy()
    if clean.empty:
        return clean

    cutoff = clean['LapTime_s'].median() * margin
    return clean[clean['LapTime_s'] <= cutoff]


def stint_geometry(race_df):
    """
    Raw stint extent per (Driver, Stint), measured before any cleaning.

    The real stint length has to come from the uncleaned laps: an in-lap or a
    safety car lap is still part of the stint even though it is unusable for
    pace. Using the cleaned laps here would make every stint look shorter than
    it was and push cliff_lap fractions around.
    """
    g = race_df.groupby(['Driver', 'Stint'])
    geo = g.agg(stint_first_lap=('LapNumber', 'min'),
                stint_last_lap=('LapNumber', 'max'),
                tyre_life_start=('TyreLife', 'min'),
                tyre_life_end=('TyreLife', 'max'),
                n_laps_raw=('LapNumber', 'size')).reset_index()
    geo['stint_length'] = geo['tyre_life_end'] - geo['tyre_life_start'] + 1
    return geo


def build_table(laps, margin):
    lap_frames, index_frames = [], []

    for (season, race), race_df in laps.groupby(['Season', 'Race'], sort=True):
        race_laps = int(race_df['race_laps'].iloc[0])
        geo = stint_geometry(race_df)

        clean = clean_for_cliff(race_df, margin)
        if clean.empty:
            continue

        base = clean['LapTime_s'].median()

        agg = (clean.groupby(['Driver', 'Stint'])
                    .agg(Team=('Team', lambda s: s.mode().iloc[0]),
                         Compound=('Compound', lambda s: s.mode().iloc[0]),
                         n_laps_clean=('LapTime_s', 'size'),
                         tyre_life_min=('TyreLife', 'min'),
                         tyre_life_max=('TyreLife', 'max'),
                         lap_time_min=('LapTime_s', 'min'),
                         lap_time_max=('LapTime_s', 'max'),
                         lap_time_median=('LapTime_s', 'median'))
                    .reset_index())

        idx = agg.merge(geo, on=['Driver', 'Stint'], how='left')
        idx['Season'] = season
        idx['Race'] = race
        idx['race_laps'] = race_laps
        idx['race_median_lap'] = base
        idx['tyre_era'] = (race_df['tyre_era'].iloc[0]
                           if 'tyre_era' in race_df.columns else '18inch')

        idx['tyre_span'] = idx['tyre_life_max'] - idx['tyre_life_min'] + 1
        idx['coverage'] = idx['n_laps_clean'] / idx['stint_length']

        # a stint running to the flag was never ended by a pit decision, so it
        # is the one most likely to have been pushed into the cliff
        idx['runs_to_flag'] = idx['stint_last_lap'] >= race_laps - 1
        idx['is_first_stint'] = idx['tyre_life_start'] <= 2

        idx['usable'] = ((idx['n_laps_clean'] >= MIN_CLEAN_LAPS)
                         & (idx['coverage'] >= MIN_COVERAGE)
                         & idx['Compound'].isin(DRY_COMPOUNDS))

        index_frames.append(idx)

        usable_keys = set(map(tuple, idx.loc[idx['usable'], ['Driver', 'Stint']].values))
        keep = clean.apply(lambda r: (r['Driver'], r['Stint']) in usable_keys, axis=1)
        lap_frames.append(clean[keep])

    if not index_frames:
        raise SystemExit('No stint survived. Check the lap cache.')

    stint_index = pd.concat(index_frames, ignore_index=True)
    stint_laps = pd.concat(lap_frames, ignore_index=True)
    return stint_index, stint_laps


def main():
    margin = CLIFF_MARGIN
    if '--margin' in sys.argv:
        margin = float(sys.argv[sys.argv.index('--margin') + 1])

    os.makedirs(DATA_DIR, exist_ok=True)

    print('Loading laps (first run builds the cache and takes several minutes,')
    print('later runs read the CSV in seconds)...\n')
    laps = load_laps(seasons=SEASONS)
    print(f'Seasons : {sorted(laps["Season"].unique())}')
    print(f'Races   : {laps.groupby(["Season", "Race"]).ngroups}')
    print(f'Laps    : {len(laps)}\n')

    # how much the wider margin actually changes things
    strict_n = cliff_n = 0
    for _, race_df in laps.groupby(['Season', 'Race']):
        strict_n += len(clean_for_cliff(race_df, STRICT_MARGIN))
        cliff_n += len(clean_for_cliff(race_df, margin))
    extra = cliff_n - strict_n
    print(f'Outlier margin {STRICT_MARGIN}: {strict_n} laps')
    print(f'Outlier margin {margin}: {cliff_n} laps')
    print(f'Recovered by widening: {extra} laps ({extra / max(strict_n, 1):.1%})\n')

    stint_index, stint_laps = build_table(laps, margin)

    usable = stint_index[stint_index['usable']]
    print(f'Stints total  : {len(stint_index)}')
    print(f'Stints usable : {len(usable)} '
          f'({len(usable) / len(stint_index):.1%})')
    print(f'Laps in usable stints : {len(stint_laps)}\n')

    print('--- why stints were dropped ---')
    rej = stint_index[~stint_index['usable']]
    print(f'  too few clean laps (<{MIN_CLEAN_LAPS}) : '
          f'{(rej["n_laps_clean"] < MIN_CLEAN_LAPS).sum()}')
    print(f'  coverage below {MIN_COVERAGE}          : '
          f'{(rej["coverage"] < MIN_COVERAGE).sum()}')

    print('\n--- usable stints per season ---')
    print(usable.groupby('Season')
               .agg(stints=('Stint', 'size'),
                    races=('Race', 'nunique'),
                    median_len=('n_laps_clean', 'median'))
               .to_string())

    print('\n--- usable stints per compound ---')
    print(usable.groupby('Compound')
               .agg(stints=('Stint', 'size'),
                    median_clean_laps=('n_laps_clean', 'median'),
                    median_stint_len=('stint_length', 'median'),
                    max_stint_len=('stint_length', 'max'))
               .to_string())

    print('\n--- stint length distribution (clean laps) ---')
    print(usable['n_laps_clean'].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())

    print('\n--- stints running to the flag (cliff most visible here) ---')
    print(usable.groupby('Compound')['runs_to_flag']
               .agg(['sum', 'mean']).round(3).to_string())

    ipath = os.path.join(DATA_DIR, 'stint_index.csv')
    lpath = os.path.join(DATA_DIR, 'stint_laps.csv')
    stint_index.to_csv(ipath, index=False)
    stint_laps.to_csv(lpath, index=False)
    print(f'\nSaved: {ipath}')
    print(f'Saved: {lpath}')


if __name__ == '__main__':
    main()