"""
F1 Prediction Simulation - deg.py
Measures tyre degradation per driver, team and compound.

Method
------
1. One robust linear fit per stint: LapTime_s ~ TyreLife (Theil-Sen).
   TyreLife is the regressor, so laps dropped by clean.py leave harmless gaps.
2. Fuel correction: the burn rate is added back before the slope is read,
   otherwise deg is measured too low.
3. Stints shorter than MIN_LAPS_PER_STINT are dropped. Below ~12 laps the fits
   are noise: more than half of them come out negative.
4. Within-race normalisation: deg_rel = stint deg_pct / that race's median.
   Track abrasiveness varies far more than drivers do (Barcelona ~0.15 vs
   Canada ~0.00), so pooling raw slopes across tracks measures the calendar,
   not the driver.
5. deg_index = driver median deg_rel. Below 1.0 = kinder on tyres than the field.

Known bias: cars running in traffic show flatter slopes because they were never
pushing. Needs gap-to-car-ahead to fix properly - deferred.

Outputs:
    data/deg_by_stint.csv
    data/deg_by_driver_compound.csv
    data/deg_by_driver.csv
    data/deg_by_team.csv
"""

import os

import numpy as np
import pandas as pd

from Simülasyon.fuel_effect import get_track_params

try:
    from scipy.stats import theilslopes
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASON = 2026

MIN_LAPS_PER_STINT = 12     # below this the slope is noise
MIN_TYRELIFE_SPAN = 8       # tyre age range the stint must cover
MAX_GAP_RATIO = 0.5         # at least half the stint's laps must survive cleaning
MIN_STINTS_PER_DRIVER = 8   # drivers with fewer usable stints are excluded
DRY_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD'}

# --- fitting ----------------------------------------------------------------


def fit_slope(x, y):
    """Robust slope of y on x. Theil-Sen when available, least squares otherwise."""
    if HAVE_SCIPY:
        slope, intercept, _, _ = theilslopes(y, x)
        return float(slope), float(intercept)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def fit_stints(clean):
    """One row per usable stint, with the fuel-corrected degradation slope."""
    rows = []

    race_median_lap = clean.groupby('Race')['LapTime_s'].median().to_dict()
    fuel_map = {r: get_track_params(r)['fuel_effect'] for r in clean['Race'].unique()}

    for (race, driver, stint), grp in clean.groupby(['Race', 'Driver', 'Stint']):
        compound = grp['Compound'].mode().iloc[0]
        if compound not in DRY_COMPOUNDS:
            continue
        if len(grp) < MIN_LAPS_PER_STINT:
            continue

        x = grp['TyreLife'].to_numpy(dtype=float)
        span = x.max() - x.min()
        if span < MIN_TYRELIFE_SPAN:
            continue
        if len(grp) / (span + 1) < MAX_GAP_RATIO:
            continue

        # remove the fuel gain so the slope is pure tyre ageing
        y = grp['LapTime_s'].to_numpy(dtype=float)
        lap_no = grp['LapNumber'].to_numpy(dtype=float)
        y_corrected = y + fuel_map[race] * (lap_no - 1)

        slope, _ = fit_slope(x, y_corrected)
        base = race_median_lap.get(race, np.nan)

        rows.append({
            'Race': race,
            'Driver': driver,
            'Team': grp['Team'].mode().iloc[0],
            'Stint': stint,
            'Compound': compound,
            'n_laps': len(grp),
            'tyre_span': int(span),
            'deg_abs': slope,                 # s per lap of tyre age
            'deg_pct': slope / base * 100,    # % of a lap, per lap of tyre age
        })

    stints = pd.DataFrame(rows)
    if stints.empty:
        return stints

    # within-race normalisation: strips the track out of the number
    race_med_deg = stints.groupby('Race')['deg_pct'].transform('median')
    stints['race_median_deg_pct'] = race_med_deg
    stints['deg_rel'] = stints['deg_pct'] / race_med_deg
    return stints


# --- aggregation ------------------------------------------------------------


def aggregate(stints):
    counts = stints.groupby('Driver').size()
    keep = counts[counts >= MIN_STINTS_PER_DRIVER].index
    dropped = sorted(set(counts.index) - set(keep))
    stints = stints[stints['Driver'].isin(keep)]

    by_driver_compound = (stints.groupby(['Driver', 'Compound'])
                                .agg(deg_rel=('deg_rel', 'median'),
                                     deg_pct=('deg_pct', 'median'),
                                     deg_abs=('deg_abs', 'median'),
                                     n_stints=('deg_rel', 'size'))
                                .reset_index())

    by_driver = (stints.groupby('Driver')
                       .agg(Team=('Team', lambda s: s.mode().iloc[0]),
                            deg_index=('deg_rel', 'median'),
                            deg_pct=('deg_pct', 'median'),
                            deg_abs=('deg_abs', 'median'),
                            n_stints=('deg_rel', 'size'),
                            n_races=('Race', 'nunique'))
                       .reset_index()
                       .sort_values('deg_index')
                       .reset_index(drop=True))

    by_team = (stints.groupby('Team')
                     .agg(deg_index=('deg_rel', 'median'),
                          deg_pct=('deg_pct', 'median'),
                          deg_abs=('deg_abs', 'median'),
                          n_stints=('deg_rel', 'size'))
                     .reset_index()
                     .sort_values('deg_index')
                     .reset_index(drop=True))

    # driver effect inside the team: same car, so traffic bias partly cancels
    team_index = by_team.set_index('Team')['deg_index'].to_dict()
    by_driver['vs_team'] = by_driver['deg_index'] / by_driver['Team'].map(team_index)

    return by_driver_compound, by_driver, by_team, stints, dropped


def main():
    clean_path = os.path.join(DATA_DIR, f'f1_{SEASON}_laps_clean.csv')
    df = pd.read_csv(clean_path)

    clean = df[df['clean_lap'].astype(str).str.lower() == 'true'].copy()
    if clean.empty:                       # boolean dtype instead of string
        clean = df[df['clean_lap'] == True].copy()

    clean['Compound'] = clean['Compound'].astype(str).str.upper()
    clean = clean[clean['LapTime_s'].notna() & clean['TyreLife'].notna()]

    print(f'Clean laps  : {len(clean)}')
    print(f'Regression  : {"Theil-Sen" if HAVE_SCIPY else "least squares (scipy missing)"}')
    print(f'Stint filter: >= {MIN_LAPS_PER_STINT} laps, >= {MIN_TYRELIFE_SPAN} tyre-life span\n')

    stints = fit_stints(clean)
    if stints.empty:
        print('No stint passed the filters. Loosen MIN_LAPS_PER_STINT.')
        return

    by_dc, by_driver, by_team, kept, dropped = aggregate(stints)

    print(f'Usable stints : {len(stints)}')
    print(f'Negative fits : {(stints["deg_abs"] < 0).mean():.1%}')
    if dropped:
        print(f'Dropped (<{MIN_STINTS_PER_DRIVER} stints): {", ".join(dropped)}')

    print('\n--- per-race median deg (raw, before normalisation) ---')
    print(stints.groupby('Race')['deg_pct'].agg(['median', 'size']).round(4).to_string())

    print('\n--- compound medians (normalised) ---')
    print(stints.groupby('Compound')
                .agg(deg_rel=('deg_rel', 'median'),
                     deg_pct=('deg_pct', 'median'),
                     n_stints=('deg_rel', 'size'))
                .round(4).to_string())

    print('\n--- by driver (deg_index < 1 = kinder on tyres) ---')
    print(by_driver.round(4).to_string(index=False))

    print('\n--- by team ---')
    print(by_team.round(4).to_string(index=False))

    stints.to_csv(os.path.join(DATA_DIR, 'deg_by_stint.csv'), index=False)
    by_dc.to_csv(os.path.join(DATA_DIR, 'deg_by_driver_compound.csv'), index=False)
    by_driver.to_csv(os.path.join(DATA_DIR, 'deg_by_driver.csv'), index=False)
    by_team.to_csv(os.path.join(DATA_DIR, 'deg_by_team.csv'), index=False)
    print(f'\nSaved 4 files to {DATA_DIR}')


if __name__ == '__main__':
    main()