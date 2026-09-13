"""
F1 Prediction Simulation - track_history.py
Measures per-track tyre degradation from 2022-2025, one pass per race.

Why not 2026: the current season has only 12 dry races, so a single bad weekend
sets a whole track's number. Four seasons give every circuit 3-4 samples.

Why season-normalised: compounds and aero changed across those years, so the
absolute level of degradation is not comparable between seasons. Each race is
divided by its own season's median, leaving the track's relative character -
which is the part that carries over to 2026.

    race_mult  = race median deg_pct / season median deg_pct
    track_mult = weighted mean of race_mult across seasons, recent years heavier

Outputs:
    data/track_deg_history.csv    race level, every season
    data/track_deg_measured.csv   track level, ready to paste into track_deg.py
"""

import os
import warnings
from collections import defaultdict

import fastf1
import numpy as np
import pandas as pd

from Simülasyon.fuel_effect import get_track_params

try:
    from scipy.stats import theilslopes
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

warnings.filterwarnings('ignore')

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]
RECENCY_HALFLIFE = 2.0        # years; weight halves every this many years

# cleaning, same rules as clean.py
OUTLIER_MARGIN = 1.07
DRY_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD'}

# stint fitting, same filters as deg.py
MIN_LAPS_PER_STINT = 12
MIN_TYRELIFE_SPAN = 8
MAX_GAP_RATIO = 0.5
MIN_STINTS_PER_RACE = 10      # below this the race median is not trustworthy
MAX_WET_SHARE = 0.15          # skip the race if more laps than this were wet

# --- helpers ----------------------------------------------------------------


def fit_slope(x, y):
    if HAVE_SCIPY:
        slope, _, _, _ = theilslopes(y, x)
        return float(slope)
    return float(np.polyfit(x, y, 1)[0])


def clean_laps(laps):
    """Applies the project's clean_lap rules to one race."""
    df = laps.copy()
    df['LapTime_s'] = df['LapTime'].dt.total_seconds()
    df['TrackStatus'] = df['TrackStatus'].astype(str)
    df['Compound'] = df['Compound'].astype(str).str.upper()

    wet_share = (~df['Compound'].isin(DRY_COMPOUNDS) & df['Compound'].ne('NAN')).mean()

    mask = (df['LapTime_s'].notna()
            & (df['TrackStatus'] == '1')
            & df['PitInTime'].isna()
            & df['PitOutTime'].isna()
            & (df['LapNumber'] > 1)
            & (df['TyreLife'] > 1)
            & df['Compound'].isin(DRY_COMPOUNDS))

    clean = df[mask].copy()
    if clean.empty:
        return clean, wet_share

    cutoff = clean['LapTime_s'].median() * OUTLIER_MARGIN
    clean = clean[clean['LapTime_s'] <= cutoff]
    return clean, wet_share


def race_deg(clean, fuel_effect):
    """Median stint degradation for one race, as % of a lap per lap of tyre age."""
    base_lap = clean['LapTime_s'].median()
    slopes = []

    for _, grp in clean.groupby(['Driver', 'Stint']):
        if len(grp) < MIN_LAPS_PER_STINT:
            continue
        x = grp['TyreLife'].to_numpy(dtype=float)
        span = x.max() - x.min()
        if span < MIN_TYRELIFE_SPAN:
            continue
        if len(grp) / (span + 1) < MAX_GAP_RATIO:
            continue

        y = grp['LapTime_s'].to_numpy(dtype=float)
        lap_no = grp['LapNumber'].to_numpy(dtype=float)
        y_corrected = y + fuel_effect * (lap_no - 1)
        slopes.append(fit_slope(x, y_corrected))

    if len(slopes) < MIN_STINTS_PER_RACE:
        return None, len(slopes), base_lap

    return float(np.median(slopes)) / base_lap * 100, len(slopes), base_lap


# --- main -------------------------------------------------------------------


def collect():
    rows = []
    for season in SEASONS:
        print(f'\n===== {season} =====')
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
        except Exception as exc:
            print(f'  schedule failed: {exc}')
            continue

        for _, event in schedule.iterrows():
            name = event['EventName']
            try:
                s = fastf1.get_session(season, name, 'R')
                s.load(telemetry=False, weather=False, messages=False)
                laps = s.laps
                if laps is None or laps.empty:
                    print(f'  {name:32s} no laps')
                    continue
            except Exception as exc:
                print(f'  {name:32s} skipped ({type(exc).__name__})')
                continue

            clean, wet_share = clean_laps(laps)
            if wet_share > MAX_WET_SHARE:
                print(f'  {name:32s} wet ({wet_share:.0%}), skipped')
                continue
            if clean.empty:
                print(f'  {name:32s} no clean laps')
                continue

            fuel_effect = get_track_params(name)['fuel_effect']
            deg_pct, n_stints, base_lap = race_deg(clean, fuel_effect)
            if deg_pct is None:
                print(f'  {name:32s} only {n_stints} stints, skipped')
                continue

            rows.append({
                'Season': season,
                'Race': name,
                'deg_pct': deg_pct,
                'n_stints': n_stints,
                'base_lap': base_lap,
            })
            print(f'  {name:32s} deg {deg_pct:7.4f} %/lap   ({n_stints} stints)')

    return pd.DataFrame(rows)


def aggregate(hist):
    """Season-normalise, then weight recent seasons more heavily."""
    season_median = hist.groupby('Season')['deg_pct'].transform('median')
    hist = hist.copy()
    hist['season_median'] = season_median
    hist['race_mult'] = hist['deg_pct'] / season_median

    newest = hist['Season'].max()
    hist['weight'] = 0.5 ** ((newest - hist['Season']) / RECENCY_HALFLIFE)

    rows = []
    for race, grp in hist.groupby('Race'):
        w = grp['weight'].to_numpy()
        rows.append({
            'Race': race,
            'track_mult': float(np.average(grp['race_mult'], weights=w)),
            'n_seasons': len(grp),
            'spread': float(grp['race_mult'].max() - grp['race_mult'].min()),
            'seasons': ','.join(str(s) for s in sorted(grp['Season'])),
        })

    out = pd.DataFrame(rows).sort_values('track_mult', ascending=False)
    return hist, out.reset_index(drop=True)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    print(f'Seasons: {SEASONS}')
    print(f'Regression: {"Theil-Sen" if HAVE_SCIPY else "least squares"}')
    print('First run downloads ~90 race sessions. This takes a while.')

    hist = collect()
    if hist.empty:
        print('\nNothing collected.')
        return

    hist, tracks = aggregate(hist)

    print('\n===== per-season medians =====')
    print(hist.groupby('Season')['deg_pct'].agg(['median', 'size']).round(4).to_string())

    print('\n===== track multipliers (season-normalised, recency weighted) =====')
    print(tracks.round(3).to_string(index=False))

    print('\n===== paste into track_deg.py =====')
    for _, r in tracks.iterrows():
        print(f"    # {r['Race']:34s} 'measured_mult': {r['track_mult']:.2f},"
              f"  # {r['n_seasons']} seasons, spread {r['spread']:.2f}")

    hist.to_csv(os.path.join(DATA_DIR, 'track_deg_history.csv'), index=False)
    tracks.to_csv(os.path.join(DATA_DIR, 'track_deg_measured.csv'), index=False)
    print(f'\nSaved 2 files to {DATA_DIR}')


if __name__ == '__main__':
    main()