"""
F1 Prediction Simulation - track_evolution.py
Separates track evolution from tyre degradation, per race.

The problem
-----------
Inside a stint, TyreLife and LapNumber rise together, so a single stint fit
cannot tell "the tyre is ageing" from "the track is rubbering in". That is why
the stint-only method returned negative degradation at Silverstone, Australia
and Jeddah: evolution was eating the slope.

The fix
-------
Fit the whole field at once. Drivers pit on different laps, so on lap 30 one car
is on a 2-lap-old tyre and another on a 25-lap-old tyre. That decoupling lets a
single regression separate the two effects:

    lap_time = driver_effect + b * LapNumber + c * TyreLife

    c = degradation, s per lap of tyre age
    b = fuel burn + track evolution, both functions of race lap
    evolution = -b - fuel_effect        (both make lap times fall)

Driver effects are removed by within-driver demeaning, so pace differences do
not leak into either slope.

Outputs:
    data/track_evolution.csv       race level
    data/track_evolution_agg.csv   track level, season-normalised
"""

import os
import warnings

import numpy as np
import pandas as pd

from Simülasyon.dataset import load_laps, clean_race, iter_races
from Simülasyon.fuel_effect import get_track_params

warnings.filterwarnings('ignore')

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]
RECENCY_HALFLIFE = 2.0

QUADRATIC_EVOLUTION = True    # rubbering-in is front-loaded, not linear

MIN_LAPS_PER_RACE = 200       # pooled clean laps needed to fit a race
MIN_DRIVERS_PER_RACE = 8
MAX_WET_SHARE = 0.15

# --- joint fit --------------------------------------------------------------


def fit_race(clean, fuel_effect):
    """
    Pooled within-driver regression of lap time on race lap and tyre age.
    Returns dict or None.
    """
    if len(clean) < MIN_LAPS_PER_RACE:
        return None
    if clean['Driver'].nunique() < MIN_DRIVERS_PER_RACE:
        return None

    y = clean['LapTime_s'].to_numpy(dtype=float)
    lap = clean['LapNumber'].to_numpy(dtype=float)
    tyre = clean['TyreLife'].to_numpy(dtype=float)

    cols = [lap, tyre]
    names = ['b_lap', 'deg']
    if QUADRATIC_EVOLUTION:
        cols.append(lap ** 2)
        names.append('b_lap2')

    X = np.column_stack(cols)

    # within transformation: subtract each driver's own mean
    drivers = clean['Driver'].to_numpy()
    df_tmp = pd.DataFrame(X, columns=names)
    df_tmp['y'] = y
    df_tmp['Driver'] = drivers
    demeaned = df_tmp.groupby('Driver')[names + ['y']].transform(lambda s: s - s.mean())

    Xd = demeaned[names].to_numpy()
    yd = demeaned['y'].to_numpy()

    coef, residuals, rank, _ = np.linalg.lstsq(Xd, yd, rcond=None)
    if rank < Xd.shape[1]:
        return None

    fitted = Xd @ coef
    ss_res = float(np.sum((yd - fitted) ** 2))
    ss_tot = float(np.sum(yd ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    b_lap = float(coef[0])
    deg = float(coef[1])
    b_lap2 = float(coef[2]) if QUADRATIC_EVOLUTION else 0.0

    # evolution at the start of the race, s/lap; positive = track speeding up
    evo_rate = -b_lap - fuel_effect
    base_lap = float(np.median(y))
    n_laps = int(lap.max())

    # total evolution gain over the race, integrating the quadratic
    total_evo = evo_rate * n_laps - b_lap2 * (n_laps ** 2) / 2.0

    return {
        'deg_abs': deg,
        'deg_pct': deg / base_lap * 100,
        'evo_rate': evo_rate,
        'evo_curvature': b_lap2,
        'evo_total': total_evo,
        'b_lap': b_lap,
        'fuel_effect': fuel_effect,
        'base_lap': base_lap,
        'n_clean_laps': len(clean),
        'n_drivers': int(clean['Driver'].nunique()),
        'r2': r2,
    }


# --- collection -------------------------------------------------------------


def collect():
    rows = []
    laps = load_laps(SEASONS)

    for season, name, race_laps, race_df in iter_races(laps):
        clean, wet_share = clean_race(race_df)
        if wet_share > MAX_WET_SHARE:
            print(f'  {season} {name:32s} wet ({wet_share:.0%}), skipped')
            continue

        fuel_effect = get_track_params(name)['fuel_effect']
        fit = fit_race(clean, fuel_effect)
        if fit is None:
            print(f'  {season} {name:32s} not enough clean data')
            continue

        fit.update({'Season': season, 'Race': name})
        rows.append(fit)
        print(f"  {season} {name:32s} deg {fit['deg_abs']:+.4f}  "
              f"evo {fit['evo_rate']:+.4f} s/lap  R2 {fit['r2']:.2f}")

    return pd.DataFrame(rows)


def aggregate(hist):
    hist = hist.copy()
    hist['season_median_deg'] = hist.groupby('Season')['deg_pct'].transform('median')
    hist['deg_mult'] = hist['deg_pct'] / hist['season_median_deg']

    newest = hist['Season'].max()
    hist['weight'] = 0.5 ** ((newest - hist['Season']) / RECENCY_HALFLIFE)

    rows = []
    for race, grp in hist.groupby('Race'):
        w = grp['weight'].to_numpy()
        rows.append({
            'Race': race,
            'deg_mult': float(np.average(grp['deg_mult'], weights=w)),
            'deg_pct': float(np.average(grp['deg_pct'], weights=w)),
            'evo_rate': float(np.average(grp['evo_rate'], weights=w)),
            'n_seasons': len(grp),
            'deg_spread': float(grp['deg_mult'].max() - grp['deg_mult'].min()),
            'seasons': ','.join(str(x) for x in sorted(grp['Season'])),
        })

    return hist, pd.DataFrame(rows).sort_values('deg_mult', ascending=False).reset_index(drop=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f'Seasons: {SEASONS}')
    print(f'Evolution model: {"quadratic" if QUADRATIC_EVOLUTION else "linear"}')

    hist = collect()
    if hist.empty:
        print('\nNothing collected.')
        return

    hist, tracks = aggregate(hist)

    print('\n===== per-season medians =====')
    print(hist.groupby('Season')[['deg_pct', 'evo_rate', 'r2']]
              .median().round(4).to_string())

    print('\n===== per track (season-normalised, recency weighted) =====')
    print(tracks.round(4).to_string(index=False))

    print('\n===== paste into track_deg.py =====')
    for _, r in tracks.iterrows():
        print(f"    # {r['Race']:32s} 'measured_mult': {r['deg_mult']:.2f},"
              f"  # {r['n_seasons']}s, spread {r['deg_spread']:.2f}, "
              f"evo {r['evo_rate']:+.3f}")

    hist.to_csv(os.path.join(DATA_DIR, 'track_evolution.csv'), index=False)
    tracks.to_csv(os.path.join(DATA_DIR, 'track_evolution_agg.csv'), index=False)
    print(f'\nSaved 2 files to {DATA_DIR}')


if __name__ == '__main__':
    main()