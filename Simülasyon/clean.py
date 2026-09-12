"""
F1 Prediction Simulation - v0.3 - clean.py
Cleans raw laps and computes each driver's delta to pole and lap-to-lap sigma,
weighting recent races more heavily.

Why weighting
-------------
A flat season average treats round 1 and the most recent race as equally
informative. They are not: cars are developed, and a team that was midfield in
March may be quickest in August. The roadmap fixes the weight of the last race
at 30% of the total, and lambda is solved from that condition rather than
guessed:

    w_i = exp(-lambda * race_age)        race_age 0 = most recent

    sum(w) = 1 / 0.30  ->  solve for lambda

delta  = weighted mean over races of (driver's median clean lap - pole lap)
sigma  = weighted pooled std of clean laps around the driver's own race median

Known gap: a lap spent stuck behind another car still counts as clean, because
nothing here measures the gap to the car ahead. Traffic only ever slows a car,
so every driver's median is biased slow, and backmarkers are biased most. That
inflates the spread of delta. Fixing it needs a gap column - deferred.

Outputs:
    data/f1_2026_laps_clean.csv    lap level, clean_lap flag kept
    data/driver_pace_2026.csv      driver level, feeds simulate.py
"""

import os

import numpy as np
import pandas as pd

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASON = 2026

OUTLIER_MARGIN = 1.07          # median * 1.07 within a race -> outlier
MIN_CLEAN_LAPS_PER_RACE = 5    # driver needs this many clean laps to count
WET_COMPOUNDS = {'INTERMEDIATE', 'WET'}

LAST_RACE_WEIGHT = 0.30        # share of total weight given to the newest race
MIN_RACES_FOR_WEIGHTING = 3    # below this, fall back to equal weights

# Soft-knee compression of the delta spread.
# Measured deltas run to 4.2 s/lap, where real race-pace spread is nearer 2 s.
# The cause is traffic: a lap spent stuck behind another car still counts as
# clean, and backmarkers spend far more of the race there, so their median is
# biased slow the most. This is a patch on the symptom, not the cause - when a
# gap-to-car-ahead filter lands, set DELTA_RATIO back to 1.0 and re-measure.
DELTA_KNEE = 1.00              # below this, deltas pass through untouched
DELTA_RATIO = 2.50             # above it, every extra second counts this much less

# --- weighting --------------------------------------------------------------


def compress_delta(delta, knee=DELTA_KNEE, ratio=DELTA_RATIO):
    """
    Shrinks the tail of the delta spread while leaving the front of the field
    exactly as measured.

    Identity below the knee, so the cars fighting for the win keep their real
    pace differences. Above it, each further second is divided by ratio, which
    pulls a 4.2 s backmarker back to something a race distance can absorb.
    """
    d = np.asarray(delta, dtype=float)
    return np.where(d <= knee, d, knee + (d - knee) / ratio)


def solve_lambda(n_races, last_share=LAST_RACE_WEIGHT, tol=1e-9):
    """
    Finds lambda such that exp(0) / sum(exp(-lambda*i)) equals last_share.

    Bisection rather than a closed form: the sum is a geometric series in
    exp(-lambda), and inverting it for lambda is not worth the algebra.
    """
    if n_races < MIN_RACES_FOR_WEIGHTING:
        return 0.0

    target = 1.0 / last_share          # the sum of weights we need

    def total(lam):
        return float(np.exp(-lam * np.arange(n_races)).sum())

    # lambda = 0 gives sum = n_races (flat), large lambda gives sum -> 1
    if total(0.0) <= target:
        return 0.0                      # too few races to reach the target

    lo, hi = 0.0, 5.0
    while total(hi) > target:
        hi *= 2
        if hi > 1e4:
            return hi

    for _ in range(200):
        mid = (lo + hi) / 2
        if total(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


# --- cleaning ---------------------------------------------------------------


def add_clean_flag(laps):
    """Applies the project's clean_lap rules. Returns laps with new columns."""
    df = laps.copy()

    df['LapTime_s'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    df['TrackStatus'] = df['TrackStatus'].astype(str)
    df['Compound'] = df['Compound'].astype(str).str.upper()
    df['is_wet'] = df['Compound'].isin(WET_COMPOUNDS)

    valid = df['LapTime_s'].notna()
    green = df['TrackStatus'] == '1'
    no_pit = df['PitInTime'].isna() & df['PitOutTime'].isna()
    not_first = df['LapNumber'] > 1
    not_outlap = df['TyreLife'] > 1

    df['clean_lap'] = valid & green & no_pit & not_first & not_outlap & ~df['is_wet']

    for race, grp in df[df['clean_lap']].groupby('Race'):
        cutoff = grp['LapTime_s'].median() * OUTLIER_MARGIN
        df.loc[grp.index[grp['LapTime_s'] > cutoff], 'clean_lap'] = False

    return df


def build_driver_pace(clean, poles):
    """Recency-weighted delta to pole and lap-to-lap sigma across the season."""
    pole_map = poles.set_index('Race')['pole_time'].to_dict()
    clean = clean.copy()
    clean['pole_time'] = clean['Race'].map(pole_map)
    clean = clean[clean['pole_time'].notna()]
    clean['delta_lap'] = clean['LapTime_s'] - clean['pole_time']

    # race order: Round if we have it, otherwise first appearance
    if 'Round' in clean.columns:
        order = (clean.groupby('Race')['Round'].min()
                      .sort_values().index.tolist())
    else:
        order = list(dict.fromkeys(clean['Race']))

    n_races = len(order)
    lam = solve_lambda(n_races)
    age = {race: n_races - 1 - i for i, race in enumerate(order)}
    race_weight = {race: float(np.exp(-lam * age[race])) for race in order}

    per_race = (clean.groupby(['Driver', 'Race'])
                     .agg(race_median=('delta_lap', 'median'),
                          race_std=('delta_lap', 'std'),
                          n_laps=('delta_lap', 'size'))
                     .reset_index())
    per_race = per_race[per_race['n_laps'] >= MIN_CLEAN_LAPS_PER_RACE]
    per_race['age'] = per_race['Race'].map(age)
    per_race['weight'] = per_race['Race'].map(race_weight)

    rows = []
    for driver, grp in per_race.groupby('Driver'):
        team = clean[clean['Driver'] == driver]['Team'].mode().iloc[0]
        w = grp['weight'].to_numpy()

        delta = float(np.average(grp['race_median'], weights=w))

        # pooled within-race variance: race effects out, lap-to-lap noise left
        lap_w = w * (grp['n_laps'].to_numpy() - 1)
        var = grp['race_std'].to_numpy() ** 2
        ok = ~np.isnan(var) & (lap_w > 0)
        sigma = float(np.sqrt(np.average(var[ok], weights=lap_w[ok]))) \
            if ok.any() else np.nan

        rows.append({
            'Driver': driver,
            'Team': team,
            'delta': delta,
            'sigma': sigma,
            'n_races': len(grp),
            'n_laps': int(grp['n_laps'].sum()),
            'newest_race_weight': float(w.max() / w.sum()),
        })

    pace = pd.DataFrame(rows)
    pace['delta'] = pace['delta'] - pace['delta'].min()     # rebase on fastest
    pace['delta_raw'] = pace['delta']
    pace['delta'] = compress_delta(pace['delta'])
    pace = pace.sort_values('delta').reset_index(drop=True)
    pace[['delta', 'delta_raw', 'sigma']] = \
        pace[['delta', 'delta_raw', 'sigma']].round(3)

    return pace, lam, order, race_weight


def main():
    laps = pd.read_csv(os.path.join(DATA_DIR, f'f1_{SEASON}_laps.csv'))
    poles = pd.read_csv(os.path.join(DATA_DIR, f'f1_{SEASON}_poles.csv'))

    df = add_clean_flag(laps)
    clean = df[df['clean_lap']]

    print(f'Total laps      : {len(df)}')
    print(f'Clean laps kept : {len(clean)}  ({len(clean) / len(df):.1%})')
    print(f'Wet laps flagged: {int(df["is_wet"].sum())}')
    print(f'Races           : {clean["Race"].nunique()}\n')

    pace, lam, order, race_weight = build_driver_pace(clean, poles)

    total_w = sum(race_weight.values())
    print(f'lambda = {lam:.4f}  (newest race gets '
          f'{max(race_weight.values()) / total_w:.1%} of the weight)')
    print('\n--- race weights, oldest first ---')
    for race in order:
        share = race_weight[race] / total_w
        bar = '#' * max(1, int(share * 100))
        print(f'  {race:32s} {share:6.1%}  {bar}')

    if DELTA_RATIO != 1.0:
        print(f'\n--- delta compression: knee {DELTA_KNEE:.2f} s, '
              f'ratio {DELTA_RATIO:.2f} ---')
        print(f"raw spread      : {pace['delta_raw'].max():.3f} s/lap")
        print(f"after squeeze   : {pace['delta'].max():.3f} s/lap")
        touched = (pace['delta_raw'] > DELTA_KNEE).sum()
        print(f"drivers touched : {touched} of {len(pace)} "
              f"(the front {len(pace) - touched} are unchanged)")

    print(f'\n{pace.to_string(index=False)}')

    clean_path = os.path.join(DATA_DIR, f'f1_{SEASON}_laps_clean.csv')
    pace_path = os.path.join(DATA_DIR, f'driver_pace_{SEASON}.csv')
    df.to_csv(clean_path, index=False)
    pace.to_csv(pace_path, index=False)

    print(f'\nSaved: {clean_path}')
    print(f'Saved: {pace_path}')


if __name__ == '__main__':
    main()