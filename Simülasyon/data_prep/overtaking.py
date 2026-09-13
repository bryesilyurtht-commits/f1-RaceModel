"""
F1 Prediction Simulation - overtaking.py
Measures how hard it is to pass, conditioned on actually being faster.

Why the first version was wrong
-------------------------------
Counting passes per close lap mixed two different things. At Monza the field
runs nose to tail in the slipstream, so hundreds of laps count as "close" even
though nobody following has a pace advantage. Equal cars cannot pass each other,
and that is not the circuit being difficult. Monza came out harder than
Zandvoort, which is backwards.

What this version does
----------------------
A close lap only counts as a chance to pass when the following car is genuinely
quicker: its clean-lap median for that race beats the car ahead by at least
MIN_ADVANTAGE. The rate then answers the question the simulation asks, which is
"a faster car is stuck behind a slower one - does it get through this lap?"

It also fits a logistic curve to pass probability against pace advantage:

    p_pass = 1 / (1 + exp(-(advantage - threshold) / scale))

threshold is where a pass becomes even money, scale is how sharply that
transition happens. Those two numbers go straight into the lap loop.

Outputs:
    data/overtaking_by_race.csv
    data/overtaking_events.csv   one row per close lap, for refitting later
    data/overtaking.csv          per track, ready for tracks.py
"""

import os

import numpy as np
import pandas as pd

from Simülasyon.data_prep.dataset import load_laps, clean_race, iter_races
from Simülasyon.tracks import ATTACK_GAP

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]

CLOSE_GAP = 1.5           # seconds; inside this a pass is physically possible
MIN_ADVANTAGE = 0.15      # s/lap the follower must be quicker to count as a chance
PIT_BLACKOUT = 2          # laps around a stop where swaps are pit cycle, not racing
MIN_DRIVERS = 12
MIN_CLEAN_LAPS = 8        # laps needed before a driver's race pace is trusted
MIN_EVENTS_FOR_FIT = 40   # per-track logistic fits below this are not reported

# --- race reconstruction ----------------------------------------------------


def race_time_matrix(race_df):
    """Cumulative race time, pit-lap flags and green flags, indexed by lap."""
    piv = race_df.pivot_table(index='LapNumber', columns='Driver',
                              values='LapTime_s', aggfunc='first')
    if piv.empty:
        return None, None, None

    valid = piv.notna()
    cum = piv.fillna(0).cumsum().where(valid)

    pit = race_df.pivot_table(index='LapNumber', columns='Driver',
                              values='PitInTime_s', aggfunc='first').notna()
    out = race_df.pivot_table(index='LapNumber', columns='Driver',
                              values='PitOutTime_s', aggfunc='first').notna()
    pit_lap = (pit | out).reindex_like(valid).fillna(False)

    status = race_df.pivot_table(index='LapNumber', columns='Driver',
                                 values='TrackStatus', aggfunc='first')
    green = status.apply(lambda col: col.astype(str) == '1')

    return cum, pit_lap, green


def blackout_mask(pit_lap):
    mask = pit_lap.copy()
    for shift in range(1, PIT_BLACKOUT + 1):
        mask = mask | pit_lap.shift(shift).fillna(False)
        mask = mask | pit_lap.shift(-shift).fillna(False)
    return mask


def driver_pace(race_df):
    """Clean-lap median per driver: the race pace used to judge who is faster."""
    clean, _ = clean_race(race_df)
    if clean.empty:
        return {}
    counts = clean.groupby('Driver')['LapTime_s'].size()
    medians = clean.groupby('Driver')['LapTime_s'].median()
    return medians[counts >= MIN_CLEAN_LAPS].to_dict()


# --- event extraction -------------------------------------------------------


def collect_events(race_df):
    """
    One row per close lap: who was following whom, by how much they were faster,
    and whether the pass happened.
    """
    cum, pit_lap, green = race_time_matrix(race_df)
    if cum is None or cum.shape[1] < MIN_DRIVERS:
        return []

    pace = driver_pace(race_df)
    if len(pace) < MIN_DRIVERS:
        return []

    blackout = blackout_mask(pit_lap)
    laps = cum.index.to_numpy()
    events = []

    for i in range(1, len(laps)):
        if laps[i] <= 2:                      # lap 1 is the start, not racing
            continue

        prev, cur = cum.iloc[i - 1], cum.iloc[i]
        alive = prev.notna() & cur.notna()
        ok = alive & ~blackout.iloc[i] & ~blackout.iloc[i - 1] & green.iloc[i]
        drivers = [d for d in cum.columns
                   if bool(ok.get(d, False)) and d in pace]
        if len(drivers) < 2:
            continue

        prev_order = prev[drivers].sort_values().index.tolist()

        for p in range(1, len(prev_order)):
            behind, ahead = prev_order[p], prev_order[p - 1]
            gap = prev[behind] - prev[ahead]
            if gap > CLOSE_GAP:
                continue

            # positive when the follower is the quicker car
            advantage = pace[ahead] - pace[behind]
            events.append({
                'lap': int(laps[i]),
                # who, added in v1.5 for the per-team breakdown in
                # overtake_speed.py. Purely additive: no filter, threshold or
                # decision above reads these, so the set of events collected
                # and every existing column are unchanged.
                'attacker': str(behind),
                'defender': str(ahead),
                'gap': float(gap),
                'advantage': float(advantage),
                'passed': bool(cur[behind] < cur[ahead]),
            })

    return events


# --- logistic fit -----------------------------------------------------------


def fit_logistic(advantage, passed):
    """
    Fits p = 1 / (1 + exp(-(x - threshold) / scale)) by maximum likelihood.
    Returns (threshold, scale) or (nan, nan).
    """
    x = np.asarray(advantage, dtype=float)
    y = np.asarray(passed, dtype=float)
    if len(x) < MIN_EVENTS_FOR_FIT or y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan, np.nan

    def nll(params):
        threshold, log_scale = params
        scale = np.exp(log_scale)
        z = (x - threshold) / scale
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    start = [float(np.median(x)), np.log(0.3)]

    if HAVE_SCIPY:
        res = minimize(nll, start, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        if not res.success:
            return np.nan, np.nan
        threshold, log_scale = res.x
    else:                                   # coarse grid, good enough
        best, threshold, log_scale = np.inf, np.nan, np.nan
        for t in np.linspace(-1.0, 2.0, 61):
            for ls in np.log(np.linspace(0.05, 1.5, 30)):
                v = nll([t, ls])
                if v < best:
                    best, threshold, log_scale = v, t, ls

    return float(threshold), float(np.exp(log_scale))


def fit_gap_term(chances):
    """
    Pass odds against pace advantage AND how close the follower actually is.

    The model above asks only how much quicker the following car is, and
    treats a car sitting 0.2 s back the same as one hanging on at 0.95 s.
    Those are not the same situation. Inside the simulation's own window the
    pass rate runs from 0.69 under a quarter of a second to 0.10 approaching
    a full second, and the gap is almost uncorrelated with the advantage
    (-0.12), so it is separate information rather than a restatement.

    Adding it takes the pseudo-R2 from 0.050 to 0.164 on 2022-25 - the term
    explains more than twice what the advantage does on its own.

    Fitted by IRLS in numpy, because scipy's optimisers are blocked in this
    environment and a logistic likelihood has a closed-form Newton step.
    """
    x = np.column_stack([np.ones(len(chances)),
                         chances['advantage'].to_numpy(float),
                         chances['gap'].to_numpy(float)])
    y = chances['passed'].to_numpy(float)
    if len(y) < 200:
        return None

    beta = np.zeros(3)
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-6, None)
        try:
            step = np.linalg.solve((x * w[:, None]).T @ x, x.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break

    def loglik(b):
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(x @ b, -30, 30))), 1e-9, 1 - 1e-9)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    base = float(y.mean())
    ll0 = float(np.sum(y * np.log(base) + (1 - y) * np.log(1 - base)))
    return {'intercept': float(beta[0]), 'advantage': float(beta[1]),
            'gap': float(beta[2]), 'n': len(y),
            'pseudo_r2': 1 - loglik(beta) / ll0,
            'mean_gap': float(chances['gap'].mean())}


# --- main -------------------------------------------------------------------


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    laps = load_laps(SEASONS)

    race_rows, all_events = [], []

    for season, name, race_laps, race_df in iter_races(laps):
        events = collect_events(race_df)
        if not events:
            print(f'  {season} {name:32s} no usable events')
            continue

        ev = pd.DataFrame(events)
        chance = ev[ev['advantage'] >= MIN_ADVANTAGE]

        raw_rate = float(ev['passed'].mean())
        cond_rate = float(chance['passed'].mean()) if len(chance) else np.nan

        race_rows.append({
            'Season': season,
            'Race': name,
            'close_laps': len(ev),
            'chances': len(chance),
            'passes': int(ev['passed'].sum()),
            'passes_when_faster': int(chance['passed'].sum()) if len(chance) else 0,
            'raw_rate': raw_rate,
            'pass_rate': cond_rate,
        })

        ev['Season'], ev['Race'] = season, name
        all_events.append(ev)

        print(f"  {season} {name:32s} close {len(ev):4d}  chances {len(chance):4d}  "
              f"raw {raw_rate:.3f}  conditioned {cond_rate:.3f}")

    if not race_rows:
        print('Nothing collected.')
        return

    hist = pd.DataFrame(race_rows)
    events = pd.concat(all_events, ignore_index=True)

    field_rate = hist['pass_rate'].median()
    chances = events[events['advantage'] >= MIN_ADVANTAGE]

    pooled_t, pooled_s = fit_logistic(chances['advantage'], chances['passed'])
    gap_fit = fit_gap_term(chances[chances['gap'] < ATTACK_GAP])

    rows = []
    for race, grp in hist.groupby('Race'):
        ev = events[events['Race'] == race]
        ch = ev[ev['advantage'] >= MIN_ADVANTAGE]
        t, s = fit_logistic(ch['advantage'], ch['passed'])

        # The same rate measured on the window the simulation actually rolls
        # in. CLOSE_GAP is 1.5 s here because a pass can start from there,
        # but simulate.py only rolls dice inside ATTACK_GAP, which is 1.0.
        # Twenty-eight percent of the laps behind pass_rate therefore sit in
        # a band the simulation never visits, and passes are much rarer out
        # there - 0.057 between 1.00 and 1.25 s against 0.691 inside 0.25.
        # Feeding the wide-window rate into a narrow-window roll understates
        # passing by about a quarter, so the narrow one is published
        # alongside it. Nothing about the detection above changed.
        near = ch[ch['gap'] < ATTACK_GAP]
        rows.append({
            'Race': race,
            'pass_rate': float(grp['pass_rate'].median()),
            'raw_rate': float(grp['raw_rate'].median()),
            'pass_rate_near': float(near['passed'].mean()) if len(near) else np.nan,
            'mean_gap_near': float(near['gap'].mean()) if len(near) else np.nan,
            'chances_near': int(len(near)),
            'chances': float(grp['chances'].median()),
            'close_laps': float(grp['close_laps'].median()),
            'threshold': t,
            'scale': s,
            'n_seasons': len(grp),
        })

    tracks = pd.DataFrame(rows)
    tracks['rate_index'] = tracks['pass_rate'] / field_rate
    ranked = tracks['pass_rate'].rank(ascending=False, method='first')
    tracks['overtake_difficulty'] = np.ceil(ranked / len(tracks) * 5).clip(1, 5).astype(int)
    tracks = tracks.sort_values('pass_rate', ascending=False).reset_index(drop=True)

    print('\n===== field-wide =====')
    print(f'Close laps per race      : {hist["close_laps"].median():.0f}')
    print(f'Of those, real chances   : {hist["chances"].median():.0f} '
          f'({hist["chances"].median() / max(hist["close_laps"].median(), 1):.0%})')
    print(f'Raw pass rate            : {hist["raw_rate"].median():.3f}')
    print(f'Conditioned pass rate    : {field_rate:.3f}')
    print(f'Logistic fit, all tracks : threshold {pooled_t:.3f} s/lap, '
          f'scale {pooled_s:.3f}  ({"scipy" if HAVE_SCIPY else "grid search"})')

    if gap_fit is not None:
        near = chances[chances['gap'] < ATTACK_GAP]
        print(f'\n===== the gap the simulation rolls in (< {ATTACK_GAP:.2f} s) =====')
        print(f'Laps inside it           : {len(near):,} of {len(chances):,} '
              f'({len(near) / len(chances):.0%})')
        print(f'Pass rate inside it      : {near["passed"].mean():.3f}  '
              f'against {chances["passed"].mean():.3f} over the full window')
        print(f'Mean gap inside it       : {gap_fit["mean_gap"]:.3f} s')
        print(f'With the gap as a term   : advantage {gap_fit["advantage"]:+.3f}, '
              f'gap {gap_fit["gap"]:+.3f} per second')
        print(f'                           pseudo-R2 {gap_fit["pseudo_r2"]:.3f} '
              f'on {gap_fit["n"]:,} laps')
        print('Paste into simulate.py   : '
              f'GAP_COEF = {abs(gap_fit["gap"]):.2f}, '
              f'REFERENCE_GAP = {gap_fit["mean_gap"]:.2f}')

    print('\n===== per track (high rate = easy to pass) =====')
    cols = ['Race', 'pass_rate', 'raw_rate', 'rate_index', 'overtake_difficulty',
            'threshold', 'scale', 'chances', 'close_laps', 'n_seasons']
    print(tracks[cols].round(3).to_string(index=False))

    print('\n===== paste into tracks.py =====')
    for _, r in tracks.iterrows():
        print(f"    # {r['Race']:32s} 'overtake_difficulty': "
              f"{int(r['overtake_difficulty'])},  # p_pass {r['pass_rate']:.3f}")

    hist.to_csv(os.path.join(DATA_DIR, 'overtaking_by_race.csv'), index=False)
    events.to_csv(os.path.join(DATA_DIR, 'overtaking_events.csv'), index=False)
    tracks.to_csv(os.path.join(DATA_DIR, 'overtaking.csv'), index=False)
    print(f'\nSaved 3 files to {DATA_DIR}')


if __name__ == '__main__':
    main()