"""
F1 Prediction Simulation - v0.7 - track_affinity.py
Measures which circuits suit which drivers and teams.

Definition
----------
    affinity = driver's season average position that year
             - driver's position at this circuit that year

Subtracting the season average is what makes this affinity rather than a
ranking of who is good. A driver who finishes 4th everywhere and 4th here has an
affinity of zero; one who averages 12th but takes 6th at Monaco scores +6, and
that is the signal worth carrying into the model.

Retirements are dropped. A blown engine says nothing about whether a circuit
suits a driver, and leaving DNFs in would punish exactly the teams whose cars
break most, which is a reliability effect the simulation does not model.

Seasons are weighted exponentially, newest heaviest, because a 2022 result
describes a car that no longer exists.

Signal choice
-------------
The roadmap prefers qualifying position over finishing position: a race result
carries strategy, safety cars and first-lap incidents, while a grid slot is
closer to raw pace at that circuit. Both are computed here; SIGNAL picks one.

Outputs:
    data/track_affinity_driver.csv
    data/track_affinity_team.csv
    data/track_affinity.csv          both, long format, for simulate.py
"""

import os

import numpy as np
import pandas as pd

from Simülasyon.data_prep.dataset import load_results

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
RECENCY_HALFLIFE = 2.0        # years; weight halves every this many years

SIGNAL = 'finish'             # 'finish' or 'grid'

MIN_RACES_IN_SEASON = 8       # season average needs this many classified races
MIN_SAMPLES = 2               # circuit needs this many usable years per driver

# --- similarity pooling (v1.4) ---
# Raw affinity is measured from two or three races per circuit, and Monaco has
# three. That is not enough to tell a real preference from one good weekend.
# track_features.py describes each circuit by what it physically demands, so
# circuits that ask the same questions can lend each other data: Monaco's
# affinity is supported by Singapore and Hungary, and not by Monza.
#
# No clustering. A cluster boundary is arbitrary and a circuit that falls on
# the wrong side of one gets a completely different answer for no physical
# reason. The weight instead falls off smoothly with distance in feature
# space, and the circuit's own data sits at distance zero, so it always
# carries the largest single weight. Pooling supports a circuit's own
# measurement, it does not replace it.
# Off, because the measurement said so.
#
# The bandwidth is chosen by leaving each circuit out and predicting it from
# the rest. If similar circuits carried each other's affinity there would be
# an interior optimum: some h where neighbours help and distant circuits do
# not. There is none. The error falls monotonically as h grows and bottoms out
# at h = infinity, which is plain unweighted averaging:
#
#     h = 0.25 (neighbours only)   mse 7.56
#     h = 1.00                     mse 5.33
#     h = 4.00                     mse 4.08
#     h = inf  (all circuits equal) mse 3.83
#
# Weighting by similarity is worse than ignoring it, and the tighter the
# weighting the worse it gets - the exact opposite of the premise.
#
# Worse still, predicting no affinity at all scores 3.46, better than any
# pooled version and better than a team's own average over its other
# circuits. The raw affinity numbers have a variance of 3.46, so what the
# cross-validation is really saying is that there is no structure here to
# transfer between circuits - not by similarity, and not by team either.
#
# The machinery below is finished and tested, and flipping this to True runs
# it. It is off because the circuits do not support it, not because it does
# not work. Re-check when there are more seasons per circuit: the raw affinity
# rests on two or three races, and noise on both sides of the comparison is
# enough on its own to produce this result.
AFFINITY_POOLING = False

# Widened past the directive's 4.0 after the optimum was found sitting on that
# edge. It is not an edge effect - the error keeps falling to infinity - and
# the grid now reaches far enough to show that rather than hide it.
H_GRID = np.concatenate([np.arange(0.25, 4.01, 0.25),
                         np.array([6.0, 10.0, 20.0, 50.0, 200.0])])

# Teams only. Sensitivity to circuit type is a property of the car - aero
# efficiency, power unit, suspension. If a driver is genuinely better in slow
# corners the effect is far smaller than the car's, and the sample per driver
# is smaller than the sample per team, so pooling driver affinity would be
# borrowing noise to explain noise.
POOL_TEAMS_ONLY = True

# statuses that count as a classified finish
FINISHED_PREFIXES = ('Finished', '+')

# --- helpers ----------------------------------------------------------------


def is_classified(status):
    s = str(status).strip()
    return s.startswith(FINISHED_PREFIXES)


def prepare(results):
    df = results.copy()
    df = df[df['Season'].isin(SEASONS)]

    df['classified'] = df['Status'].apply(is_classified) if 'Status' in df else True
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')

    if SIGNAL == 'grid':
        # a grid slot exists even when the car later retires
        df['signal'] = df['GridPosition']
        df = df[df['signal'].notna() & (df['signal'] > 0)]
    else:
        df = df[df['classified'] & df['Position'].notna()]
        df['signal'] = df['Position']

    return df


def season_baselines(df, key):
    """Each entity's own average signal per season - the bar affinity is measured against."""
    base = (df.groupby([key, 'Season'])
              .agg(season_avg=('signal', 'mean'), n_races=('signal', 'size'))
              .reset_index())
    return base[base['n_races'] >= MIN_RACES_IN_SEASON]


def affinity_for(df, key):
    base = season_baselines(df, key)
    merged = df.merge(base, on=[key, 'Season'], how='inner')

    # positive when the circuit flatters this entity
    merged['affinity'] = merged['season_avg'] - merged['signal']

    newest = max(SEASONS)
    merged['weight'] = 0.5 ** ((newest - merged['Season']) / RECENCY_HALFLIFE)

    rows = []
    for (entity, race), grp in merged.groupby([key, 'Race']):
        if len(grp) < MIN_SAMPLES:
            continue
        w = grp['weight'].to_numpy()
        rows.append({
            key: entity,
            'Race': race,
            'affinity': float(np.average(grp['affinity'], weights=w)),
            'n_samples': len(grp),
            'spread': float(grp['affinity'].max() - grp['affinity'].min()),
            'seasons': ','.join(str(s) for s in sorted(grp['Season'])),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['Race', 'affinity'], ascending=[True, False]).reset_index(drop=True)


# --- similarity pooling ------------------------------------------------------


def load_track_features():
    """
    The z-scored circuit features, keyed by race name.

    Returns None when track_features.py has not been run - pooling then falls
    back to the raw affinity rather than failing, because a missing feature
    table should cost accuracy, not stop the pipeline.
    """
    path = os.path.join(DATA_DIR, 'track_features.csv')
    if not os.path.exists(path):
        return None, []

    feats = pd.read_csv(path)
    cols = [c for c in feats.columns if c.endswith('_z')]
    if not cols:
        # older layout: the z-scored values share the raw column names
        scale_path = os.path.join(DATA_DIR, 'track_features_scaling.csv')
        if not os.path.exists(scale_path):
            return None, []
        scaling = pd.read_csv(scale_path)
        cols = []
        for _, r in scaling.iterrows():
            name = f'{r["feature"]}_z'
            feats[name] = (feats[r['feature']] - r['mean']) / r['std']
            cols.append(name)

    return feats.set_index('Race')[cols], cols


def kernel_weights(features, cols, h):
    """
    w(i, j) = exp(-d(i, j)^2 / h) over every pair of circuits.

    The diagonal is 1 by construction - a circuit is its own closest
    neighbour - which is what keeps its own races the heaviest single
    contribution to its pooled value.
    """
    x = features[cols].to_numpy(dtype=float)
    diff = x[:, None, :] - x[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    return np.exp(-d2 / h)


def pool_affinity(raw, features, cols, h, key='Team'):
    """
    Weighted average of an entity's affinity over every circuit it has raced,
    with the weights coming from how similar those circuits are to the target.

    A circuit the entity has no data for still gets a value, entirely from its
    neighbours. That is the case the existing model cannot handle at all and
    the main thing this version buys.
    """
    circuits = list(features.index)
    index = {name: i for i, name in enumerate(circuits)}
    w = kernel_weights(features, cols, h)

    rows = []
    for entity, grp in raw.groupby(key):
        have = grp[grp['Race'].isin(index)]
        if have.empty:
            continue
        js = np.array([index[r] for r in have['Race']])
        vals = have['affinity'].to_numpy(dtype=float)

        for target in circuits:
            i = index[target]
            weights = w[i, js]
            total = weights.sum()
            if total <= 1e-12:
                continue
            own = float(grp.loc[grp['Race'] == target, 'affinity'].mean()) \
                if target in set(grp['Race']) else np.nan
            rows.append({
                key: entity,
                'Race': target,
                'affinity': float((weights * vals).sum() / total),
                'affinity_raw': own,
                'own_weight': float(weights[js == i].sum() / total) if i in js else 0.0,
                'n_circuits': int(len(js)),
            })
    return pd.DataFrame(rows)


def choose_bandwidth(raw, features, cols, key='Team'):
    """
    Picks h by leaving each circuit out and predicting it from the rest.

    A bandwidth chosen by eye is a constant nobody can defend later. This one
    is measured: for every entity and every circuit it actually raced, the
    affinity is predicted from that entity's other circuits, and the h with
    the smallest total error wins.

    A winner sitting on the edge of the grid is reported rather than accepted.
    That is what an unidentified parameter looks like - the same failure the
    tyre cliff fit hit with W and m, where the optimum slid to the boundary
    because nothing in the data stopped it.
    """
    circuits = list(features.index)
    index = {name: i for i, name in enumerate(circuits)}
    x = features[cols].to_numpy(dtype=float)
    diff = x[:, None, :] - x[None, :, :]
    d2 = (diff * diff).sum(axis=-1)

    scored = []
    for h in H_GRID:
        w = np.exp(-d2 / h)
        errors = []
        for _, grp in raw.groupby(key):
            have = grp[grp['Race'].isin(index)]
            if len(have) < 2:
                continue
            js = np.array([index[r] for r in have['Race']])
            vals = have['affinity'].to_numpy(dtype=float)
            for k in range(len(js)):
                others = np.arange(len(js)) != k
                weights = w[js[k], js[others]]
                total = weights.sum()
                if total <= 1e-12:
                    continue
                pred = (weights * vals[others]).sum() / total
                errors.append((pred - vals[k]) ** 2)
        if errors:
            scored.append((float(h), float(np.mean(errors)), len(errors)))

    # two baselines the kernel has to beat to be worth anything:
    # ignoring similarity entirely, and having no affinity at all
    flat, zero = [], []
    for _, grp in raw.groupby(key):
        have = grp[grp['Race'].isin(index)]
        if len(have) < 2:
            continue
        vals = have['affinity'].to_numpy(dtype=float)
        for k in range(len(vals)):
            others = np.arange(len(vals)) != k
            flat.append((vals[others].mean() - vals[k]) ** 2)
            zero.append(vals[k] ** 2)

    baselines = {
        'flat': float(np.mean(flat)) if flat else np.nan,
        'zero': float(np.mean(zero)) if zero else np.nan,
    }

    if not scored:
        return None, scored, baselines
    best = min(scored, key=lambda t: t[1])
    return best[0], scored, baselines


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    results = load_results(SEASONS)
    df = prepare(results)

    print(f'Signal          : {SIGNAL}')
    print(f'Entries used    : {len(df)} of {len(results)}')
    if 'Status' in results.columns and SIGNAL == 'finish':
        dnf = (~results['Status'].apply(is_classified)).sum()
        print(f'Retirements cut : {dnf}')
    print(f'Seasons         : {sorted(df["Season"].unique())}')
    print(f'Circuits        : {df["Race"].nunique()}\n')

    drivers = affinity_for(df, 'Driver')
    teams = affinity_for(df, 'Team')

    if drivers.empty:
        print('Not enough data for driver affinity.')
        return

    print('--- strongest circuit affinities, drivers ---')
    top = drivers.reindex(drivers['affinity'].abs().sort_values(ascending=False).index)
    print(top.head(20).round(2).to_string(index=False))

    if not teams.empty:
        print('\n--- strongest circuit affinities, teams ---')
        top_t = teams.reindex(teams['affinity'].abs().sort_values(ascending=False).index)
        print(top_t.head(12).round(2).to_string(index=False))

    print(f'\nDriver affinity spread: '
          f'{drivers["affinity"].min():.2f} to {drivers["affinity"].max():.2f} positions')
    print(f'Standard deviation    : {drivers["affinity"].std():.2f} positions')

    teams_out = teams
    if AFFINITY_POOLING and not teams.empty:
        features, cols = load_track_features()
        if features is None:
            print('\n! track_features.csv missing - pooling skipped. '
                  'Run: python -m Simülasyon.data_prep.track_features')
        else:
            print(f'\n--- similarity pooling over {len(features)} circuits '
                  f'({len(cols)} features) ---')
            h, scored, base = choose_bandwidth(teams, features, cols, 'Team')
            if h is None:
                print('  not enough paired circuits to choose a bandwidth')
            else:
                lo, hi = float(H_GRID[0]), float(H_GRID[-1])
                err = dict((a, b) for a, b, _ in scored)
                print(f'  bandwidth h = {h:.2f} by leave-one-circuit-out '
                      f'(mse {err[h]:.4f}, {scored[0][2]} predictions)')
                print(f'  ignoring similarity entirely : mse {base["flat"]:.4f}')
                print(f'  predicting no affinity at all: mse {base["zero"]:.4f}')

                # An interior optimum is the thing to look for. Without one the
                # kernel has no width it prefers, which means the features are
                # not telling it anything about which circuits go together.
                on_edge = h <= lo or h >= hi
                beats_flat = err[h] < base['flat'] - 1e-9
                beats_zero = err[h] < base['zero'] - 1e-9

                if on_edge:
                    print(f'  ! h sits on the edge of the grid [{lo:g}, {hi:g}] - '
                          'no interior optimum.')
                    print('    The kernel has no width it prefers, which is what an '
                          'unidentified parameter looks like.')
                if not beats_flat:
                    print('  ! similarity weighting does not beat ignoring '
                          'similarity. The features are not')
                    print('    explaining affinity; pooling here is averaging, '
                          'not borrowing.')
                if not beats_zero:
                    print('  ! nothing here beats predicting zero affinity. There '
                          'is no structure to transfer')
                    print('    between circuits - the raw numbers rest on two or '
                          'three races each.')
                if not (beats_flat and beats_zero and not on_edge):
                    print('  -> treat the pooled output as provisional. '
                          'AFFINITY_POOLING is off by default for this reason.')

                pooled = pool_affinity(teams, features, cols, h, 'Team')
                if not pooled.empty:
                    merged = pooled.merge(
                        teams[['Team', 'Race', 'n_samples', 'spread', 'seasons']],
                        on=['Team', 'Race'], how='left')
                    shift = (merged['affinity'] - merged['affinity_raw']).abs()
                    print(f'  circuits per team after pooling: '
                          f'{merged.groupby("Team")["Race"].nunique().mean():.1f} '
                          f'(was {teams.groupby("Team")["Race"].nunique().mean():.1f})')
                    print(f'  mean shift where a raw value existed: '
                          f'{shift.mean():.3f} positions')
                    biggest = merged.loc[shift.sort_values(ascending=False).index[:5]]
                    print('  largest moves:')
                    for _, r in biggest.iterrows():
                        if pd.isna(r['affinity_raw']):
                            continue
                        print(f'    {r["Team"]:18s} {r["Race"]:28s} '
                              f'{r["affinity_raw"]:+.2f} -> {r["affinity"]:+.2f} '
                              f'(own weight {r["own_weight"]:.0%})')
                    teams_out = merged

    combined = pd.concat([
        drivers.rename(columns={'Driver': 'entity'}).assign(kind='driver'),
        teams_out.rename(columns={'Team': 'entity'}).assign(kind='team'),
    ], ignore_index=True)

    drivers.to_csv(os.path.join(DATA_DIR, 'track_affinity_driver.csv'), index=False)
    teams_out.to_csv(os.path.join(DATA_DIR, 'track_affinity_team.csv'), index=False)
    combined.to_csv(os.path.join(DATA_DIR, 'track_affinity.csv'), index=False)
    print(f'\nSaved 3 files to {DATA_DIR}')


if __name__ == '__main__':
    main()