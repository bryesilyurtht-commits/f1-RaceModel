"""
F1 Prediction Simulation - tyre_cliff.py
v1.1-A / A4: fits the two-segment degradation model to the profiles.

Reads  data/tyre_age_profile.csv   (written by tyre_profile.py)
Writes data/tyre_cliff_params.csv  W and m per compound
       data/tyre_k_by_race.csv     k per season, race, compound
       data/tyre_k_by_track.csv    k pooled across seasons
       data/tyre_cliff_fits.csv    per-profile detail, for inspection

Model
-----
    loss(a) = k * a                        a <= c
    loss(a) = k * c + k * m * (a - c)      a >  c
    c = W / k

loss(c) = W, so the cliff always arrives at W seconds of accumulated loss
whatever k is. W and m belong to the compound, k to the track.

Two stages, not one
-------------------
The first version searched k, W and m together and came back with W pinned to
the bottom of its grid for two compounds and m pinned to 1.00 for a third -
all sitting exactly on a boundary, which is what least squares returns when a
parameter is not identified: nothing stops the slide, so it slides until the
grid ends. It dragged k along with it and inverted the compound order.

So k is measured first, alone, from the early part of each profile. Before the
cliff the curve is a straight line through the origin and k is its slope, with
no cliff parameter involved. Every compound is measured through the same
window, which is what makes their k values comparable and the SOFT/MEDIUM
ratio meaningful.

W and m are fitted afterwards with k held fixed, and only on profiles long
enough to have seen a cliff. Short stints say nothing about where a tyre falls
off; including them just adds points any W explains equally well, which is how
the boundary solutions happened in the first place.

A boundary optimum is now rejected instead of reported. If the best W or m
sits on the edge of the grid, or too few profiles run past the fitted cliff,
the compound is declared unidentified and CLIFF_FALLBACK is used. Those are
hand-set numbers, the output says so, and they belong on the v2.3 list of
constants to be replaced by measurement.

Usage:
    python tyre_cliff.py
    python tyre_cliff.py --k-max-age 12 --min-stints 3
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

MIN_STINTS = 3             # stints behind a cell before it is used at all
MIN_BINS = 3               # points a profile needs to give a slope

# --- stage 1: k -------------------------------------------------------------
K_MAX_AGE = 12             # only bins at or below this age define the slope
K_MIN_POINTS = 3
K_CLAMP = (0.0, 0.30)      # negative wear is not physical

# --- stage 2: W and m -------------------------------------------------------
CLIFF_MIN_MAX_AGE = 24     # a profile must reach this age to say anything
CLIFF_MIN_PROFILES = 10    # per compound
CLIFF_MIN_PAST = 0.25      # share of profiles that must run past the cliff

W_GRID = np.round(np.arange(0.40, 6.01, 0.10), 2)
M_GRID = np.round(np.arange(1.00, 6.01, 0.10), 2)

# Used only when the fit is rejected. Hand-set, and flagged as such.
CLIFF_FALLBACK = {
    'SOFT':   {'W': 1.2, 'm': 2.6},
    'MEDIUM': {'W': 1.8, 'm': 2.2},
    'HARD':   {'W': 2.4, 'm': 1.9},
}


def load_profiles(min_stints):
    prof = pd.read_csv(os.path.join(DATA_DIR, 'tyre_age_profile.csv'))
    prof = prof[prof['Compound'].isin(DRY_COMPOUNDS)]

    groups = []
    thin = 0
    for (season, race, comp), g in prof.groupby(['Season', 'Race', 'Compound']):
        g = g.sort_values('age')
        if len(g) < MIN_BINS:
            continue
        if 'n_stints' in g.columns and g['n_stints'].max() < min_stints:
            thin += 1
            continue
        a = g['age'].to_numpy(float)
        y = g['loss_s'].to_numpy(float)
        w = g['n_laps'].to_numpy(float)
        w = w / w.sum()
        if 'is_reference' in g.columns and g['is_reference'].any():
            ref = float(g.loc[g['is_reference'], 'age'].iloc[0])
        else:
            ref = float(a[0])
        groups.append({'Season': season, 'Race': race, 'Compound': comp,
                       'a': a, 'y': y, 'w': w, 'a_ref': ref,
                       'max_age': float(a.max()), 'n_bins': len(g),
                       'n_stints': int(g['n_stints'].max())
                       if 'n_stints' in g.columns else 0})
    if thin:
        print(f'Cells dropped for too few stints (<{min_stints}): {thin}')
    return groups


# --- stage 1 ----------------------------------------------------------------


def fit_k(grp, max_age):
    """
    Slope of the linear region, through the reference point.

    The profile is measured relative to its youngest bin, so the quantity
    fitted is y(a) = k * (a - a_ref): a regression through the origin. No
    intercept, because the reference bin is zero by construction and letting
    the line float would spend a degree of freedom re-estimating a known value.
    """
    m = grp['a'] <= max_age
    if m.sum() < K_MIN_POINTS:
        return np.nan, 0
    x = grp['a'][m] - grp['a_ref']
    y = grp['y'][m]
    w = grp['w'][m]
    denom = float((w * x * x).sum())
    if denom <= 1e-12:
        return np.nan, int(m.sum())
    k = float((w * x * y).sum() / denom)
    return float(np.clip(k, *K_CLAMP)), int(m.sum())


# --- stage 2 ----------------------------------------------------------------


def sse_grid(grp, k):
    """Weighted SSE over the whole (W, m) grid for one profile, k fixed."""
    a = grp['a'][None, None, :]
    y = grp['y'][None, None, :]
    w = grp['w'][None, None, :]
    W = W_GRID[:, None, None]
    m = M_GRID[None, :, None]

    c = W / k
    f = np.where(a <= c, k * a, W + k * m * (a - c))
    aref = np.full((1, 1, 1), grp['a_ref'])
    fref = np.where(aref <= c, k * aref, W + k * m * (aref - c))
    pred = f - fref
    return (w * (pred - y) ** 2).sum(axis=2)


def fit_shape(groups, comp, k_by_profile):
    sub = [g for g in groups
           if g['Compound'] == comp
           and g['max_age'] >= CLIFF_MIN_MAX_AGE
           and np.isfinite(k_by_profile.get(id(g), np.nan))
           and k_by_profile[id(g)] > 1e-6]

    if len(sub) < CLIFF_MIN_PROFILES:
        return {'ok': False, 'n_long': len(sub),
                'reason': f'only {len(sub)} long profiles '
                          f'(need {CLIFF_MIN_PROFILES})'}

    total = np.zeros((len(W_GRID), len(M_GRID)))
    for g in sub:
        total += sse_grid(g, k_by_profile[id(g)])

    iw, im = np.unravel_index(int(np.argmin(total)), total.shape)
    W, m = float(W_GRID[iw]), float(M_GRID[im])
    past = float(np.mean([g['max_age'] > W / k_by_profile[id(g)] for g in sub]))
    on_edge = (iw in (0, len(W_GRID) - 1)) or (im in (0, len(M_GRID) - 1))

    if on_edge:
        return {'ok': False, 'n_long': len(sub), 'W_raw': W, 'm_raw': m,
                'share_past_cliff': past,
                'reason': f'optimum on the grid boundary (W={W}, m={m})'}
    if past < CLIFF_MIN_PAST:
        return {'ok': False, 'n_long': len(sub), 'W_raw': W, 'm_raw': m,
                'share_past_cliff': past,
                'reason': f'only {past:.0%} of profiles run past the fitted '
                          f'cliff (need {CLIFF_MIN_PAST:.0%})'}

    return {'ok': True, 'W': W, 'm': m, 'n_long': len(sub),
            'share_past_cliff': past, 'reason': ''}


def main():
    k_max_age = K_MAX_AGE
    min_stints = MIN_STINTS
    if '--k-max-age' in sys.argv:
        k_max_age = float(sys.argv[sys.argv.index('--k-max-age') + 1])
    if '--min-stints' in sys.argv:
        min_stints = int(sys.argv[sys.argv.index('--min-stints') + 1])

    print(f'k window       : tyre age <= {k_max_age} laps')
    print(f'Cliff profiles : reaching >= {CLIFF_MIN_MAX_AGE} laps')
    print(f'Grid           : W {W_GRID[0]}-{W_GRID[-1]}, '
          f'm {M_GRID[0]}-{M_GRID[-1]}\n')

    groups = load_profiles(min_stints)
    print(f'Profiles loaded : {len(groups)}')
    for c in DRY_COMPOUNDS:
        n = sum(1 for g in groups if g['Compound'] == c)
        nl = sum(1 for g in groups
                 if g['Compound'] == c and g['max_age'] >= CLIFF_MIN_MAX_AGE)
        print(f'   {c:7s} {n:4d}   long enough for a cliff: {nl}')

    # --- stage 1 ------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STAGE 1: k from the linear region')
    print('=' * 70)

    k_by_profile, rows = {}, []
    for g in groups:
        k, npts = fit_k(g, k_max_age)
        k_by_profile[id(g)] = k
        rows.append({'Season': g['Season'], 'Race': g['Race'],
                     'Compound': g['Compound'], 'k': k, 'k_points': npts,
                     'max_age': g['max_age'], 'n_bins': g['n_bins'],
                     'n_stints': g['n_stints']})
    kdf = pd.DataFrame(rows)
    kdf = kdf[kdf['k'].notna()]

    print(f'Slopes fitted        : {len(kdf)}')
    print(f'Clamped at zero      : {(kdf["k"] <= 1e-9).sum()}\n')
    print(kdf.groupby('Compound')
             .agg(n=('k', 'size'), median_k=('k', 'median'),
                  q10=('k', lambda s: s.quantile(0.10)),
                  q90=('k', lambda s: s.quantile(0.90)),
                  zero=('k', lambda s: (s <= 1e-9).mean()))
             .round(4).to_string())
    print('\nExpected order is SOFT > MEDIUM > HARD.')

    # --- stage 2 ------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STAGE 2: W and m, pooled per compound, k held fixed')
    print('=' * 70)

    results = []
    for comp in DRY_COMPOUNDS:
        res = fit_shape(groups, comp, k_by_profile)
        if res['ok']:
            W, m, src = res['W'], res['m'], 'measured'
            print(f'  {comp:7s} W = {W:.2f} s   m = {m:.2f}   '
                  f'({res["n_long"]} profiles, '
                  f'{res["share_past_cliff"]:.0%} past the cliff)')
        else:
            fb = CLIFF_FALLBACK[comp]
            W, m, src = fb['W'], fb['m'], 'hand-set'
            print(f'  {comp:7s} NOT IDENTIFIED - {res["reason"]}')
            print(f'          using W = {W:.2f}, m = {m:.2f} (hand-set)')
        results.append({'Compound': comp, 'W': W, 'm': m, 'shape_source': src,
                        'n_long_profiles': res.get('n_long', 0),
                        'share_past_cliff': res.get('share_past_cliff', np.nan),
                        'reject_reason': res.get('reason', '')})

    params = pd.DataFrame(results)

    # --- k by track ---------------------------------------------------------
    by_track = (kdf.groupby(['Race', 'Compound'])
                   .agg(k=('k', 'median'),
                        k_std=('k', 'std'),
                        n_seasons=('Season', 'nunique'),
                        n_profiles=('k', 'size'),
                        n_stints=('n_stints', 'sum'))
                   .reset_index())
    # build_tyre_model gates on this; the two-stage fit has no per-cell R2, so
    # the gate is carried by stint count instead and this stays neutral
    by_track['median_r2'] = 1.0

    print('\n' + '=' * 70)
    print('k BY TRACK (s/lap of tyre age, linear region)')
    print('=' * 70)
    piv = by_track.pivot(index='Race', columns='Compound', values='k')
    key = 'MEDIUM' if 'MEDIUM' in piv.columns else piv.columns[0]
    order = piv[key].sort_values(ascending=False).index
    print(piv.reindex(order).round(4).to_string())

    print('\n' + '=' * 70)
    print('IMPLIED CLIFF LAP  (c = W / k)')
    print('=' * 70)
    wmap = params.set_index('Compound')['W'].to_dict()
    cl = by_track.copy()
    cl['cliff_lap'] = [wmap[c] / k if k > 1e-6 else np.nan
                       for c, k in zip(cl['Compound'], cl['k'])]
    print(cl.pivot(index='Race', columns='Compound', values='cliff_lap')
            .reindex(order).round(1).to_string())

    # --- ratios -------------------------------------------------------------
    print('\n' + '=' * 70)
    print('COMPOUND RATIOS (from stage-1 k, same window for all three)')
    print('=' * 70)
    wide = by_track.pivot(index='Race', columns='Compound', values='k')
    ratios = {}
    for comp in ['SOFT', 'HARD']:
        if {comp, 'MEDIUM'} <= set(wide.columns):
            both = wide.dropna(subset=[comp, 'MEDIUM'])
            both = both[both['MEDIUM'] > 1e-6]
            if len(both):
                r = both[comp] / both['MEDIUM']
                ratios[comp] = float(r.median())
                print(f'{comp} / MEDIUM : median {r.median():.3f}   '
                      f'range {r.min():.2f}-{r.max():.2f}   '
                      f'over {len(both)} tracks')

    if ratios.get('SOFT', 0) <= 1.0 or ratios.get('HARD', 9) >= 1.0:
        print('\nThe ordering is wrong: SOFT should wear faster than MEDIUM')
        print('and HARD slower. Fix stage 1 before A5 derives anything from')
        print('these ratios.')

    params['soft_over_medium'] = ratios.get('SOFT', np.nan)
    params['hard_over_medium'] = ratios.get('HARD', np.nan)

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(params.round(3).to_string(index=False))

    print()
    for fname, frame in [('tyre_cliff_params.csv', params),
                         ('tyre_k_by_race.csv', kdf),
                         ('tyre_k_by_track.csv', by_track),
                         ('tyre_cliff_fits.csv', kdf)]:
        p = os.path.join(DATA_DIR, fname)
        frame.to_csv(p, index=False)
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()