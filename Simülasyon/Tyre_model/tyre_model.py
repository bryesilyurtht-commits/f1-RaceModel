"""
F1 Prediction Simulation - tyre_model.py
Parameter module: serves the measured degradation curve to the simulation.

Reads data/tyre_model.csv, built by build_tyre_model.py.

The curve
---------
Tyre age a, in laps. The value returned is the pace penalty on that lap, in
seconds, versus a fresh tyre of the same compound:

    penalty(a) = k * a                       a <= c
    penalty(a) = k * c + k * m * (a - c)     a >  c
    c = W / k

k is the track's abrasion, W and m are the compound's. The cliff arrives once
the tyre has accumulated W seconds of loss, so an abrasive track reaches it
early and a smooth one may never reach it at all. No cliff lap is hand-set.

This replaces the single `deg` coefficient the simulation used through v1.0,
where SOFT and HARD aged at the same rate and a soft tyre could therefore run
half a race without consequence.

Relationship to the other degradation numbers:
    tyre_model  - track x compound, the curve itself
    deg.py      - driver and team, deg_index, who is kind on tyres
The two multiply. deg.py deliberately normalises the track out; this module
is where the track comes back in.

Usage:
    from tyre_model import lap_penalty, get_params, cliff_lap, stint_limit

    lap_penalty('Dutch Grand Prix', 'SOFT', 14)          -> seconds
    lap_penalty('Dutch Grand Prix', 'SOFT', ages_array)  -> array
    lap_penalty('Dutch Grand Prix', 'SOFT', 14, deg_index=0.9)

    python tyre_model.py            # print the table and a few sample curves
"""

import os

import numpy as np
import pandas as pd

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'tyre_model.csv')
LIMITS_PATH = os.path.join(DATA_DIR, 'stint_limits.csv')

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

# a stint is treated as finished once the tyre costs this much per lap. Only
# used where stint_limits.csv has nothing to say: the measured cap is better
# evidence than a threshold on a curve whose cliff was never observed.
STINT_LIMIT_PENALTY = 2.5

# used only when the CSV is missing entirely, so a run does not simply die
FALLBACK = {
    'SOFT':   {'k': 0.075, 'W': 1.2, 'm': 2.6},
    'MEDIUM': {'k': 0.050, 'W': 1.6, 'm': 2.2},
    'HARD':   {'k': 0.038, 'W': 2.0, 'm': 1.9},
}

_TABLE = None
_INDEX = None
_LIMITS = None


# --- loading ----------------------------------------------------------------


def _normalise(name):
    key = str(name).lower()
    for src, dst in (('ã', 'a'), ('á', 'a'), ('à', 'a'), ('â', 'a'),
                     ('é', 'e'), ('è', 'e'), ('ê', 'e'), ('í', 'i'),
                     ('ó', 'o'), ('ô', 'o'), ('õ', 'o'), ('ú', 'u'),
                     ('ü', 'u'), ('ñ', 'n'), ('ç', 'c')):
        key = key.replace(src, dst)
    return key.replace('grand prix', '').replace('gp', '').strip()


def load(path=MODEL_PATH, force=False):
    """Loads the table once and keeps it in memory."""
    global _TABLE, _INDEX
    if _TABLE is not None and not force:
        return _TABLE

    if not os.path.exists(path):
        print(f'[tyre_model] {path} not found, using fallback constants. '
              f'Run build_tyre_model.py.')
        _TABLE = pd.DataFrame([
            {'Race': '_FALLBACK', 'Compound': c, 'k': v['k'], 'W': v['W'],
             'm': v['m'], 'cliff_lap': v['W'] / v['k'], 'source': 'fallback',
             'note': 'tyre_model.csv missing'}
            for c, v in FALLBACK.items()])
    else:
        _TABLE = pd.read_csv(path)
        _TABLE['Compound'] = _TABLE['Compound'].astype(str).str.upper()

    _INDEX = {}
    for _, r in _TABLE.iterrows():
        _INDEX[(_normalise(r['Race']), r['Compound'])] = {
            'k': float(r['k']), 'W': float(r['W']), 'm': float(r['m']),
            'source': r.get('source', ''), 'Race': r['Race']}
    return _TABLE


def _global_params(compound):
    tbl = load()
    sub = tbl[tbl['Compound'] == compound]
    if sub.empty:
        f = FALLBACK.get(compound, FALLBACK['MEDIUM'])
        return {'k': f['k'], 'W': f['W'], 'm': f['m'], 'source': 'fallback',
                'Race': '_FALLBACK'}
    return {'k': float(sub['k'].median()), 'W': float(sub['W'].median()),
            'm': float(sub['m'].median()), 'source': 'global',
            'Race': '_GLOBAL'}


def get_params(track, compound):
    """
    Returns {'k', 'W', 'm', 'cliff_lap', 'source', 'Race'} for a track and
    compound. Unknown tracks fall back to the calendar-wide median rather
    than raising: a new venue should not stop a race from being simulated.
    """
    load()
    compound = str(compound).upper()
    key = _normalise(track)

    hit = _INDEX.get((key, compound))
    if hit is None:
        # substring match, so 'Dutch' finds 'Dutch Grand Prix' and vice versa
        for (k_track, k_comp), v in _INDEX.items():
            if k_comp == compound and (k_track in key or key in k_track):
                hit = v
                break
    if hit is None:
        hit = _global_params(compound)

    k, W, m = hit['k'], hit['W'], hit['m']
    cliff = W / k if k > 1e-6 and np.isfinite(W) else np.inf
    return {'k': k, 'W': W, 'm': m, 'cliff_lap': cliff,
            'source': hit.get('source', ''), 'Race': hit.get('Race', track)}


# --- the curve --------------------------------------------------------------


def lap_penalty(track, compound, tyre_life, deg_index=1.0):
    """
    Pace penalty in seconds for a tyre of this age, versus a fresh one.

    tyre_life may be a scalar or a numpy array, so the Monte Carlo loop can
    price a whole field in one call.

    deg_index scales the track abrasion by the driver's own tyre usage, the
    number deg.py measures. Below 1.0 is kinder than the field. It scales k,
    which means a kind driver also reaches the cliff later - the cliff sits at
    a fixed amount of accumulated wear, not a fixed lap.
    """
    p = get_params(track, compound)
    k = p['k'] * float(deg_index)
    W, m = p['W'], p['m']

    a = np.asarray(tyre_life, dtype=float)
    if k <= 1e-9 or not np.isfinite(W):
        out = k * a
        return float(out) if np.isscalar(tyre_life) or out.ndim == 0 else out

    c = W / k
    out = np.where(a <= c, k * a, W + k * m * (a - c))
    if np.isscalar(tyre_life) or out.ndim == 0:
        return float(out)
    return out


def cliff_lap(track, compound, deg_index=1.0):
    """Tyre age where degradation steepens. inf when it never does."""
    p = get_params(track, compound)
    k = p['k'] * float(deg_index)
    return p['W'] / k if k > 1e-9 and np.isfinite(p['W']) else np.inf


def load_limits(path=LIMITS_PATH, force=False):
    """
    Longest stint anyone actually ran, per track and compound.

    Built by stint_limits.py from 2022-2025. This is the cap the simulation
    should respect, and it is stronger evidence than anything the lap times
    offer: the cliff cannot be measured because teams never run a tyre into
    it, but the fact that they stop where they stop is itself the measurement.
    """
    global _LIMITS
    if _LIMITS is not None and not force:
        return _LIMITS

    _LIMITS = {}
    if not os.path.exists(path):
        print(f'[tyre_model] {path} not found. Stint caps will fall back to '
              f'the {STINT_LIMIT_PENALTY} s/lap threshold, which rests on a '
              f'cliff that was never observed. Run stint_limits.py.')
        return _LIMITS

    df = pd.read_csv(path)
    for _, r in df.iterrows():
        _LIMITS[(_normalise(r['Race']), str(r['Compound']).upper())] = {
            'limit': float(r['limit']), 'source': r.get('source', '')}
    return _LIMITS


def max_stint(track, compound):
    """
    Measured cap in laps, or inf when the table has no entry.

    Not scaled by deg_index: this is what teams were willing to do with the
    tyre, and a driver being kind on it does not license running a set nobody
    has ever run.
    """
    limits = load_limits()
    if not limits:
        return np.inf
    compound = str(compound).upper()
    key = _normalise(track)
    hit = limits.get((key, compound))
    if hit is None:
        for (t, c), v in limits.items():
            if c == compound and (t in key or key in t):
                hit = v
                break
    return hit['limit'] if hit else np.inf


def max_stint_source(track, compound):
    limits = load_limits()
    compound = str(compound).upper()
    key = _normalise(track)
    hit = limits.get((key, compound))
    if hit is None:
        for (t, c), v in limits.items():
            if c == compound and (t in key or key in t):
                hit = v
                break
    return hit['source'] if hit else 'none'


def stint_limit(track, compound, deg_index=1.0, penalty=STINT_LIMIT_PENALTY):
    """
    Longest sensible stint, in laps.

    Prefers the measured cap from stint_limits.csv. Falls back to the age at
    which the curve reaches `penalty` seconds a lap, which is only meaningful
    where the cliff was actually identified - and for most compounds it was
    not, which is exactly why the measured cap exists.
    """
    measured = max_stint(track, compound)
    if np.isfinite(measured):
        return measured

    p = get_params(track, compound)
    k = p['k'] * float(deg_index)
    W, m = p['W'], p['m']
    if k <= 1e-9:
        return np.inf
    if penalty <= W:
        return penalty / k
    if not np.isfinite(m) or m <= 0:
        return penalty / k
    return W / k + (penalty - W) / (k * m)


def table():
    """The whole table, for diagnostics and reports."""
    return load().copy()


# --- self test --------------------------------------------------------------

if __name__ == '__main__':
    tbl = load()
    print(f'Rows: {len(tbl)}   Tracks: {tbl["Race"].nunique()}\n')

    if 'source' in tbl.columns:
        print('--- provenance ---')
        print(tbl.groupby(['Compound', 'source']).size()
                 .unstack(fill_value=0).to_string())

    print('\n--- sample curves, penalty in seconds ---')
    ages = np.array([5, 10, 15, 20, 25, 30, 40])
    for track in ['Dutch Grand Prix', 'Bahrain Grand Prix',
                  'Monaco Grand Prix', 'British Grand Prix']:
        print(f'\n{track}')
        print('  age    ' + '  '.join(f'{a:>6d}' for a in ages))
        for comp in COMPOUNDS:
            vals = lap_penalty(track, comp, ages)
            p = get_params(track, comp)
            cl = p['cliff_lap']
            cl_s = f'{cl:.0f}' if np.isfinite(cl) else 'never'
            cap = max_stint(track, comp)
            cap_s = f'{cap:.0f}' if np.isfinite(cap) else '-'
            print(f'  {comp:7s}' + '  '.join(f'{v:6.2f}' for v in vals)
                  + f'   cliff {cl_s:>5s}  cap {cap_s:>3s}'
                    f' [{max_stint_source(track, comp)}]')

    print('\n--- driver sensitivity, Zandvoort SOFT at 20 laps ---')
    for di in [0.85, 1.00, 1.15]:
        print(f'  deg_index {di:.2f}: '
              f'{lap_penalty("Dutch Grand Prix", "SOFT", 20, di):.2f} s   '
              f'cliff at {cliff_lap("Dutch Grand Prix", "SOFT", di):.0f} laps')