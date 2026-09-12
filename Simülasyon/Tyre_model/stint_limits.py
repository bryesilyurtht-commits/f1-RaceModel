"""
F1 Prediction Simulation - stint_limits.py
v1.1-A: the longest stint anyone actually ran, per track and compound.

Reads  data/stint_index.csv   (written by stint_table.py)
Writes data/stint_limits.csv

Why this exists
---------------
The cliff cannot be measured from lap times. Teams do not run a tyre off the
edge, so the data is censored on the right: the median SOFT stint is 14 laps
and what the profiles contain is the first fourteen laps of a soft tyre, which
really are gentle. Three separate estimators all reported SOFT ageing slowly,
and all three were right about the data and wrong about the tyre.

But the censoring is itself the measurement. A team that pits at lap 22 on
softs is telling you something they know and the lap times do not show. So
instead of asking the lap times how bad a 43-lap soft stint would be, this
asks whether anyone has ever run one. Across 2022-2025, at that circuit, on
that compound, what was the longest anybody went.

That number becomes a hard cap. It is not a guess about grip; it is an
observed bound on how the tyre gets used, which is the thing the simulation
was getting wrong.

Which statistic
---------------
Not the maximum. Across 2022-2025 the longest HARD and the longest MEDIUM
stint are both 77 laps, and both come from the same afternoon: Monaco 2024 was
red-flagged on lap one, everyone changed tyres for free, and most of the field
then ran the full distance on one set. One unusual race was setting the cap
for two compounds at once, and the ordering came out impossible - SOFT lasting
longer than MEDIUM at six circuits.

The 95th percentile fixes it without any hand-tuning: 47 laps for HARD, 37 for
MEDIUM, 29 for SOFT, which is the order a tyre engineer would expect. It reads
as "the longest anyone routinely goes" rather than "the longest anyone has
ever gone", and the second question was never the useful one.

Ordering is enforced on top of that, per circuit, clipping the softer compound
down where a cell still comes out above its neighbour.

Stints that ran to the flag were never ended by a decision, so the true limit
at those cells may be higher than what was seen. The share of such stints is
reported per cell.

Usage:
    python stint_limits.py
    python stint_limits.py --stat max
"""

import os
import sys

import numpy as np
import pandas as pd

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

MAX_STAT = 'p95'          # 'max' or 'p95'
MIN_STINTS_PER_CELL = 3   # below this the cell is filled from elsewhere

ERA_BREAK = 2022          # 18-inch wheels from here; see split_eras

# SOFT cannot outlast MEDIUM, and MEDIUM cannot outlast HARD. Where the
# measured caps say otherwise the softer compound is clipped down to its
# neighbour, never the other way round: a cap that is too tight costs a few
# seconds of strategy, a cap that is too loose puts a car on softs for half
# the race.
ENFORCE_ORDERING = True

# a cap can only ever be as long as the race
ABSOLUTE_CEILING = 80


def load_index():
    path = os.path.join(DATA_DIR, 'stint_index.csv')
    df = pd.read_csv(path)
    if 'usable' in df.columns:
        df = df[df['usable']]
    df = df[df['Compound'].isin(COMPOUNDS)]
    df['stint_length'] = pd.to_numeric(df['stint_length'], errors='coerce')
    df = df[df['stint_length'].notna()]
    if 'tyre_era' not in df.columns:
        df['tyre_era'] = np.where(df['Season'] >= ERA_BREAK, '18inch', '13inch')
    return df


def split_eras(idx):
    """
    Current-era stints, and the older ones kept aside to fill gaps.

    The two are not interchangeable. Measured over the circuits that appear in
    both, a 13-inch stint runs longer than an 18-inch one on every compound -
    HARD 32 laps against 27, MEDIUM 23 against 20, SOFT 20 against 16 - so a
    cap pooled over both eras is loose by several laps, and a loose cap is the
    one failure this file exists to prevent. The older seasons are here for the
    cells 2022-2025 cannot measure at all, and for nothing else.
    """
    return idx[idx['tyre_era'] == '18inch'], idx[idx['tyre_era'] == '13inch']


def build(idx, stat):
    def top(s):
        return float(s.max()) if stat == 'max' else float(s.quantile(0.95))

    cells = (idx.groupby(['Race', 'Compound'])
                .agg(limit=('stint_length', top),
                     max_stint=('stint_length', 'max'),
                     p95_stint=('stint_length', lambda s: s.quantile(0.95)),
                     median_stint=('stint_length', 'median'),
                     n_stints=('stint_length', 'size'),
                     n_seasons=('Season', 'nunique'),
                     to_flag=('runs_to_flag', 'mean'))
                .reset_index())

    thin = cells['n_stints'] < MIN_STINTS_PER_CELL
    cells['source'] = np.where(thin, 'thin', 'measured')

    # global fallbacks, and the ratio each compound holds against MEDIUM so a
    # missing cell can borrow the track's own character rather than the
    # calendar's average
    solid = cells[~thin]
    global_limit = solid.groupby('Compound')['limit'].median().to_dict()
    med_by_track = (solid[solid['Compound'] == 'MEDIUM']
                    .set_index('Race')['limit'].to_dict())
    ratio = {}
    for c in COMPOUNDS:
        sub = solid[solid['Compound'] == c]
        pairs = [(r, v) for r, v in zip(sub['Race'], sub['limit'])
                 if r in med_by_track and med_by_track[r] > 0]
        ratio[c] = (float(np.median([v / med_by_track[r] for r, v in pairs]))
                    if pairs else 1.0)

    rows = list(cells.to_dict('records'))
    have = {(r['Race'], r['Compound']) for r in rows}

    for track in sorted(set(idx['Race'])):
        for comp in COMPOUNDS:
            if (track, comp) in have:
                continue
            if track in med_by_track:
                limit = med_by_track[track] * ratio.get(comp, 1.0)
                src = f'derived:MEDIUM x {ratio.get(comp, 1.0):.2f}'
            else:
                limit = global_limit.get(comp, 30.0)
                src = 'global'
            rows.append({'Race': track, 'Compound': comp, 'limit': limit,
                         'max_stint': np.nan, 'p95_stint': np.nan,
                         'median_stint': np.nan, 'n_stints': 0,
                         'n_seasons': 0, 'to_flag': np.nan, 'source': src})

    out = pd.DataFrame(rows)

    # thin cells keep their observation but are backed by the derived value,
    # whichever is larger: one short stint should not cap a compound
    for i, r in out.iterrows():
        if r['source'] != 'thin':
            continue
        base = (med_by_track.get(r['Race'], np.nan) * ratio.get(r['Compound'], 1.0))
        if np.isfinite(base):
            out.at[i, 'limit'] = max(r['limit'], base)
            out.at[i, 'source'] = 'thin+derived'
        else:
            out.at[i, 'limit'] = max(r['limit'], global_limit.get(r['Compound'], 30.0))
            out.at[i, 'source'] = 'thin+global'

    out['limit'] = out['limit'].clip(upper=ABSOLUTE_CEILING).round(0)
    return out.sort_values(['Race', 'Compound']).reset_index(drop=True)


def enforce_ordering(tbl):
    """
    Clips SOFT down to MEDIUM and MEDIUM down to HARD, per circuit.

    Only downward. Raising a cap because a neighbour is high would be
    inventing evidence; lowering one because a neighbour is low is just
    refusing to believe an ordering that no tyre produces.

    Returns (table, list of changes).
    """
    changes = []
    wide = tbl.pivot(index='Race', columns='Compound', values='limit')

    for track, row in wide.iterrows():
        hard = row.get('HARD', np.nan)
        med = row.get('MEDIUM', np.nan)
        soft = row.get('SOFT', np.nan)

        if np.isfinite(hard) and np.isfinite(med) and med > hard:
            changes.append((track, 'MEDIUM', med, hard))
            med = hard
        if np.isfinite(med) and np.isfinite(soft) and soft > med:
            changes.append((track, 'SOFT', soft, med))
            soft = med

        for comp, val in (('MEDIUM', med), ('SOFT', soft)):
            mask = (tbl['Race'] == track) & (tbl['Compound'] == comp)
            if mask.any() and np.isfinite(val):
                tbl.loc[mask, 'limit'] = val

    if changes:
        for track, comp, was, now in changes:
            mask = (tbl['Race'] == track) & (tbl['Compound'] == comp)
            tbl.loc[mask, 'source'] = tbl.loc[mask, 'source'] + '+ordered'
    return tbl, changes


def main():
    stat = MAX_STAT
    if '--stat' in sys.argv:
        stat = sys.argv[sys.argv.index('--stat') + 1]

    idx = load_index()
    modern, legacy = split_eras(idx)
    print(f'Stints      : {len(idx)}  '
          f'({len(modern)} on 18-inch, {len(legacy)} on 13-inch)')
    print(f'Tracks      : {idx["Race"].nunique()}')
    print(f'Statistic   : {stat}\n')

    # the current era decides every cell it can measure; the older seasons
    # only answer where it cannot
    tbl = build(modern, stat)
    solid = tbl['n_stints'] >= MIN_STINTS_PER_CELL

    if len(legacy):
        old = build(legacy, stat).set_index(['Race', 'Compound'])
        filled = 0
        for i, row in tbl[~solid].iterrows():
            key = (row['Race'], row['Compound'])
            if key in old.index and old.loc[key, 'n_stints'] >= MIN_STINTS_PER_CELL:
                tbl.loc[i, 'limit'] = old.loc[key, 'limit']
                tbl.loc[i, 'n_stints'] = old.loc[key, 'n_stints']
                tbl.loc[i, 'source'] = 'legacy-era'
                filled += 1

        # tracks the current era never visits at all
        missing = set(map(tuple, old.index)) - set(
            map(tuple, tbl[['Race', 'Compound']].values))
        extra = []
        for race, comp in sorted(missing):
            r = old.loc[(race, comp)]
            if r['n_stints'] < MIN_STINTS_PER_CELL:
                continue
            row = r.to_dict()
            row.update({'Race': race, 'Compound': comp, 'source': 'legacy-era'})
            extra.append(row)
        if extra:
            tbl = pd.concat([tbl, pd.DataFrame(extra)], ignore_index=True)
        print(f'Era fill    : {filled} thin cell(s) taken from 13-inch, '
              f'{len(extra)} track/compound(s) only in 13-inch\n')

    changes = []
    if ENFORCE_ORDERING:
        tbl, changes = enforce_ordering(tbl)

    print('=' * 70)
    print('STINT LIMIT BY TRACK (laps)')
    print('=' * 70)
    piv = tbl.pivot(index='Race', columns='Compound', values='limit')
    print(piv.reindex(piv['MEDIUM'].sort_values().index).to_string())

    print('\n' + '=' * 70)
    print('ORDERING CORRECTIONS')
    print('=' * 70)
    if not changes:
        print('  none - the measured caps were already ordered')
    else:
        print(f'{len(changes)} cell(s) clipped down to their neighbour:\n')
        for track, comp, was, now in changes:
            print(f'  {track:30s} {comp:7s} {was:5.0f} -> {now:.0f}')

    print('\n' + '=' * 70)
    print('HOW EACH CELL WAS FILLED')
    print('=' * 70)
    print(tbl.groupby(['Compound', 'source']).size().unstack(fill_value=0).to_string())

    print('\n' + '=' * 70)
    print('LONGEST EVER vs LONGEST ROUTINELY')
    print('=' * 70)
    print('A wide gap means the cap rests on one unusual race. Monaco 2024 was')
    print('red-flagged on lap one, so most of the field ran the whole distance')
    print('on one set - real, but not a normal Monaco.\n')
    m = tbl[tbl['n_stints'] >= MIN_STINTS_PER_CELL].copy()
    m['gap'] = m['max_stint'] - m['p95_stint']
    print(m.nlargest(12, 'gap')[['Race', 'Compound', 'median_stint',
                                 'p95_stint', 'max_stint', 'n_stints']]
           .to_string(index=False))

    print('\n' + '=' * 70)
    print('CELLS WHERE THE LIMIT IS PROBABLY UNDERSTATED')
    print('=' * 70)
    print('A stint that ran to the flag was never ended by a decision, so the')
    print('tyre might have gone further. Where most stints end that way, the')
    print('cap is a floor rather than a bound.\n')
    flag = tbl[(tbl['n_stints'] >= MIN_STINTS_PER_CELL) & (tbl['to_flag'] > 0.5)]
    if flag.empty:
        print('  none')
    else:
        print(flag[['Race', 'Compound', 'limit', 'to_flag', 'n_stints']]
              .round(2).to_string(index=False))

    print('\n' + '=' * 70)
    print('COMPOUND SUMMARY')
    print('=' * 70)
    print(tbl.groupby('Compound')
             .agg(median_limit=('limit', 'median'),
                  min_limit=('limit', 'min'),
                  max_limit=('limit', 'max'))
             .to_string())
    print('\nExpected order is HARD > MEDIUM > SOFT.')

    out = os.path.join(DATA_DIR, 'stint_limits.csv')
    tbl.to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()