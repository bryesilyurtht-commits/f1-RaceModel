"""
F1 Prediction Simulation - pit_analysis.py
Per-track pit stop counts, pit windows, compound scaling and the strategy space.

What it produces
----------------
1. Stop count distribution per track, and an upper bound = max observed + 1.
2. Pit window per compound: the 25th-75th percentile of observed stint lengths.
3. Compound scaling in seconds relative to MEDIUM, measured with fuel and tyre
   age held constant:

       lap_time = driver_effect + b*LapNumber + c*TyreLife + compound_offset

   Driver effects are removed by within-driver demeaning, so a compound does not
   look fast just because the fast cars ran it.
4. The strategy space: every compound sequence up to the upper bound, using only
   compounds actually seen at that track, filtered by the FIA two-compound rule.
   Each plan is priced twice. `est_cost` is this file's own estimate: one deg
   slope shared by up to two compounds, from compound_offsets() above.
   `est_cost_curve` is the same plan priced against Tyre_model's fitted
   per-compound curve (b1, b2, cliff), the one simulate.py actually runs on.
   Both are kept - est_cost is this file's independent check, est_cost_curve
   is what the simulation will really charge - and simulate.py still recosts
   everything itself before a race, so neither number here decides the final
   run. A plan can rank well on one and poorly on the other; that gap is the
   flat-slope assumption showing up in the open, not a bug in either number.

Outputs:
    data/pit_by_race.csv          race level
    data/pit_summary.csv          track level
    data/compound_scaling.csv     track level compound offsets
    data/pit_strategies.csv       track x strategy
"""

import os
import warnings
from itertools import product

import numpy as np
import pandas as pd

from Simülasyon.dataset import load_laps, clean_race, iter_races
from Simülasyon.fuel_effect import get_track_params
from Simülasyon.Tyre_model.tyre_curve import AGE_OFFSET
from Simülasyon.Tyre_model.tyre_store import UnknownProfile
from Simülasyon.Tyre_model.tyre_store import load as load_tyre_store

warnings.filterwarnings('ignore')

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]

DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
REFERENCE_COMPOUND = 'MEDIUM'

MAX_WET_SHARE = 0.15

MIN_RACE_SHARE = 0.80         # driver must complete this share of the race distance
MIN_STINT_LAPS = 4            # shorter stints are damage / red flag artefacts
MIN_LAPS_PER_RACE = 200

MIN_COMPOUND_SHARE = 0.02     # compound must be this common at the track to count
MIN_LAPS_FOR_OWN_DEG = 40     # laps a compound needs before it gets its own slope
HARD_STOP_CAP = 3             # never enumerate beyond this many stops
WINDOW_TOLERANCE = 0.30       # a stint may sit this far outside the observed window

# --- cleaning ---------------------------------------------------------------


# --- stint and stop structure ----------------------------------------------


def stint_table(laps, race_laps):
    """One row per driver stint, using all laps rather than clean laps only."""
    df = laps[laps['Compound'].isin(DRY_COMPOUNDS)]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    stints = (df.groupby(['Driver', 'Stint'])
                .agg(Compound=('Compound', lambda s: s.mode().iloc[0]),
                     start_lap=('LapNumber', 'min'),
                     end_lap=('LapNumber', 'max'),
                     stint_laps=('LapNumber', 'size'))
                .reset_index())
    stints = stints[stints['stint_laps'] >= MIN_STINT_LAPS]
    if stints.empty:
        return pd.DataFrame(), pd.DataFrame()

    # only drivers who went near full distance define the stop count
    covered = stints.groupby('Driver')['stint_laps'].sum()
    finishers = covered[covered >= MIN_RACE_SHARE * race_laps].index
    stops = (stints[stints['Driver'].isin(finishers)]
             .groupby('Driver').size().rename('n_stints').reset_index())
    stops['n_stops'] = stops['n_stints'] - 1

    return stints, stops


def pit_loss_estimate(laps, clean):
    """Rough pit loss: in-lap plus out-lap minus two clean laps."""
    df = laps
    base = clean.groupby('Driver')['LapTime_s'].median()

    losses = []
    for driver, grp in df.groupby('Driver'):
        if driver not in base.index:
            continue
        ref = base[driver]
        in_laps = grp[grp['PitInTime_s'].notna()]['LapTime_s'].dropna()
        out_laps = grp[grp['PitOutTime_s'].notna()]['LapTime_s'].dropna()
        n = min(len(in_laps), len(out_laps))
        if n == 0:
            continue
        loss = (in_laps.iloc[:n].to_numpy() + out_laps.iloc[:n].to_numpy()) - 2 * ref
        losses.extend(loss.tolist())

    if not losses:
        return np.nan
    losses = np.array(losses)
    losses = losses[(losses > 5) & (losses < 60)]      # drop SC and damage laps
    return float(np.median(losses)) if len(losses) else np.nan


# --- compound scaling -------------------------------------------------------


def compound_offsets(clean, fuel_effect):
    """
    Seconds relative to REFERENCE_COMPOUND, controlling for fuel and tyre age.
    Returns dict compound -> offset, plus the fitted deg per compound.

    Why tyre age gets one slope per compound
    ----------------------------------------
    A single `deg` regressor forces SOFT, MEDIUM and HARD to lose time with
    tyre age at exactly the same rate. They do not, and the rest of the
    project now measures that they do not - Tyre_model fits a separate curve
    per compound. Leaving the assumption here made two models in the same
    simulation contradict each other.

    It also biased the thing this function exists to produce. The shared slope
    comes out near the field average, which over-charges HARD for
    degradation, and HARD runs by far the longest stints (mean tyre age 17.5
    laps against 13.7 for MEDIUM and 12.2 for SOFT). The offset is what
    absorbs the difference, so the compound with the longest stints was handed
    a spurious pace advantage: across 80 races off_HARD moved +0.052 s/lap
    when the slope was freed, and at Zandvoort - where the split between when
    mediums and hards are run is widest - it was enough to make the hard tyre
    look faster than the medium.

    A compound needs MIN_LAPS_FOR_OWN_DEG laps of its own before it gets its
    own slope; below that it shares the reference compound's, because a slope
    fitted to a handful of laps is noise that the offset would then absorb in
    the other direction.
    """
    present = [c for c in DRY_COMPOUNDS if (clean['Compound'] == c).any()]
    others = [c for c in present if c != REFERENCE_COMPOUND]
    if len(clean) < MIN_LAPS_PER_RACE or not others:
        return None

    compound = clean['Compound'].to_numpy()
    tyre_life = clean['TyreLife'].to_numpy(dtype=float)

    own_deg = [c for c in present
               if (compound == c).sum() >= MIN_LAPS_FOR_OWN_DEG]
    if REFERENCE_COMPOUND not in own_deg:
        own_deg = [REFERENCE_COMPOUND] + own_deg
    pooled = [c for c in present if c not in own_deg]

    cols = [clean['LapNumber'].to_numpy(dtype=float)]
    names = ['b_lap']

    for c in own_deg:
        mask = (compound == c).astype(float)
        if c == REFERENCE_COMPOUND and pooled:
            # the thin compounds ride on the reference slope
            for p in pooled:
                mask = mask + (compound == p).astype(float)
        cols.append(tyre_life * mask)
        names.append(f'deg_{c}')

    for c in others:
        cols.append((compound == c).astype(float))
        names.append(f'off_{c}')

    frame = pd.DataFrame(np.column_stack(cols), columns=names)
    frame['y'] = clean['LapTime_s'].to_numpy(dtype=float)
    frame['Driver'] = clean['Driver'].to_numpy()

    demeaned = frame.groupby('Driver')[names + ['y']].transform(lambda s: s - s.mean())
    Xd, yd = demeaned[names].to_numpy(), demeaned['y'].to_numpy()

    coef, _, rank, _ = np.linalg.lstsq(Xd, yd, rcond=None)
    if rank < Xd.shape[1]:
        return None

    fitted = dict(zip(names, (float(v) for v in coef)))

    out = {REFERENCE_COMPOUND: 0.0}
    for c in others:
        out[c] = fitted[f'off_{c}']

    deg_by_compound = {}
    for c in present:
        key = f'deg_{c}' if c in own_deg else f'deg_{REFERENCE_COMPOUND}'
        deg_by_compound[c] = fitted[key]

    return {
        'offsets': out,
        # the strategy enumerator still prices with one number; the reference
        # compound's slope is the honest single answer now that they differ
        'deg': deg_by_compound[REFERENCE_COMPOUND],
        'deg_by_compound': deg_by_compound,
        'pooled_deg': pooled,
        'b_lap': fitted['b_lap'],
        'fuel_effect': fuel_effect,
    }


# --- strategy space ---------------------------------------------------------


def allocate_stints(seq, race_laps, windows):
    """
    Split the race into stints proportional to each compound's typical length.
    Even splitting was wrong: a MEDIUM-HARD one-stopper is short-then-long, not
    two equal halves, so even splits rejected the strategies teams actually run.
    """
    typical = []
    for c in seq:
        lo, hi = windows.get(c, (race_laps / len(seq), race_laps / len(seq)))
        typical.append((lo + hi) / 2.0)

    total = sum(typical)
    if total <= 0:
        return [race_laps / len(seq)] * len(seq)
    return [race_laps * t / total for t in typical]


def strategy_cost(seq, lengths, offsets, deg, pit_loss):
    """
    Rough race time relative to a zero-offset fresh tyre, in seconds.

    Each stint costs its compound offset every lap, plus degradation that
    accumulates with tyre age: sum over i of deg*i = deg*L*(L-1)/2.
    Pit stops cost pit_loss each.
    """
    total = 0.0
    for c, L in zip(seq, lengths):
        off = offsets.get(c, 0.0)
        if np.isnan(off):
            off = 0.0
        total += L * off + deg * L * (L - 1) / 2.0
    return total + pit_loss * (len(seq) - 1)


def stint_curve_cost(curve, length):
    """
    Seconds a stint loses to wear, read off the fitted D(a) curve for this one
    compound - the same formula stint_tyre_cost in simulate.py evaluates lap
    by lap. Ages past the curve's measured range are clamped, not
    extrapolated, same as everywhere else this curve is used.
    """
    n = max(int(round(length)), 1)
    ages = (np.arange(1, n + 1, dtype=float) - AGE_OFFSET)
    ages = np.clip(ages, 0.0, curve.max_age)
    d = curve.b1 * ages + curve.b2 * ages * ages
    if curve.tau is not None and curve.gamma > 0:
        over = np.maximum(ages - curve.tau, 0.0)
        d = d + curve.gamma * over * over
    return float(d.sum())


def strategy_cost_curve(seq, lengths, offsets, curves, pit_loss):
    """
    strategy_cost's counterpart, priced with today's per-compound tyre curve
    instead of the one shared deg slope.

    Returns None if any compound in the sequence has no fitted curve for this
    track, rather than pricing some stints from the curve and others from
    nothing - a partial number here would look precise and not be.
    """
    total = 0.0
    for c, L in zip(seq, lengths):
        curve = curves.get(c)
        if curve is None:
            return None
        off = offsets.get(c, 0.0)
        if np.isnan(off):
            off = 0.0
        total += L * off + stint_curve_cost(curve, L)
    return total + pit_loss * (len(seq) - 1)


def curves_for_race(store, race, compounds):
    """This track's fitted curve for each compound that has one, or {}."""
    if store is None:
        return {}
    out = {}
    for c in compounds:
        try:
            out[c] = store.get(race, c)
        except UnknownProfile:
            continue
    return out


def build_strategies(compounds, upper_bound, race_laps, windows, offsets, deg,
                     pit_loss, curves=None):
    """Every compound sequence up to upper_bound stops, FIA two-compound rule."""
    rows = []
    order = [c for c in DRY_COMPOUNDS if c in compounds]
    if len(order) < 2:
        return rows

    for stops in range(1, upper_bound + 1):
        for seq in product(order, repeat=stops + 1):
            if len(set(seq)) < 2:                 # dry race must use two compounds
                continue

            lengths = allocate_stints(seq, race_laps, windows)
            feasible = True
            for c, L in zip(seq, lengths):
                if c not in windows:
                    continue
                lo, hi = windows[c]
                if not (lo * (1 - WINDOW_TOLERANCE) <= L <= hi * (1 + WINDOW_TOLERANCE)):
                    feasible = False
                    break

            curve_cost = (strategy_cost_curve(seq, lengths, offsets, curves, pit_loss)
                         if curves else None)

            rows.append({
                'n_stops': stops,
                'strategy': '-'.join(s[0] for s in seq),
                'sequence': '-'.join(seq),
                'stint_laps': '-'.join(f'{L:.0f}' for L in lengths),
                'est_cost': round(strategy_cost(seq, lengths, offsets, deg, pit_loss), 1),
                'est_cost_curve': (round(curve_cost, 1) if curve_cost is not None
                                   else np.nan),
                'within_windows': feasible,
            })

    # rank within each track, cheapest first - on est_cost, as before, and
    # separately on est_cost_curve so a reader can rank either way
    if rows:
        best = min(r['est_cost'] for r in rows)
        for r in rows:
            r['cost_vs_best'] = round(r['est_cost'] - best, 1)

        curve_vals = [r['est_cost_curve'] for r in rows
                     if not np.isnan(r['est_cost_curve'])]
        best_curve = min(curve_vals) if curve_vals else None
        for r in rows:
            r['cost_vs_best_curve'] = (round(r['est_cost_curve'] - best_curve, 1)
                                       if best_curve is not None
                                       and not np.isnan(r['est_cost_curve'])
                                       else np.nan)
    return rows


# --- main -------------------------------------------------------------------


def collect():
    race_rows, stint_rows, scale_rows = [], [], []
    laps = load_laps(SEASONS)

    for season, name, race_laps, race_df in iter_races(laps):
        clean, wet_share = clean_race(race_df)
        if wet_share > MAX_WET_SHARE or clean.empty:
            print(f'  {season} {name:32s} wet or empty, skipped')
            continue

        stints, stops = stint_table(race_df, race_laps)
        if stints.empty or stops.empty:
            print(f'  {season} {name:32s} no usable stints')
            continue

        fuel_effect = get_track_params(name)['fuel_effect']
        scaling = compound_offsets(clean, fuel_effect)
        pit_loss = pit_loss_estimate(race_df, clean)

        race_rows.append({
            'Season': season,
            'Race': name,
            'race_laps': race_laps,
            'median_stops': float(stops['n_stops'].median()),
            'modal_stops': int(stops['n_stops'].mode().iloc[0]),
            'max_stops': int(stops['n_stops'].max()),
            'n_finishers': len(stops),
            'pit_loss': pit_loss,
        })

        st = stints.copy()
        st['Season'], st['Race'] = season, name
        stint_rows.append(st)

        if scaling:
            row = {'Season': season, 'Race': name, 'deg': scaling['deg']}
            for c in DRY_COMPOUNDS:
                row[f'off_{c}'] = scaling['offsets'].get(c, np.nan)
                row[f'deg_{c}'] = scaling['deg_by_compound'].get(c, np.nan)
            row['pooled_deg'] = '-'.join(scaling['pooled_deg']) or ''
            scale_rows.append(row)

        offs = scaling['offsets'] if scaling else {}
        offs_txt = ' '.join(f'{c[0]}{offs[c]:+.2f}' for c in DRY_COMPOUNDS if c in offs)
        print(f"  {season} {name:32s} stops {stops['n_stops'].median():.0f} "
              f"(max {stops['n_stops'].max()})  pit_loss {pit_loss:5.1f}s  {offs_txt}")

    return (pd.DataFrame(race_rows),
            pd.concat(stint_rows, ignore_index=True) if stint_rows else pd.DataFrame(),
            pd.DataFrame(scale_rows))


def summarise(races, stints, scaling, tyre_store=None):
    summary_rows, strategy_rows = [], []

    for race, grp in races.groupby('Race'):
        st = stints[stints['Race'] == race]
        race_laps = int(round(grp['race_laps'].median()))
        max_stops = int(grp['max_stops'].max())
        modal_stops = int(grp['modal_stops'].median())
        # modal, not max: one damaged car pitting five times is not a strategy
        upper_bound = min(modal_stops + 1, HARD_STOP_CAP)

        share = st['Compound'].value_counts(normalize=True)
        compounds = [c for c in DRY_COMPOUNDS if share.get(c, 0) >= MIN_COMPOUND_SHARE]
        if not compounds:
            compounds = list(share.index[:2])

        windows = {}
        for c in compounds:
            lens = st[st['Compound'] == c]['stint_laps']
            if len(lens) >= 3:
                windows[c] = (float(lens.quantile(0.25)), float(lens.quantile(0.75)))

        sc = scaling[scaling['Race'] == race]
        offs = {c: float(sc[f'off_{c}'].median()) if not sc.empty else np.nan
                for c in DRY_COMPOUNDS}

        summary_rows.append({
            'Race': race,
            'race_laps': race_laps,
            'modal_stops': modal_stops,
            'max_stops': max_stops,
            'upper_bound': upper_bound,
            'pit_loss': float(grp['pit_loss'].median()),
            'compounds': '-'.join(compounds),
            'off_SOFT': offs.get('SOFT'),
            'off_MEDIUM': offs.get('MEDIUM'),
            'off_HARD': offs.get('HARD'),
            'window_SOFT': str(tuple(round(v) for v in windows['SOFT'])) if 'SOFT' in windows else '',
            'window_MEDIUM': str(tuple(round(v) for v in windows['MEDIUM'])) if 'MEDIUM' in windows else '',
            'window_HARD': str(tuple(round(v) for v in windows['HARD'])) if 'HARD' in windows else '',
            'n_seasons': len(grp),
        })

        race_deg = float(sc['deg'].median()) if not sc.empty else 0.0
        race_pit_loss = float(grp['pit_loss'].median())
        curves = curves_for_race(tyre_store, race, compounds)
        for row in build_strategies(compounds, upper_bound, race_laps, windows,
                                    offs, race_deg, race_pit_loss, curves):
            row['Race'] = race
            strategy_rows.append(row)

    return (pd.DataFrame(summary_rows).sort_values('Race').reset_index(drop=True),
            pd.DataFrame(strategy_rows))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f'Seasons: {SEASONS}')
    races, stints, scaling = collect()
    if races.empty:
        print('\nNothing collected.')
        return

    try:
        tyre_store = load_tyre_store()
        print(f'\nTyre_model curves loaded: {len(tyre_store)} track x compound '
              f'(v{tyre_store.model_version})')
    except FileNotFoundError:
        tyre_store = None
        print('\nNo tyre_curve_params.json found - est_cost_curve will be blank. '
              'Run fit_tyre_curve.py first to fill it in.')

    summary, strategies = summarise(races, stints, scaling, tyre_store)

    print('\n===== compound scaling, all races pooled (s vs MEDIUM) =====')
    print(scaling[[f'off_{c}' for c in DRY_COMPOUNDS]].median().round(3).to_string())

    print('\n===== per track =====')
    cols = ['Race', 'race_laps', 'modal_stops', 'max_stops', 'upper_bound',
            'pit_loss', 'compounds', 'off_SOFT', 'off_HARD',
            'window_SOFT', 'window_MEDIUM', 'window_HARD', 'n_seasons']
    print(summary[cols].round(2).to_string(index=False))

    print(f'\n===== strategy space: {len(strategies)} rows =====')
    feasible = strategies[strategies['within_windows']]
    print(f'Within pit windows: {len(feasible)}')
    print('\nBest three per track (est_cost, this file\'s single-slope deg, '
          'lower is faster):')
    for race in summary['Race']:
        sub = feasible[feasible['Race'] == race].nsmallest(3, 'est_cost')
        if sub.empty:
            print(f'  {race:32s} none feasible')
            continue
        txt = '  '.join(f"{r['strategy']} [{r['stint_laps']}] {r['est_cost']:+.0f}s"
                        for _, r in sub.iterrows())
        print(f'  {race:32s} {txt}')

    has_curve = feasible['est_cost_curve'].notna()
    print(f'\nBest three per track (est_cost_curve, today\'s per-compound '
          f'Tyre_model curve, {int(has_curve.sum())} of {len(feasible)} rows '
          f'priced):')
    for race in summary['Race']:
        sub = (feasible[(feasible['Race'] == race) & has_curve]
              .nsmallest(3, 'est_cost_curve'))
        if sub.empty:
            print(f'  {race:32s} no curve for the compounds here')
            continue
        txt = '  '.join(
            f"{r['strategy']} [{r['stint_laps']}] {r['est_cost_curve']:+.0f}s"
            for _, r in sub.iterrows())
        print(f'  {race:32s} {txt}')

    races.to_csv(os.path.join(DATA_DIR, 'pit_by_race.csv'), index=False)
    summary.to_csv(os.path.join(DATA_DIR, 'pit_summary.csv'), index=False)
    scaling.to_csv(os.path.join(DATA_DIR, 'compound_scaling.csv'), index=False)
    strategies.to_csv(os.path.join(DATA_DIR, 'pit_strategies.csv'), index=False)
    print(f'\nSaved 4 files to {DATA_DIR}')


if __name__ == '__main__':
    main()