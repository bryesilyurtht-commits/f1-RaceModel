"""
F1 Prediction Simulation - tyre_profile.py
v1.1-A / A3: per-stint degradation profiles, per race and compound.

Reads  data/stint_laps.csv          (written by stint_table.py)
       data/track_evolution_agg.csv (optional, written by track_evolution.py)
Writes data/tyre_age_profile.csv        binned profile per season, race, compound
       data/tyre_age_profile_track.csv  pooled across seasons
       data/stint_slopes.csv            one row per stint, for inspection

Method
------
One stint at a time, which is how a tyre is actually used.

    corrected = LapTime_s + fuel_effect * (lap - 1) + evo_rate * (lap - 1)

Fuel burn and track evolution both make later laps quicker for everyone, so
they are added back before the tyre is looked at. Fuel comes from
fuel_effect.py, evolution from track_evolution.py where it has been measured
and zero where it has not.

Each stint is then expressed relative to its own opening laps:

    rel(a) = corrected(a) - median(corrected over the first few laps)

That subtraction removes the driver and the car. A slow driver's stint starts
slow and stays slow; only the shape survives, which is the thing being
measured. No driver dummies are needed.

Stints are pooled per race and compound into tyre-age bins, taking the median
across stints so one car stuck in traffic cannot drag a bin.

Why not the previous design
---------------------------
The version before this fitted one panel per race with a dummy for every lap,
and read degradation out of the contrast between cars on old and fresh tyres
at the same moment. That was clean for MEDIUM and HARD, where the field is
always spread across tyre ages. It failed completely for SOFT: softs get used
in one narrow phase of a race, everyone on them is on roughly the same age at
the same time, so there was no contrast to measure and k_SOFT collapsed to
near zero. Zandvoort came out with softs wearing at a third the rate of hards,
which is backwards.

The trade is explicit. Working stint by stint means tyre age and lap number
move together, so anything else that varies with lap number and is not removed
first will be read as tyre wear. That is why fuel and evolution are subtracted
using outside estimates rather than fitted here, and it is why an error in the
evolution rate lands directly on k. Measuring SOFT badly beats not measuring it.

Known and unfixed: cars running in traffic show flatter stints because they
were never pushing. That needs gap-to-car-ahead and is deferred.

Usage:
    python tyre_profile.py
    python tyre_profile.py --min-laps 10
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from Simülasyon.fuel_effect import get_track_params
    HAVE_FUEL = True
except Exception:
    HAVE_FUEL = False

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

MIN_STINT_LAPS = 10        # clean laps a stint needs to carry a shape
MAX_START_AGE = 4          # stints joined already worn have no usable baseline
BASELINE_LAPS = 3          # opening laps that define the stint's zero
MIN_SPAN = 6               # tyre-age range the stint must cover

MIN_LAPS_PER_BIN = 4       # bins thinner than this are dropped
MIN_STINTS_PER_CELL = 3    # a race x compound cell needs this many stints

OUTLIER_CLIP = 3.0         # seconds above the stint baseline; beyond is traffic

FUEL_EFFECT_FALLBACK = 0.045

BIN_WIDTH = 3
MAX_AGE = 60


def age_bin(age):
    return np.floor((np.minimum(age, MAX_AGE) - 1) / BIN_WIDTH).astype(int)


def bin_centre(idx):
    return idx * BIN_WIDTH + 1 + (BIN_WIDTH - 1) / 2.0


def fuel_rate(race):
    if HAVE_FUEL:
        try:
            return float(get_track_params(race, quiet=True)['fuel_effect'])
        except TypeError:
            try:
                return float(get_track_params(race)['fuel_effect'])
            except Exception:
                pass
        except Exception:
            pass
    return FUEL_EFFECT_FALLBACK


def load_evolution():
    """
    Track evolution in s/lap, positive meaning the track speeds up.

    Optional on purpose: without it the model still runs, it just hands the
    evolution to the tyre. The printout says how many races have a measured
    rate so that is visible rather than assumed.
    """
    path = os.path.join(DATA_DIR, 'track_evolution_agg.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if 'evo_rate' not in df.columns or 'Race' not in df.columns:
        return {}
    return {str(r['Race']): float(r['evo_rate'])
            for _, r in df.iterrows() if pd.notna(r['evo_rate'])}


def evo_for(race, evo_map):
    if race in evo_map:
        return evo_map[race]
    key = str(race).lower()
    for name, rate in evo_map.items():
        n = str(name).lower()
        if n in key or key in n:
            return rate
    return 0.0


# --- per-stint --------------------------------------------------------------


def stint_profiles(laps, evo_map):
    """
    One row per lap of every usable stint, with the tyre-only residual.

    Returns (long_df, slopes_df).
    """
    rows, slopes = [], []

    for (season, race), race_df in laps.groupby(['Season', 'Race'], sort=True):
        burn = fuel_rate(race)
        evo = evo_for(race, evo_map)

        for (driver, stint), g in race_df.groupby(['Driver', 'Stint']):
            g = g.sort_values('TyreLife')
            comp = g['Compound'].mode().iloc[0]
            if comp not in DRY_COMPOUNDS or len(g) < MIN_STINT_LAPS:
                continue

            age = g['TyreLife'].to_numpy(float)
            if age.min() > MAX_START_AGE:
                continue
            if age.max() - age.min() < MIN_SPAN:
                continue

            lap = g['LapNumber'].to_numpy(float)
            y = g['LapTime_s'].to_numpy(float)

            # put back everything that made later laps quicker for the whole
            # field, so what is left belongs to the tyre
            corrected = y + burn * (lap - 1) + evo * (lap - 1)

            base_mask = age <= age.min() + BASELINE_LAPS - 1
            if base_mask.sum() < 1:
                continue
            baseline = float(np.median(corrected[base_mask]))
            rel = corrected - baseline

            # a lap several seconds off the stint's own baseline is traffic or
            # a mistake, not wear; the cliff is worth about a second
            keep = rel <= OUTLIER_CLIP
            if keep.sum() < MIN_STINT_LAPS:
                continue

            sid = f'{season}|{race}|{driver}|{int(stint)}'
            for a, r in zip(age[keep], rel[keep]):
                rows.append({'Season': season, 'Race': race, 'Compound': comp,
                             'Driver': driver, 'stint_id': sid,
                             'age': a, 'rel': r})

            early = keep & (age <= age.min() + 11)
            slope = np.nan
            if early.sum() >= 4:
                x = age[early] - age[early].mean()
                slope = float((x * (rel[early] - rel[early].mean())).sum()
                              / max((x * x).sum(), 1e-9))
            slopes.append({'Season': season, 'Race': race, 'Driver': driver,
                           'Compound': comp, 'stint_id': sid,
                           'n_laps': int(keep.sum()),
                           'start_age': float(age.min()),
                           'max_age': float(age[keep].max()),
                           'early_slope': slope})

    return pd.DataFrame(rows), pd.DataFrame(slopes)


# --- binning ----------------------------------------------------------------


def bin_profiles(long_df):
    """Median across stints, per race, compound and tyre-age bin."""
    df = long_df.copy()
    df['bin'] = age_bin(df['age'].to_numpy(float))

    g = (df.groupby(['Season', 'Race', 'Compound', 'bin'])
           .agg(loss_s=('rel', 'median'),
                spread=('rel', 'std'),
                n_laps=('rel', 'size'),
                n_stints=('stint_id', 'nunique'))
           .reset_index())

    g = g[g['n_laps'] >= MIN_LAPS_PER_BIN]
    g['age'] = bin_centre(g['bin'])
    g['se'] = g['spread'] / np.sqrt(g['n_laps'].clip(lower=1))

    out = []
    for (season, race, comp), sub in g.groupby(['Season', 'Race', 'Compound']):
        sub = sub.sort_values('bin').copy()
        if sub['n_stints'].max() < MIN_STINTS_PER_CELL or len(sub) < 3:
            continue
        # rebase on the youngest surviving bin so the profile reads as
        # "seconds lost versus a fresh tyre"
        sub['loss_s'] = sub['loss_s'] - sub['loss_s'].iloc[0]
        sub['is_reference'] = False
        sub.iloc[0, sub.columns.get_loc('is_reference')] = True
        out.append(sub)

    if not out:
        raise SystemExit('No race x compound cell survived binning.')
    return pd.concat(out, ignore_index=True)


def main():
    global MIN_STINT_LAPS
    if '--min-laps' in sys.argv:
        MIN_STINT_LAPS = int(sys.argv[sys.argv.index('--min-laps') + 1])

    laps = pd.read_csv(os.path.join(DATA_DIR, 'stint_laps.csv'))
    laps['Compound'] = laps['Compound'].astype(str).str.upper()
    laps = laps[laps['Compound'].isin(DRY_COMPOUNDS)]

    evo_map = load_evolution()

    print(f'Laps loaded    : {len(laps)}')
    print(f'Fuel model     : {"fuel_effect.py" if HAVE_FUEL else "FALLBACK constant"}')
    print(f'Evolution      : {len(evo_map)} races with a measured rate'
          f'{" (none - evolution will be read as wear)" if not evo_map else ""}')
    print(f'Min stint laps : {MIN_STINT_LAPS}')
    print('Design         : per stint, relative to its own opening laps\n')

    long_df, slopes = stint_profiles(laps, evo_map)
    if long_df.empty:
        raise SystemExit('No stint passed the filters. Lower MIN_STINT_LAPS.')

    print(f'Stints usable  : {slopes["stint_id"].nunique()}')
    print(f'Laps kept      : {len(long_df)}\n')

    print('--- stints per compound ---')
    print(slopes.groupby('Compound')
                .agg(stints=('stint_id', 'nunique'),
                     median_laps=('n_laps', 'median'),
                     median_max_age=('max_age', 'median'))
                .to_string())

    print('\n--- early slope per stint, s/lap (sanity, not the model) ---')
    print(slopes.groupby('Compound')
                .agg(n=('early_slope', 'count'),
                     median=('early_slope', 'median'),
                     negative=('early_slope', lambda s: (s < 0).mean()))
                .round(4).to_string())
    print('Expected order is SOFT > MEDIUM > HARD. This is the raw per-stint')
    print('slope before any pooling, so it is the earliest place a sign error')
    print('would show up.')

    prof = bin_profiles(long_df)

    print('\n--- profiles built ---')
    cells = prof.groupby(['Season', 'Race', 'Compound']).ngroups
    print(f'race x compound x season cells : {cells}')
    print(prof.groupby('Compound')
              .agg(cells=('bin', lambda s: len(s)),
                   median_stints=('n_stints', 'median'),
                   max_age=('age', 'max'))
              .to_string())

    checks = []
    for (season, race, comp), g in prof.groupby(['Season', 'Race', 'Compound']):
        g = g.sort_values('age')
        checks.append({'Compound': comp,
                       'slope': np.polyfit(g['age'], g['loss_s'], 1)[0],
                       'max_age': g['age'].max(),
                       'rise': g['loss_s'].iloc[-1]})
    chk = pd.DataFrame(checks)
    print('\n--- pooled profile slope by compound ---')
    print(chk.groupby('Compound')
             .agg(profiles=('slope', 'size'),
                  median_slope=('slope', 'median'),
                  negative=('slope', lambda s: (s < 0).mean()),
                  median_rise_s=('rise', 'median'),
                  median_max_age=('max_age', 'median'))
             .round(4).to_string())

    pooled = (prof.groupby(['Race', 'Compound', 'bin'])
                  .agg(age=('age', 'first'),
                       loss_s=('loss_s', 'median'),
                       n_laps=('n_laps', 'sum'),
                       n_stints=('n_stints', 'sum'),
                       n_seasons=('Season', 'nunique'))
                  .reset_index()
                  .sort_values(['Race', 'Compound', 'age']))

    print()
    for fname, frame in [('tyre_age_profile.csv', prof),
                         ('tyre_age_profile_track.csv', pooled),
                         ('stint_slopes.csv', slopes)]:
        p = os.path.join(DATA_DIR, fname)
        frame.to_csv(p, index=False)
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()