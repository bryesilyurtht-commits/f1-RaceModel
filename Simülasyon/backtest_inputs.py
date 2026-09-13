"""
F1 Prediction Simulation - v2.6 - backtest_inputs.py
Building the inputs a past race could have been predicted from.

The model reads a directory. This writes one, per target race, containing only
what was knowable at the cutoff - qualifying over, race not started - and then
points the model at it.

Why a directory rather than a date argument
-------------------------------------------
Every table the model reads would otherwise need its own date filter, and a
filter that is missing from one of them fails silently: the prediction still
comes out, slightly too good, with nothing to show for it. Building a
directory inverts that. A file that is not written cannot be read, so a table
nobody filtered is a table the model does not get, and the failure is a
fallback the run summary names rather than a number that quietly improves.

What is in the directory
------------------------
- The target race's qualifying: pole time, pole driver, grid order. After the
  cutoff by definition, and the model needs it to run at all.
- Driver pace, rebuilt from races strictly before the target.

What is deliberately absent
---------------------------
The per-circuit tables - overtaking rates, safety car rates, pit loss, the
strategy menu, tyre curves - as committed are fitted on the whole 2018-2025
archive. For a 2024 target that archive contains 2025, so handing them over
would be the pooling leak the brief asks to rule out. They are withheld, the
model falls back to the constants it documents for exactly this case, and the
report says so. That measures the model without its per-circuit measurements,
which is a weaker model than the one that runs on a current race - an
understatement of its strength, not an overstatement, and the direction an
unproven claim should err in.

Race identity across seasons
----------------------------
A pole time is keyed by race name, and "Italian Grand Prix" happens every year.
Mapping poles onto a multi-season history by name alone would give every Monza
the most recent Monza's pole, so the history built here carries a season-and-
round label instead, and the ordering that recency weighting depends on is a
global sequence rather than a round number that restarts each January.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from Simülasyon import clean as C

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LAPS = os.path.join(DATA_DIR, 'laps_2018_2025.csv')
QUALI = os.path.join(DATA_DIR, 'quali_history.csv')

# The tables withheld because the committed versions are fitted on the whole
# archive. Named here so the report can list them rather than describing a
# directory's absences in prose.
WITHHELD = ('overtaking.csv', 'neutralization.csv', 'pit_summary.csv',
            'pit_strategies.csv', 'tyre_curve_params.json',
            'compound_scaling.csv', 'grid_stats.csv', 'track_affinity.csv',
            'track_evolution_agg.csv', 'team_affinity', 'dnf_driver_risk.csv',
            'dnf_team_risk.csv', 'dnf_profile.csv')


def load_laps(path=None):
    """The lap archive, with a global chronological label per race."""
    laps = pd.read_csv(path or LAPS)
    laps['race_id'] = (laps['Season'].astype(str) + '-'
                       + laps['RoundNumber'].astype(int).astype(str).str.zfill(2)
                       + ' ' + laps['Race'].astype(str))
    return laps


def load_quali(path=None):
    quali = pd.read_csv(path or QUALI)
    quali['race_id'] = (quali['Season'].astype(str) + '-'
                        + quali['Round'].astype(int).astype(str).str.zfill(2)
                        + ' ' + quali['Race'].astype(str))
    return quali


def laps_before(laps, season, rnd):
    """Laps from races strictly before the target. The walk-forward cut."""
    return laps[(laps['Season'] < season)
                | ((laps['Season'] == season)
                   & (laps['RoundNumber'] < rnd))].copy()


def clean_flag(laps):
    """
    The project's clean-lap rules, over the archive's column names.

    clean.add_clean_flag() expects the raw FastF1 columns - a LapTime timedelta
    and PitInTime/PitOutTime. The archive has already parsed those into seconds
    under different names. The rules are the same rules; only where they read
    from differs, and they are applied per race rather than per race name so
    that two seasons at the same circuit get their own outlier cutoff.
    """
    df = laps.copy()
    df['Compound'] = df['Compound'].astype(str).str.upper()
    df['TrackStatus'] = df['TrackStatus'].astype(str)
    df['is_wet'] = df['Compound'].isin(C.WET_COMPOUNDS)

    valid = df['LapTime_s'].notna()
    green = df['TrackStatus'] == '1.0'
    no_pit = df['PitInTime_s'].isna() & df['PitOutTime_s'].isna()
    not_first = df['LapNumber'] > 1
    not_outlap = df['TyreLife'] > 1
    df['clean_lap'] = (valid & green & no_pit & not_first & not_outlap
                       & ~df['is_wet'])

    for _, group in df[df['clean_lap']].groupby('race_id'):
        cutoff = group['LapTime_s'].median() * C.OUTLIER_MARGIN
        df.loc[group.index[group['LapTime_s'] > cutoff], 'clean_lap'] = False
    return df


def driver_pace(laps, quali, season, rnd, min_races=3):
    """
    Delta to pole and lap-to-lap sigma, from races before the target only.

    Reuses clean.build_driver_pace so the backtest measures the model's own
    definition of pace rather than a second implementation of it that happens
    to agree on the easy cases.
    """
    past = laps_before(laps, season, rnd)
    if past.empty:
        return None, 'no laps before this race'

    flagged = clean_flag(past)
    clean = flagged[flagged['clean_lap']].copy()
    if clean.empty:
        return None, 'no clean laps before this race'

    poles = (quali.groupby('race_id')['pole_time'].min().reset_index()
             .rename(columns={'race_id': 'Race'}))

    # build_driver_pace orders races by Round and keys poles by Race, both of
    # which repeat across seasons. Handing it the global label as the race name
    # and a global sequence as the round keeps both meanings intact.
    order = sorted(clean['race_id'].unique())
    sequence = {name: i + 1 for i, name in enumerate(order)}
    clean = clean.rename(columns={'Race': 'race_name'})
    clean['Race'] = clean['race_id']
    clean['Round'] = clean['race_id'].map(sequence)

    pace, lam, race_order, weights = C.build_driver_pace(clean, poles)
    pace = pace[pace['n_races'] >= min_races]
    if len(pace) < 10:
        return None, f'only {len(pace)} drivers with {min_races}+ races of pace'
    return pace, None


def entry_list(results, season, rnd):
    """Who started and where, from the target race's own grid."""
    race = results[(results['Season'] == season)
                   & (results['RoundNumber'] == rnd)].copy()
    race = race[race['grid_pos'].notna()]
    return race.sort_values('grid_pos')


def build(target, laps, quali, results, into=None):
    """
    Write one race's pre-race directory, or say why it cannot be written.

    Returns (path, event_name, note). The caller owns the directory and is
    expected to remove it; nothing here writes into the project's own data
    directory, so a backtest cannot damage the inputs of a real prediction.
    """
    season, rnd, event = target
    pace, why = driver_pace(laps, quali, season, rnd)
    if pace is None:
        return None, event, why

    race_quali = quali[(quali['Season'] == season) & (quali['Round'] == rnd)]
    if race_quali.empty:
        return None, event, 'no qualifying on file for the target race'
    timed = race_quali[race_quali['quali_time'].notna()]
    if timed.empty:
        return None, event, 'no timed qualifying laps for the target race'

    entries = entry_list(results, season, rnd)
    if entries.empty:
        return None, event, 'no grid on file for the target race'

    path = into or tempfile.mkdtemp(prefix=f'bt_{season}_{rnd}_')
    os.makedirs(path, exist_ok=True)

    pole_row = timed.sort_values('quali_time').iloc[0]
    pd.DataFrame([{'Race': event, 'Round': rnd,
                   'pole_time': float(pole_row['quali_time']),
                   'pole_driver': str(pole_row['Driver']),
                   'is_target': True}]).to_csv(
        os.path.join(path, f'f1_{season}_poles.csv'), index=False)

    pace.to_csv(os.path.join(path, f'driver_pace_{season}.csv'), index=False)

    entries[['Driver', 'Team', 'grid_pos']].rename(
        columns={'grid_pos': 'GridPosition'}).to_csv(
        os.path.join(path, f'f1_{season}_grid.csv'), index=False)

    race_quali[['Driver', 'Team', 'quali_pos']].assign(
        Race=event, Round=rnd, is_target=True).to_csv(
        os.path.join(path, f'f1_{season}_quali.csv'), index=False)

    return path, event, None


def audit(path, season):
    """
    What the directory actually contains, for the leakage section of the report.

    The claim the backtest rests on is that nothing in here was derived from
    the target race or from anything after it. This lists the files so the
    claim is checkable against the directory rather than against a sentence.
    """
    present = sorted(os.listdir(path))
    expected = {f'f1_{season}_poles.csv', f'driver_pace_{season}.csv',
                f'f1_{season}_grid.csv', f'f1_{season}_quali.csv'}
    return {'files': present,
            'unexpected': sorted(set(present) - expected),
            'withheld': list(WITHHELD)}
