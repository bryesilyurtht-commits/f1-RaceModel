"""
F1 Prediction Simulation - dataset.py
Loads race laps once, caches them as a single CSV, serves every other script.

Before this, track_evolution.py and pit_analysis.py each re-parsed ~90 FastF1
sessions on every run. Now the parse happens once and lands in
data/laps_<first>_<last>.csv; later runs read that file in seconds.

Usage:
    from dataset import load_laps, iter_races, clean_race
    laps = load_laps()                       # all cached seasons
    laps = load_laps(seasons=[2024, 2025])   # subset

    python dataset.py                        # build the cache if missing
    python dataset.py --rebuild              # ignore the CSV and re-parse
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]

# The tyre model alone reaches further back. Everything else in the project -
# driver pace, overtaking, grid, affinity, pit strategy - stays on SEASONS,
# because those describe the current cars and a 2019 result says nothing about
# who is quick in 2026.
#
# 2018 is the earliest season that exists: FastF1's timing API does not cover
# 2017 at all, and a 2017 session loads with no laps rather than with an error.
TYRE_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# 2022 moved to 18-inch wheels and a new construction, which is a real break in
# how tyres wear. Seasons either side of it are tagged so the fit can tell them
# apart rather than pooling them silently.
TYRE_ERA_BREAK = 2022

KEEP_COLUMNS = [
    'Driver', 'Team', 'LapNumber', 'LapTime', 'Compound', 'TyreLife',
    'Stint', 'TrackStatus', 'Position', 'PitInTime', 'PitOutTime',
]

DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

# Pirelli's dry range before the C1-C5 renaming, softest first. Until 2019 the
# three compounds brought to a race were named absolutely - a weekend might get
# hypersoft, ultrasoft and soft - so "SOFT" in 2018 is not the same rubber as
# "SOFT" in 2023, and pooling them on the label would be nonsense.
#
# The fix is the one Pirelli themselves adopted in 2019: within each race, the
# softest of the three compounds present becomes SOFT, the middle MEDIUM and
# the hardest HARD. That is what the modern names have always meant - they are
# relative to the weekend's selection, not absolute.
LEGACY_HARDNESS = [
    'HYPERSOFT', 'ULTRASOFT', 'SUPERSOFT', 'SOFT', 'MEDIUM', 'HARD', 'SUPERHARD',
]
LEGACY_LAST_SEASON = 2018       # last season using the absolute names
OUTLIER_MARGIN = 1.07
MIN_LAPS_PER_DRIVER = 10


RESULT_COLUMNS = [
    'Abbreviation', 'TeamName', 'GridPosition', 'Position',
    'ClassifiedPosition', 'Status', 'Points',
]


def cache_path(seasons=None):
    s = sorted(seasons or SEASONS)
    return os.path.join(DATA_DIR, f'laps_{s[0]}_{s[-1]}.csv')


def results_path(seasons=None):
    s = sorted(seasons or SEASONS)
    return os.path.join(DATA_DIR, f'results_{s[0]}_{s[-1]}.csv')


def normalise_compounds(race_laps, season):
    """
    Rewrites pre-2019 compound names as SOFT/MEDIUM/HARD, relative to the
    three that were brought to that race.

    Returns (series, mapping). The mapping is kept so the printout can show
    what was renamed rather than doing it invisibly.

    A weekend where only two dry compounds were actually run is discarded
    rather than mapped. There is no defensible middle in a pair, and the
    tempting shortcut - leave the labels alone - is the worst option available:
    if a race ran ultrasoft and soft, its "SOFT" laps are the HARD option of
    that weekend, and letting them through would file the hardest rubber
    present under the softest name.
    """
    comp = race_laps['Compound'].astype(str).str.upper()
    if season > LEGACY_LAST_SEASON:
        return comp, {}

    present = [c for c in LEGACY_HARDNESS if (comp == c).any()]
    if len(present) != 3:
        return pd.Series('UNKNOWN', index=comp.index), {}

    mapping = dict(zip(present, DRY_COMPOUNDS))
    return comp.map(lambda c: mapping.get(c, 'UNKNOWN')), mapping


# --- building ---------------------------------------------------------------


def build(seasons=None, verbose=True):
    """Parses every race session once and writes the combined CSV."""
    import fastf1

    seasons = seasons or SEASONS
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    frames, result_frames = [], []
    for season in seasons:
        if verbose:
            print(f'\n===== {season} =====')
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
        except Exception as exc:
            print(f'  schedule failed: {exc}')
            continue

        for _, event in schedule.iterrows():
            name = event['EventName']
            try:
                s = fastf1.get_session(season, name, 'R')
                s.load(telemetry=False, weather=False, messages=False)
                laps = s.laps
                if laps is None or laps.empty:
                    if verbose:
                        print(f'  {name:32s} no laps')
                    continue
            except Exception as exc:
                if verbose:
                    print(f'  {name:32s} skipped ({type(exc).__name__})')
                continue

            missing = [c for c in KEEP_COLUMNS if c not in laps.columns]
            if missing:
                if verbose:
                    print(f'  {name:32s} missing {missing}')
                continue

            out = laps[KEEP_COLUMNS].copy()
            # TrackStatus is a string of status codes, but some seasons hand it
            # back as a number. Concatenating a numeric season with a string one
            # upcasts the whole column to float, and '1' becomes '1.0' in the
            # CSV - which then matches nothing, because every clean-lap filter
            # in the project tests against '1'. Pinning it to str here keeps the
            # concat honest.
            out['TrackStatus'] = out['TrackStatus'].astype(str)
            out['Compound'], renamed = normalise_compounds(out, season)
            out['Season'] = season
            out['Race'] = name
            out['RoundNumber'] = event['RoundNumber']
            out['tyre_era'] = ('18inch' if season >= TYRE_ERA_BREAK else '13inch')
            frames.append(out)

            # GridPosition here already includes penalties, unlike the Q times
            res = getattr(s, 'results', None)
            if res is not None and not res.empty:
                have = [c for c in RESULT_COLUMNS if c in res.columns]
                r = res[have].copy()
                r = r.rename(columns={'Abbreviation': 'Driver', 'TeamName': 'Team'})
                r['Season'] = season
                r['Race'] = name
                r['RoundNumber'] = event['RoundNumber']
                result_frames.append(r)
            if verbose:
                note = ''
                if renamed:
                    note = '  renamed ' + ' '.join(
                        f'{k[:5]}->{v[0]}' for k, v in renamed.items())
                print(f'  {name:32s} {len(out)} laps{note}')

    if not frames:
        raise SystemExit('Nothing collected. Check the cache or the network.')

    df = pd.concat(frames, ignore_index=True)

    # timedeltas to seconds so the CSV round-trips cleanly
    df['LapTime_s'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    for col in ['PitInTime', 'PitOutTime']:
        df[col + '_s'] = pd.to_timedelta(df[col]).dt.total_seconds()
    df = df.drop(columns=['LapTime', 'PitInTime', 'PitOutTime'])

    df['race_laps'] = df.groupby(['Season', 'Race'])['LapNumber'].transform('max')

    path = cache_path(seasons)
    df.to_csv(path, index=False)
    print(f'\nRaces: {df.groupby(["Season", "Race"]).ngroups}   Laps: {len(df)}')
    print(f'Saved: {path}')

    if result_frames:
        res = pd.concat(result_frames, ignore_index=True)
        rpath = results_path(seasons)
        res.to_csv(rpath, index=False)
        print(f'Saved: {rpath}  ({len(res)} entries)')

    return df


# --- loading ----------------------------------------------------------------


def load_laps(seasons=None, rebuild=False):
    """
    Returns the combined lap table, building the cache on first use.

    Columns: Driver, Team, LapNumber, LapTime_s, Compound, TyreLife, Stint,
             TrackStatus, Position, PitInTime_s, PitOutTime_s, Season, Race,
             RoundNumber, race_laps
    """
    path = cache_path(seasons)
    if rebuild or not os.path.exists(path):
        if not rebuild:
            print(f'No cache at {path}, building it once...')
        return build(seasons)

    df = pd.read_csv(path)
    # a cache written while the column was numeric holds '1.0' rather than '1'
    df['TrackStatus'] = (df['TrackStatus'].astype(str)
                         .str.replace(r'\.0$', '', regex=True))
    df['Compound'] = df['Compound'].astype(str).str.upper()
    if 'tyre_era' not in df.columns:
        # caches written before the era column existed are all 18-inch
        df['tyre_era'] = np.where(df['Season'] >= TYRE_ERA_BREAK,
                                  '18inch', '13inch')
    if seasons:
        df = df[df['Season'].isin(seasons)]
    return df


def load_results(seasons=None, rebuild=False):
    """
    Race classifications: Driver, Team, GridPosition, Position, Status, Points.

    GridPosition is the grid as it actually formed, penalties included.
    Position is the finishing order, which is the ground truth a backtest
    scores against.
    """
    path = results_path(seasons)
    if rebuild or not os.path.exists(path):
        print(f'No results cache at {path}, rebuilding laps and results...')
        build(seasons)

    df = pd.read_csv(path)
    for col in ['GridPosition', 'Position', 'Points']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if seasons:
        df = df[df['Season'].isin(seasons)]
    return df


# --- shared cleaning --------------------------------------------------------


def clean_race(race_df, outlier_margin=OUTLIER_MARGIN,
               min_laps_per_driver=MIN_LAPS_PER_DRIVER):
    """
    Applies the project's clean_lap rules to one race.
    Returns (clean_df, wet_share).
    """
    known = race_df['Compound'].isin(set(DRY_COMPOUNDS) | {'INTERMEDIATE', 'WET'})
    wet_share = (race_df.loc[known, 'Compound'].isin({'INTERMEDIATE', 'WET'}).mean()
                 if known.any() else 0.0)

    mask = (race_df['LapTime_s'].notna()
            & (race_df['TrackStatus'] == '1')
            & race_df['PitInTime_s'].isna()
            & race_df['PitOutTime_s'].isna()
            & (race_df['LapNumber'] > 1)
            & (race_df['TyreLife'] > 1)
            & race_df['Compound'].isin(DRY_COMPOUNDS))

    clean = race_df[mask].copy()
    if clean.empty:
        return clean, wet_share

    cutoff = clean['LapTime_s'].median() * outlier_margin
    clean = clean[clean['LapTime_s'] <= cutoff]

    counts = clean.groupby('Driver')['LapTime_s'].transform('size')
    clean = clean[counts >= min_laps_per_driver]
    return clean, wet_share


def iter_races(laps=None, seasons=None):
    """Yields (season, race_name, race_lap_count, race_df) for every race."""
    df = laps if laps is not None else load_laps(seasons)
    for (season, race), grp in df.groupby(['Season', 'Race'], sort=True):
        yield season, race, int(grp['race_laps'].iloc[0]), grp


if __name__ == '__main__':
    rebuild = '--rebuild' in sys.argv
    df = load_laps(rebuild=rebuild)
    print(f'\nSeasons : {sorted(df["Season"].unique())}')
    print(f'Races   : {df.groupby(["Season", "Race"]).ngroups}')
    print(f'Laps    : {len(df)}')
    print(f'Columns : {list(df.columns)}')

    if os.path.exists(results_path()):
        res = load_results()
        print(f'Results : {len(res)} entries, '
              f'{res["GridPosition"].notna().sum()} with a grid position')