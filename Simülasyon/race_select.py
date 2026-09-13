"""
F1 Prediction Simulation - race_select.py
Predicting any 2026 race that has been qualified for, not only the one on disk.

The problem
-----------
The project was built around one target race at a time. `fetch.py` withholds
that race's laps, `clean.py` builds pace from what is left, and `simulate.py`
resolves its pole and grid at import. Switching race meant editing
`target_race.py` and re-running the pipeline, which needs the network.

None of that is necessary to change race, because the season's data is already
here: `f1_2026_poles.csv` carries a pole for every round that has run, and
`f1_2026_quali.csv` carries the qualifying order for each. What was missing was
driver pace measured for a different target, and that is a recomputation rather
than a download.

The rule this enforces
----------------------
Pace for a target is built from the races **before** it, never from the target
itself and never from what came after. That is not an extra safeguard bolted
on; it is what the existing file already does for the current target, and
applying it uniformly is what makes the other races mean the same thing.

Predicting round 5 therefore uses rounds 1-4 and knows nothing about rounds
6-13, which is what a prediction made that weekend could have known. The result
is weaker for early rounds because four races of pace is four races of pace,
and that is the honest answer rather than a fixable defect.

What it cannot do
-----------------
A race that has not qualified has no pole and no grid, so it is not offered.
The model needs a starting order to run at all, and inventing one would produce
a confident answer about a weekend that has not happened.

The grid is the qualifying order
--------------------------------
For the race the pipeline was built for, `f1_2026_grid.csv` holds the official
grid - qualifying plus any penalties applied. For the others only the
qualifying order is on file. Penalties move cars, so these two are not the same
thing, and a race using the qualifying order says so rather than presenting it
as the grid.
"""

import os
import shutil
import tempfile

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASON = 2026

# Files rebuilt per target. Everything else in data/ is either per-circuit and
# fitted on 2018-2025 - which cannot contain a 2026 race - or season-wide and
# unaffected by which round is being predicted, so it is copied through.
PER_RACE = ('driver_pace_2026.csv', 'f1_2026_poles.csv', 'f1_2026_grid.csv',
            'f1_2026_quali.csv')

MIN_RACES_OF_PACE = 2       # below this, pace is not a measurement


def _poles(data_dir=None):
    return pd.read_csv(os.path.join(data_dir or DATA_DIR,
                                    f'f1_{SEASON}_poles.csv'))


def _quali(data_dir=None):
    return pd.read_csv(os.path.join(data_dir or DATA_DIR,
                                    f'f1_{SEASON}_quali.csv'))


def _laps(data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, f'f1_{SEASON}_laps.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def registry_key(event_name):
    """
    The `target_race.py` key for an event name the season files use.

    The two vocabularies do not match everywhere - the season files call round
    7 "Barcelona Grand Prix" where the registry knows it as `spanish` - so the
    match is on the registry's own aliases rather than on the name.
    """
    from Simülasyon import target_race as TR

    text = str(event_name).lower()
    for key, (event, aliases) in TR.RACE_REGISTRY.items():
        if text == event.lower() or any(a in text for a in aliases):
            return key
    return None


def available(data_dir=None):
    """
    Every 2026 race that can be predicted, and why the rest cannot.

    A race qualifies when it has a pole time, a qualifying order, and enough
    races before it for pace to mean anything. Each row says which of those it
    has, so a race that is missing from the selector can be explained rather
    than silently absent.
    """
    poles = _poles(data_dir).sort_values('Round')
    quali = _quali(data_dir)
    laps = _laps(data_dir)

    with_quali = set(quali['Round'])
    lap_rounds = set()
    if laps is not None:
        by_name = dict(zip(poles['Race'], poles['Round']))
        lap_rounds = {by_name[r] for r in laps['Race'].unique() if r in by_name}

    rows = []
    for _, row in poles.iterrows():
        rnd, event = int(row['Round']), str(row['Race'])
        earlier = len([r for r in lap_rounds if r < rnd])
        key = registry_key(event)

        if pd.isna(row['pole_time']):
            state, why = 'unavailable', 'qualifying has not run'
        elif rnd not in with_quali:
            state, why = 'unavailable', 'no qualifying order on file'
        elif key is None:
            state, why = 'unavailable', 'not in the circuit registry'
        elif earlier < MIN_RACES_OF_PACE:
            state, why = 'unavailable', (f'only {earlier} race(s) of pace '
                                         f'before it')
        else:
            state, why = 'available', f'{earlier} races of pace before it'

        rows.append(dict(round=rnd, event=event, key=key,
                         pole_driver=str(row['pole_driver']),
                         pole_time=row['pole_time'],
                         is_pipeline_target=bool(row.get('is_target', False)),
                         races_of_pace=earlier, state=state, reason=why))
    return pd.DataFrame(rows)


def build_pace(target_round, data_dir=None):
    """
    Driver pace for this target, from the races before it and nothing else.

    Reuses `clean.build_driver_pace`, so this is the project's own definition
    of pace rather than a second implementation that agrees on the easy cases.
    The only thing decided here is which races it is allowed to see.
    """
    from Simülasyon.data_prep import clean as C

    laps = _laps(data_dir)
    if laps is None:
        return None, 'no 2026 lap data on disk'

    poles = _poles(data_dir)
    round_of = dict(zip(poles['Race'], poles['Round']))
    laps = laps.copy()
    laps['Round'] = laps['Race'].map(round_of)

    before = laps[laps['Round'].notna() & (laps['Round'] < target_round)]
    if before.empty:
        return None, f'no races before round {target_round}'

    flagged = _clean_flag(before)
    clean = flagged[flagged['clean_lap']].copy()
    if clean.empty:
        return None, 'no clean laps before this race'

    pace, _, _, _ = C.build_driver_pace(
        clean, poles[['Race', 'pole_time']])
    if len(pace) < 10:
        return None, f'only {len(pace)} drivers with pace'
    return pace, None


def _clean_flag(laps):
    """
    The project's clean-lap rules, over whichever column names are present.

    fetch.py writes raw FastF1 columns for the current season and the archive
    carries them already parsed. Rather than branch on which, this reads
    whichever of each pair exists - the rules themselves are clean.py's.
    """
    from Simülasyon.data_prep import clean as C

    df = laps.copy()
    if 'LapTime_s' not in df.columns:
        df['LapTime_s'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    pit_in = 'PitInTime_s' if 'PitInTime_s' in df.columns else 'PitInTime'
    pit_out = 'PitOutTime_s' if 'PitOutTime_s' in df.columns else 'PitOutTime'

    df['Compound'] = df['Compound'].astype(str).str.upper()
    df['TrackStatus'] = df['TrackStatus'].astype(str)
    df['is_wet'] = df['Compound'].isin(C.WET_COMPOUNDS)

    green = df['TrackStatus'].isin(['1', '1.0'])
    df['clean_lap'] = (df['LapTime_s'].notna() & green
                       & df[pit_in].isna() & df[pit_out].isna()
                       & (df['LapNumber'] > 1) & (df['TyreLife'] > 1)
                       & ~df['is_wet'])

    for _, group in df[df['clean_lap']].groupby('Race'):
        cutoff = group['LapTime_s'].median() * C.OUTLIER_MARGIN
        df.loc[group.index[group['LapTime_s'] > cutoff], 'clean_lap'] = False
    return df


def build_inputs(target_round, into=None, data_dir=None):
    """
    A data directory the model can be pointed at for this race.

    Everything in data/ is copied and then the four per-race files are
    replaced. Copying rather than editing in place is the point: a race
    switched in the interface must not be able to damage the inputs of the
    pipeline's own target, and a failure half way through leaves the real
    directory untouched.
    """
    source = data_dir or DATA_DIR
    poles = _poles(source)
    row = poles[poles['Round'] == target_round]
    if row.empty:
        return None, f'round {target_round} is not in the poles table'
    event = str(row.iloc[0]['Race'])

    pace, why = build_pace(target_round, source)
    if pace is None:
        return None, why

    quali = _quali(source)
    entries = quali[quali['Round'] == target_round].sort_values('quali_pos')
    if entries.empty:
        return None, 'no qualifying order for this race'

    path = into or tempfile.mkdtemp(prefix=f'race_{target_round}_')
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(source):
        src = os.path.join(source, name)
        if os.path.isfile(src) and name not in PER_RACE:
            shutil.copy2(src, os.path.join(path, name))

    pace.to_csv(os.path.join(path, f'driver_pace_{SEASON}.csv'), index=False)

    marked = poles.copy()
    marked['is_target'] = marked['Round'] == target_round
    marked.to_csv(os.path.join(path, f'f1_{SEASON}_poles.csv'), index=False)

    quali_out = quali.copy()
    quali_out['is_target'] = quali_out['Round'] == target_round
    quali_out.to_csv(os.path.join(path, f'f1_{SEASON}_quali.csv'), index=False)

    # The official grid exists on disk only for the pipeline's own target. For
    # every other race the qualifying order stands in, and the caller is told
    # which it got so the interface can say so rather than implying penalties
    # were accounted for.
    official = os.path.join(source, f'f1_{SEASON}_grid.csv')
    grid_source = 'qualifying order'
    if row.iloc[0].get('is_target') and os.path.exists(official):
        shutil.copy2(official, os.path.join(path, f'f1_{SEASON}_grid.csv'))
        grid_source = 'official grid'
    else:
        entries.assign(GridPosition=range(1, len(entries) + 1))[
            ['Driver', 'Team', 'GridPosition']].to_csv(
            os.path.join(path, f'f1_{SEASON}_grid.csv'), index=False)

    return {'path': path, 'event': event, 'round': target_round,
            'key': registry_key(event), 'grid_source': grid_source,
            'drivers': len(pace),
            'races_of_pace': int(pace['n_races'].max()) if len(pace) else 0}, None


def activate(target_round, data_dir=None):
    """
    Point the model at a race and hand back the reloaded module.

    `simulate` resolves the pole, the circuit and the grid while its module
    body runs, so switching race means re-importing it rather than assigning
    to it. The environment overrides added in v2.6 are what make that a
    supported operation instead of a trick.

    The caller owns the returned directory. Nothing here writes into data/.
    """
    import importlib

    built, why = build_inputs(target_round, data_dir=data_dir)
    if built is None:
        return None, why

    os.environ['F1_DATA_DIR'] = built['path']
    os.environ['F1_SEASON'] = str(SEASON)
    if built['key']:
        os.environ['F1_TARGET_RACE'] = built['key']

    from Simülasyon import target_race as TR
    from Simülasyon import simulate as S
    importlib.reload(TR)
    module = importlib.reload(S)
    return {**built, 'module': module}, None


def restore():
    """
    Put the model back on the project's own target and data directory.

    Clearing the overrides is not enough on its own: `simulate` read them while
    its module body ran, so it keeps pointing at the old directory until it is
    re-imported.
    """
    import importlib

    for name in ('F1_DATA_DIR', 'F1_SEASON', 'F1_TARGET_RACE'):
        os.environ.pop(name, None)

    from Simülasyon import target_race as TR
    from Simülasyon import simulate as S
    importlib.reload(TR)
    return importlib.reload(S)


def release(built, restore_module=True):
    """
    Remove the directory `activate` or `build_inputs` created.

    The module is put back first, and that ordering is the whole point. Deleting
    the directory while `simulate` still pointed at it left the module reading a
    path that no longer existed - `resolve_pole` found no file, fell through to
    its fallback constant, and the next run produced a confident prediction for
    the wrong circuit's pole. It failed by carrying on, which is the failure
    this project takes seriously.
    """
    if restore_module and os.environ.get('F1_DATA_DIR'):
        restore()
    path = (built or {}).get('path')
    if path and os.path.isdir(path) and os.path.basename(path) != 'data':
        shutil.rmtree(path, ignore_errors=True)


def main():
    frame = available()
    print(f'\n=== 2026 races on disk ===')
    for _, r in frame.iterrows():
        mark = ' (pipeline target)' if r['is_pipeline_target'] else ''
        print(f'  R{r["round"]:<3d} {r["event"]:<26s} {r["state"]:<12s} '
              f'{r["reason"]}{mark}')
    ready = frame[frame.state == 'available']
    print(f'\n  {len(ready)} of {len(frame)} can be predicted')


if __name__ == '__main__':
    main()
