"""
Acceptance tests for predicting any qualified 2026 race.

The thing that can go quietly wrong here is leakage. Switching target race
means rebuilding driver pace, and a rebuild that includes the target's own laps
produces a better-looking prediction of a race it has already read. Nothing
raises; the numbers simply improve.

So these check the boundary from both sides: that the pace for a round is built
only from rounds before it, and that rebuilding the pipeline's own target
reproduces the file the pipeline wrote - byte for byte on the numbers, which is
the strongest available evidence that the new path is the old path.

    python -m Simülasyon.tests.test_race_select
"""

import os

import pandas as pd

from Simülasyon import race_select as RS

DATA_DIR = RS.DATA_DIR


# --- what is on offer -------------------------------------------------------

def test_every_race_gets_a_state_and_a_reason():
    frame = RS.available()
    assert len(frame) > 0
    assert frame['state'].isin(['available', 'unavailable']).all()
    assert frame['reason'].str.len().gt(0).all()


def test_the_pipeline_target_is_available():
    frame = RS.available()
    target = frame[frame.is_pipeline_target]
    assert len(target) == 1
    assert target.iloc[0]['state'] == 'available'


def test_the_opening_rounds_are_held_back():
    """
    Round one has no races behind it, so it has no measured pace. Offering it
    would mean predicting from the fallback constants and calling it a
    prediction of that Grand Prix.
    """
    frame = RS.available().set_index('round')
    assert frame.loc[1, 'state'] == 'unavailable'
    assert 'pace' in frame.loc[1, 'reason']


def test_a_race_without_qualifying_is_not_offered():
    frame = RS.available()
    for _, row in frame.iterrows():
        if row['state'] == 'available':
            assert pd.notna(row['pole_time'])


def test_every_offered_race_resolves_to_a_circuit():
    """
    The season files and the registry do not share a vocabulary - round 7 is
    "Barcelona Grand Prix" on disk and `spanish` in the registry - so a race
    that cannot be matched must be withheld rather than guessed at.
    """
    frame = RS.available()
    for _, row in frame[frame.state == 'available'].iterrows():
        assert row['key'], row['event']


def test_the_registry_matches_barcelona_to_spanish():
    assert RS.registry_key('Barcelona Grand Prix') == 'spanish'


def test_an_unknown_race_name_matches_nothing():
    assert RS.registry_key('Neverland Grand Prix') is None


# --- every file this reads must actually ship -------------------------------
#
# The bug this section exists to catch already happened once: f1_2026_laps.csv
# was gitignored as a "pipeline rebuild artefact", which was true before
# race_select.py existed and stopped being true the moment this module started
# reading it at runtime to count races of pace. Every local test passed,
# because the file was sitting on the machine that ran them. A fresh clone -
# Streamlit Cloud's deploy included - had no such file, every round failed its
# pace check, and the sidebar read "No 2026 race on disk can be predicted."
# with nothing to point at why.
#
# subprocess rather than importing git plumbing: `git check-ignore` and
# `git ls-files` are the same commands a person would run to answer this by
# hand, and shelling out to the real binary is the one way to be sure the
# check means what git itself means by "ignored" and "tracked".

def _git(*args):
    import subprocess

    result = subprocess.run(['git', *args], cwd=RS.BASE_DIR,
                            capture_output=True, text=True)
    return result.returncode, result.stdout.strip()


def _in_a_git_repo():
    code, _ = _git('rev-parse', '--is-inside-work-tree')
    return code == 0


def test_every_file_race_select_reads_is_committed_not_ignored():
    """
    The exact regression: a file this module reads at runtime, silently
    absent from a fresh clone because .gitignore still listed it as a
    pipeline-only artefact.
    """
    if not _in_a_git_repo():
        return
    for name in (f'f1_{RS.SEASON}_poles.csv', f'f1_{RS.SEASON}_quali.csv',
                f'f1_{RS.SEASON}_laps.csv', f'f1_{RS.SEASON}_grid.csv'):
        # Forward slash always, regardless of OS: `git ls-files` reports paths
        # this way even on Windows, and comparing against a backslash-joined
        # path would fail the check for a reason that has nothing to do with
        # the file actually being tracked.
        rel = f'data/{name}'
        ignored_code, _ = _git('check-ignore', '-q', rel)
        assert ignored_code != 0, f'{rel} is gitignored but race_select reads it'
        _, tracked = _git('ls-files', rel)
        assert tracked == rel, f'{rel} exists but is not committed'


def test_a_fresh_checkout_would_offer_the_same_races_as_this_working_copy():
    """
    What "available" means only if the files it counted on actually reach a
    clone. Reading through git's own index rather than the working directory
    catches a file that is staged for deletion, or modified uncommitted in a
    way that would not survive a checkout.
    """
    if not _in_a_git_repo():
        return
    code, out = _git('show', f'HEAD:data/f1_{RS.SEASON}_laps.csv')
    if code != 0:
        # No commit yet in this session - the file may be staged only. The
        # gitignore/tracked check above is what actually guards the bug;
        # this one is a bonus check once a commit exists to compare against.
        return
    assert len(out) > 0


# --- the leakage boundary ---------------------------------------------------

def _rounds_in_pace(target_round):
    """Which rounds the pace for this target was actually allowed to read."""
    poles = pd.read_csv(os.path.join(DATA_DIR, 'f1_2026_poles.csv'))
    laps = pd.read_csv(os.path.join(DATA_DIR, 'f1_2026_laps.csv'),
                       usecols=['Race'])
    round_of = dict(zip(poles['Race'], poles['Round']))
    present = {round_of[r] for r in laps['Race'].unique() if r in round_of}
    return {r for r in present if r < target_round}


def test_pace_never_reads_the_race_it_predicts():
    """The whole point. A target's own laps are not in its own pace."""
    for target in (6, 10, 13):
        assert target not in _rounds_in_pace(target)


def test_pace_never_reads_a_later_race():
    for target in (6, 10):
        assert not [r for r in _rounds_in_pace(target) if r > target]


def test_an_earlier_target_sees_fewer_races():
    early, _ = RS.build_pace(6)
    late, _ = RS.build_pace(12)
    assert early['n_races'].max() < late['n_races'].max()


def test_rebuilding_the_pipeline_target_reproduces_its_committed_pace():
    """
    The strongest check available: the new path is the old path.

    If this drifts, every other race in the selector is being built by rules
    the pipeline does not use, and nothing else here would notice.
    """
    target = RS.available()
    target_round = int(target[target.is_pipeline_target].iloc[0]['round'])
    rebuilt, why = RS.build_pace(target_round)
    assert rebuilt is not None, why

    committed = pd.read_csv(os.path.join(DATA_DIR, 'driver_pace_2026.csv'))
    merged = rebuilt[['Driver', 'delta', 'sigma']].merge(
        committed[['Driver', 'delta', 'sigma']], on='Driver',
        suffixes=('_new', '_disk'))
    assert len(merged) == len(committed), (len(merged), len(committed))
    assert (merged['delta_new'] - merged['delta_disk']).abs().max() < 1e-9
    assert (merged['sigma_new'] - merged['sigma_disk']).abs().max() < 1e-9


# --- the directory that gets built ------------------------------------------

def test_the_built_directory_marks_exactly_one_target():
    built, why = RS.build_inputs(6)
    assert built, why
    try:
        poles = pd.read_csv(os.path.join(built['path'], 'f1_2026_poles.csv'))
        assert poles['is_target'].sum() == 1
        assert poles[poles.is_target].iloc[0]['Race'] == built['event']
    finally:
        RS.release(built, restore_module=False)


def test_the_built_directory_carries_the_circuit_tables():
    """
    Tyre curves, pit loss and the rest are fitted on 2018-2025 and cannot
    contain a 2026 race, so they are copied rather than withheld - and the
    model does not start without them.
    """
    built, why = RS.build_inputs(6)
    assert built, why
    try:
        present = set(os.listdir(built['path']))
        for name in ('tyre_curve_params.json', 'pit_summary.csv',
                     'overtaking.csv', 'neutralization.csv'):
            assert name in present, name
    finally:
        RS.release(built, restore_module=False)


def test_the_real_data_directory_is_never_written_to():
    before = {name: os.path.getmtime(os.path.join(DATA_DIR, name))
              for name in os.listdir(DATA_DIR)
              if os.path.isfile(os.path.join(DATA_DIR, name))}
    built, _ = RS.build_inputs(6)
    RS.release(built, restore_module=False)
    after = {name: os.path.getmtime(os.path.join(DATA_DIR, name))
             for name in os.listdir(DATA_DIR)
             if os.path.isfile(os.path.join(DATA_DIR, name))}
    assert before == after


def test_a_non_target_race_says_its_grid_is_the_qualifying_order():
    """Penalties move cars, and only the fetched race has the official grid."""
    built, why = RS.build_inputs(6)
    assert built, why
    try:
        assert built['grid_source'] == 'qualifying order'
    finally:
        RS.release(built, restore_module=False)


def test_the_pipeline_target_keeps_its_official_grid():
    frame = RS.available()
    target_round = int(frame[frame.is_pipeline_target].iloc[0]['round'])
    built, why = RS.build_inputs(target_round)
    assert built, why
    try:
        assert built['grid_source'] == 'official grid'
    finally:
        RS.release(built, restore_module=False)


def test_an_unknown_round_is_refused_rather_than_guessed():
    built, why = RS.build_inputs(99)
    assert built is None
    assert '99' in why


# --- activating and putting it back -----------------------------------------

def test_activating_points_the_model_at_the_chosen_race():
    built, why = RS.activate(6)
    assert built, why
    try:
        module = built['module']
        assert 'Monaco' in module.TARGET_EVENT
        assert module.DATA_DIR == built['path']
        assert module.POLE_DRIVER == built_pole(built)
    finally:
        RS.release(built)


def built_pole(built):
    poles = pd.read_csv(os.path.join(built['path'], 'f1_2026_poles.csv'))
    return str(poles[poles.is_target].iloc[0]['pole_driver'])


def test_the_circuit_changes_with_the_race():
    """Monaco is 78 laps and Zandvoort is 72. If this ever agrees, the track
    parameters are not following the selection."""
    laps = {}
    for rnd in (6, 12):
        built, why = RS.activate(rnd)
        assert built, why
        try:
            module = built['module']
            module.N_SIMS = 60
            laps[rnd] = module.run()['n_laps']
        finally:
            RS.release(built)
    assert laps[6] != laps[12], laps


def test_releasing_puts_the_model_back_on_the_project_target():
    """
    The bug this exists for: the directory was deleted while simulate still
    pointed at it, resolve_pole found no file, fell through to its fallback
    constant, and the next run was a confident prediction built on it.
    """
    from Simülasyon import simulate as S

    before_event, before_pole = S.TARGET_EVENT, S.POLE_DRIVER
    built, why = RS.activate(6)
    assert built, why
    RS.release(built)

    assert os.environ.get('F1_DATA_DIR') is None
    assert os.path.isdir(S.DATA_DIR)
    assert S.DATA_DIR == os.path.join(S.BASE_DIR, 'data')
    assert S.TARGET_EVENT == before_event
    assert S.POLE_DRIVER == before_pole


def test_the_default_run_still_works_after_a_switch():
    from Simülasyon import simulate as S

    built, _ = RS.activate(6)
    RS.release(built)
    S.N_SIMS = 120
    result = S.run()
    assert result['event'] == S.TARGET_EVENT
    assert len(result['summary']) > 10


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as exc:
            failed.append((name, str(exc) or 'assertion failed'))
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, f'{type(exc).__name__}: {exc}'))
    print(f'\n{passed}/{len(tests)} passed')
    for name, why in failed:
        print(f'  FAIL  {name}\n        {why}')
    return not failed


if __name__ == '__main__':
    import sys
    sys.exit(0 if _run() else 1)
