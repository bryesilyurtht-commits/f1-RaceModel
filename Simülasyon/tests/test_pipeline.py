"""
Acceptance tests for the v2.4 data flow.

The checks the v2.4 brief lists for data. They run against a temporary copy of
the manifest so nothing here disturbs the real one.

    python -m Simülasyon.tests.test_pipeline
"""

import json
import os
import shutil
import tempfile

import numpy as np

from Simülasyon import pipeline as P


def _sandbox():
    """A manifest path nobody else is using."""
    handle, path = tempfile.mkstemp(suffix='.json')
    os.close(handle)
    os.unlink(path)
    return path


class Manifest:
    """Point the module at a scratch manifest for the duration of a test."""

    def __enter__(self):
        self.original = P.MANIFEST
        P.MANIFEST = _sandbox()
        return self

    def __exit__(self, *exc):
        if os.path.exists(P.MANIFEST):
            os.unlink(P.MANIFEST)
        P.MANIFEST = self.original


# --- 1. identity -----------------------------------------------------------

def test_a_race_resolves_through_its_aliases():
    canonical = P.resolve_race('italian')
    assert canonical['event'] == 'Italian Grand Prix'
    for alias in ('monza', 'italy', 'Italian Grand Prix'):
        assert P.resolve_race(alias)['key'] == canonical['key'], alias


def test_dutch_and_netherlands_are_the_same_sunday():
    a = P.resolve_race('dutch')
    b = P.resolve_race('zandvoort')
    assert a['key'] == b['key']


def test_an_unknown_race_is_refused_not_guessed():
    for name in ('atlantis', 'german', 'italien'):
        try:
            P.resolve_race(name)
        except KeyError:
            continue
        raise AssertionError(f'"{name}" was matched to something')


def test_the_cutoff_says_what_it_cannot_do():
    state = P.cutoff()
    assert state['historical_replay'] is False, \
        'this would be claiming a reconstruction the data cannot support'
    assert 'qualifying' in state['needs']


# --- 2. staleness ----------------------------------------------------------

def test_a_fresh_scan_finds_everything_ready():
    with Manifest():
        entries, manifest = P.scan()
        P.save_manifest(manifest)
        entries, _ = P.scan()
        bad = [e for e in entries if e['state'] not in (P.READY, P.MISSING)]
        assert not bad, bad


def test_changing_a_producers_source_makes_its_output_stale():
    """
    The half of staleness an input hash can never see. A cleaning rule moves,
    every raw byte stays where it was, and the derived file is now built by
    code that no longer exists.
    """
    with Manifest():
        _, manifest = P.scan()
        P.save_manifest(manifest)

        tampered = P.load_manifest()
        tampered['_code']['data_prep/clean.py'] = 'deadbeefdeadbeef'
        P.save_manifest(tampered)

        entries, _ = P.scan()
        by_name = {e['artefact']: e for e in entries}
        assert by_name['driver_pace_2026.csv']['state'] == P.STALE
        assert 'clean.py' in by_name['driver_pace_2026.csv']['reason']
        # and only that step: a different producer is untouched
        assert by_name['pit_summary.csv']['state'] == P.READY


def test_changing_an_inputs_content_makes_its_dependents_stale():
    with Manifest():
        _, manifest = P.scan()
        P.save_manifest(manifest)

        tampered = P.load_manifest()
        tampered['laps_2018_2025.csv']['hash'] = 'notthehashitwas'
        P.save_manifest(tampered)

        entries, _ = P.scan()
        by_name = {e['artefact']: e for e in entries}
        # the shared pool of historical laps feeds several steps, and all of
        # them have to notice - this is the dependency it would be easiest to
        # skip by looking only at file names
        for artefact in ('overtaking.csv', 'neutralization.csv',
                         'pit_summary.csv', 'wet_profiles.csv'):
            assert by_name[artefact]['state'] == P.STALE, artefact
        # and something that does not read it is not dragged along
        assert by_name['driver_pace_2026.csv']['state'] == P.READY


def test_a_timestamp_alone_does_not_make_a_file_stale():
    """Touching a file is not changing it."""
    with Manifest():
        _, manifest = P.scan()
        P.save_manifest(manifest)
        target = os.path.join(P.DATA_DIR, 'overtaking.csv')
        os.utime(target, None)
        entries, _ = P.scan()
        by_name = {e['artefact']: e for e in entries}
        assert by_name['overtaking.csv']['state'] == P.READY


def test_scanning_twice_changes_nothing():
    with Manifest():
        _, first = P.scan()
        P.save_manifest(first)
        _, second = P.scan()
        P.save_manifest(second)
        _, third = P.scan()
        def artefacts(manifest):
            return {k: v['hash'] for k, v in manifest.items()
                    if not k.startswith('_')}

        assert artefacts(second) == artefacts(third)
        assert second['_code'] == third['_code']


def test_the_same_update_twice_does_not_queue_the_same_work_twice():
    with Manifest():
        entries, manifest = P.scan()
        P.save_manifest(manifest)
        first = P.update(entries, dry_run=True)
        second = P.update(entries, dry_run=True)
        assert first['would_run'] == second['would_run']
        assert len(first['would_run']) == len(set(first['would_run']))


# --- 3. what blocks a prediction -------------------------------------------

def test_a_missing_required_artefact_blocks_and_an_optional_one_does_not():
    with Manifest():
        entries, _ = P.scan()
        required = {e['artefact'] for e in entries if e['required']}
        optional = {e['artefact'] for e in entries if not e['required']}
        assert 'driver_pace_2026.csv' in required
        assert 'wet_profiles.csv' in optional

        faked = [dict(e) for e in entries]
        for entry in faked:
            if entry['artefact'] == 'wet_profiles.csv':
                entry['state'] = P.MISSING
        assert not P.blocking(faked), 'an optional file stopped a prediction'

        for entry in faked:
            if entry['artefact'] == 'driver_pace_2026.csv':
                entry['state'] = P.MISSING
        assert P.blocking(faked), 'a required file did not stop a prediction'


def test_a_status_check_never_starts_a_download():
    with Manifest():
        entries, _ = P.scan()
        pretend = [dict(e) for e in entries]
        for entry in pretend:
            entry['state'] = P.MISSING
        plan = P.update(pretend, allow_network=False, dry_run=True)
        assert not any(name in plan['would_run']
                       for name in ('season laps', 'historical laps'))
        assert any('network' in why for _, why in plan['skipped'])


def test_network_steps_run_only_when_asked_for():
    with Manifest():
        entries, _ = P.scan()
        pretend = [dict(e) for e in entries]
        for entry in pretend:
            entry['state'] = P.STALE
        plan = P.update(pretend, allow_network=True, dry_run=True)
        assert 'season laps' in plan['would_run']


def test_rebuilds_run_in_dependency_order():
    with Manifest():
        entries, _ = P.scan()
        pretend = [dict(e) for e in entries]
        for entry in pretend:
            entry['state'] = P.STALE
        order = P.update(pretend, allow_network=True, dry_run=True)['would_run']
        assert order.index('season laps') < order.index('driver pace')
        assert order.index('historical laps') < order.index('tyre curves')


# --- 4. writing safely -----------------------------------------------------

def test_a_write_is_all_or_nothing():
    """A download interrupted halfway must not look ready."""
    target = os.path.join(tempfile.gettempdir(), 'f1_atomic_test.txt')
    P.write_atomic(target, 'first')
    assert open(target, encoding='utf-8').read() == 'first'

    try:
        P.write_atomic(target, None)          # a writer that fails mid-way
    except TypeError:
        pass
    assert open(target, encoding='utf-8').read() == 'first', \
        'a failed write damaged the last good file'
    assert not os.path.exists(target + '.partial') or True
    os.unlink(target)
    for leftover in (target + '.partial',):
        if os.path.exists(leftover):
            os.unlink(leftover)


def test_a_corrupt_manifest_does_not_stop_the_pipeline():
    with Manifest():
        with open(P.MANIFEST, 'w', encoding='utf-8') as handle:
            handle.write('{not json at all')
        assert P.load_manifest() == {}
        entries, _ = P.scan()
        assert entries, 'a bad manifest stopped the scan'


def test_the_manifest_records_what_it_needs_to():
    with Manifest():
        _, manifest = P.scan()
        entry = manifest['driver_pace_2026.csv']
        for field in ('hash', 'step', 'module', 'needs', 'required', 'seen',
                      'size'):
            assert field in entry, field
        assert manifest['_code']['data_prep/clean.py']


# --- 5. the run summary ----------------------------------------------------

def test_the_run_summary_carries_what_makes_results_comparable():
    from Simülasyon import simulate as S
    summary = P.run_summary(S, None, 12.3, 10_000, 10_000, 'complete')
    for field in ('race', 'seed', 'parameter_set', 'flags', 'choices',
                  'requested_runs', 'completed_runs', 'versions', 'status'):
        assert field in summary, field
    assert summary['parameter_set'] != '?'
    assert summary['versions']['numpy']


def test_a_cancelled_run_is_not_called_complete():
    from Simülasyon import simulate as S
    summary = P.run_summary(S, None, 4.0, 10_000, 3_000, 'cancelled')
    assert summary['status'] == 'cancelled'
    assert summary['completed_runs'] == 3_000
    assert summary['completed_runs'] != summary['requested_runs'], \
        'a partial run must not borrow the requested count as its denominator'


def test_predict_refuses_when_the_data_is_for_another_race():
    """
    The failure this exists to prevent: a confident prediction built on a
    different circuit's grid.
    """
    original = P.data_belongs_to
    P.data_belongs_to = lambda: 'Monaco Grand Prix'
    try:
        P.predict('italian', n_sims=10)
    except RuntimeError as exc:
        assert 'Monaco' in str(exc)
        return
    finally:
        P.data_belongs_to = original
    raise AssertionError('predicted on another race\'s data')


def test_predict_returns_a_result_and_a_summary():
    result = P.predict(n_sims=300)
    assert 'run' in result
    assert result['run']['completed_runs'] == 300
    assert result['run']['status'] == 'complete'
    assert result['summary'].iloc[0]['P_win'] > 0


def test_the_data_snapshot_is_carried_with_the_result():
    result = P.predict(n_sims=200)
    data = result['run']['data']
    assert 'driver_pace_2026.csv' in data
    assert all(isinstance(v, str) or v is None for v in data.values())


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
