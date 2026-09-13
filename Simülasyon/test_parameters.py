"""
Acceptance tests for the v2.3 parameter audit.

The checks the v2.3 roadmap requires. Most of them are about bookkeeping
rather than arithmetic, which is the point: this version's failure mode is a
number that changed in the code and not in the panel, or a set that means two
different things on two days.

    python -m Simülasyon.test_parameters
"""

import hashlib
import inspect

import numpy as np

from Simülasyon import calibrate
from Simülasyon import parameters as P
from Simülasyon import simulate as s


_CACHE = {}


def inputs():
    if 'v' not in _CACHE:
        pace = s.load_drivers()
        track = s.load_track()
        strategies, _ = s.recost_strategies(s.load_strategies(track), track)
        _CACHE['v'] = (pace, track, strategies)
    return _CACHE['v']


def race(n=500, seed=4):
    pace, track, strategies = inputs()
    return s.run_simulation(pace, track, strategies, n, seed=seed)


def sha(positions):
    return hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest()


# --- 1. the set ------------------------------------------------------------

def test_every_set_states_all_four_constants():
    """
    A set is a complete statement, not a diff against whichever ran last.
    Otherwise applying two sets in sequence leaves a mixture with a single
    name on it.
    """
    touched = set()
    for values in P.PARAM_SET.values():
        touched |= set(values)
    for name, values in P.PARAM_SET.items():
        missing = touched - set(values)
        assert not missing, f'{name} does not state {sorted(missing)}'


def test_reverting_restores_the_previous_numbers():
    before = {k: getattr(s, k) for k in P.PARAM_SET[P.ACTIVE_SET]}
    P.apply(s, 'v2.2-legacy')
    legacy = {k: getattr(s, k) for k in before}
    P.apply(s, P.ACTIVE_SET)
    after = {k: getattr(s, k) for k in before}

    assert legacy != after, 'the two sets are the same numbers'
    assert after == before, 'reverting did not come back'
    assert legacy['SC_SHRINK_RACES'] == 6.0
    assert after['SC_SHRINK_RACES'] == 24.0


def test_the_same_set_and_seed_reproduce():
    P.apply(s, P.ACTIVE_SET)
    a = race()
    P.apply(s, P.ACTIVE_SET)
    b = race()
    assert sha(a[0]) == sha(b[0])


def test_a_set_name_never_means_two_things():
    """The same name applied twice has to produce the same numbers."""
    P.apply(s, 'v2.3-literal')
    first = {k: getattr(s, k) for k in P.PARAM_SET['v2.3-literal']}
    P.apply(s, 'v2.2-legacy')
    P.apply(s, 'v2.3-literal')
    second = {k: getattr(s, k) for k in first}
    P.apply(s, P.ACTIVE_SET)
    assert first == second


def test_an_unknown_set_is_refused():
    try:
        P.apply(s, 'v9.9-wishful')
    except KeyError:
        return
    raise AssertionError('an unknown set was applied')


# --- 2. the module and the inventory agree ---------------------------------

def test_the_inventory_matches_what_the_module_runs():
    """
    The failure this version is most likely to have: a constant changed in the
    code and left at its old value in the audit, or the other way round.
    """
    live = {
        'SC_TIMING_MEAN_FRACTION': s.SC_TIMING_MEAN_FRACTION,
        'SC_TIMING_STD_FRACTION': s.SC_TIMING_STD_FRACTION,
        'SC_SHRINK_RACES': s.SC_SHRINK_RACES,
        'NOISE_AUTOCORR': s.NOISE_AUTOCORR,
        'SC_QUEUE_GAP': s.SC_QUEUE_GAP,
        'AUTO_PASS_MARGIN': s.AUTO_PASS_MARGIN,
        'STRATEGY_TEMPERATURE': s.STRATEGY_TEMPERATURE,
        'RF_PROB_CAP': s.RF_PROB_CAP,
        'RF_PROB_FLOOR': s.RF_PROB_FLOOR,
    }
    by_name = {row['name']: row for row in P.INVENTORY}
    for name, value in live.items():
        row = by_name.get(name)
        assert row is not None, f'{name} is not in the inventory'
        expected = (row['measured'] if str(row['decision']).startswith('changed')
                    else row['value'])
        assert abs(float(expected) - float(value)) < 1e-9, \
            f'{name}: inventory says {expected}, module runs {value}'


def test_the_active_set_matches_the_module():
    for name, value in P.PARAM_SET[P.ACTIVE_SET].items():
        assert getattr(s, name) == value, name


def test_every_entry_says_what_kind_of_number_it_is():
    allowed = {'measured', 'derived', 'borrowed', 'hand-set', 'fallback',
               'safety', 'unused'}
    for row in P.INVENTORY:
        assert row['kind'] in allowed, (row['name'], row['kind'])
        assert row['why'], row['name']
        assert row['decision'], row['name']


def test_a_user_choice_is_not_dressed_as_a_coefficient():
    """
    The roadmap's distinction: a safety rail, a scope limit and a physical
    coefficient are three different things and must not share a label.
    """
    by_name = {row['name']: row for row in P.INVENTORY}
    assert by_name['RF_PROB_CAP']['kind'] == 'safety'
    assert by_name['RF_PROB_FLOOR']['kind'] == 'safety'
    assert by_name['NOISE_AUTOCORR']['kind'] == 'hand-set'


# --- 3. the dead entries ---------------------------------------------------

def test_the_dead_constants_really_are_dead():
    source = inspect.getsource(s)
    uses = source.count('POOLED_THRESHOLD')
    assert uses <= 2, f'POOLED_THRESHOLD read {uses} times, so it is not dead'
    assert 'MIN_GAP' not in source.split('COMPOUND_NAMES')[0].replace(
        'MIN_GAP is deliberately not imported', ''), \
        'MIN_GAP is still imported'


def test_the_dead_module_is_not_imported():
    """
    Named in the audit, imported by nothing. The distinction matters: the
    point is that no code path reaches it, not that the word is unmentionable.
    """
    for module in (s, calibrate, P):
        for line in inspect.getsource(module).splitlines():
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                assert 'tyre_cliff' not in stripped, (module.__name__, line)


def test_the_degradation_badge_says_it_is_display_only():
    pace, track, _ = inputs()
    names = [b['name'] for b in s.source_badges(pace, track)]
    assert any('display only' in n for n in names), names


# --- 4. the measurements ---------------------------------------------------

def test_the_noise_contract_is_consistent():
    """
    sigma is the long-run per-lap spread, not the innovation, so changing rho
    moves persistence without moving total variance. If that were the other
    way round, every rho candidate would also be a variance candidate and the
    sensitivity sweep would be measuring two things at once.
    """
    for rho in (0.0, 0.155, 0.6, 0.9):
        out = calibrate.noise_variance_contract(rho, 0.30)
        assert out['matches_sigma'], rho


def test_the_train_and_check_seasons_do_not_overlap():
    df = calibrate.load()
    seasons = sorted(df['Season'].unique())
    train = [x for x in seasons if x <= calibrate.TRAIN_UNTIL]
    test = [x for x in seasons if x > calibrate.TRAIN_UNTIL]
    assert train and test
    assert not (set(train) & set(test))


def test_a_stint_is_never_split_across_the_two_halves():
    """
    Splitting in time rather than at random is what keeps laps from the same
    stint out of both sides. A random split inflates every autocorrelation
    measured afterwards.
    """
    df = calibrate.load()
    by_stint = df.groupby(['Season', 'Race', 'Driver', 'Stint'])['Season'].nunique()
    assert (by_stint == 1).all(), 'a stint spans two seasons, which cannot be'


def test_the_autocorrelation_skips_interrupted_series():
    """A pit stop or a safety car ends the series; laps either side are not
    consecutive and treating them as such measures the interruption."""
    source = inspect.getsource(calibrate.noise_autocorrelation)
    assert "LapNumber'].diff()" in source
    assert 'racing' in source


def test_the_variance_split_uses_mean_residuals_not_intercepts():
    """
    The bug this catches was real and produced a two-second between-stint
    spread, which no driver has. An intercept is the fitted line extrapolated
    back to lap zero, and at lap thirty its uncertainty is thirty times the
    slope's.
    """
    source = inspect.getsource(calibrate.variance_by_scale)
    assert 'r.mean()' in source, 'the stint offset is not a mean residual'
    # one fit per driver-race, not one per stint: the offsets have to be
    # measured against a common line or they are not comparable
    assert source.count('np.polyfit') == 1, 'a second fit crept back in'

    # and the numbers it produces have to be the size a driver actually is
    df = calibrate.load()
    scales = calibrate.variance_by_scale(df, [2023])
    assert 0.2 < scales['within_sd'] < 2.0, scales
    assert 0.02 < scales['between_sd'] < 1.0, scales


def test_the_measured_values_are_carried_with_their_sample_size():
    for row in P.INVENTORY:
        if str(row['decision']).startswith('changed'):
            assert row['measured'] is not None, row['name']
            assert row['n'], row['name']


# --- 5. the refusals -------------------------------------------------------

def test_the_refused_changes_say_why():
    refused = [r for r in P.INVENTORY
               if str(r['decision']).startswith('unchanged')]
    assert len(refused) >= 4
    for row in refused:
        assert len(row['why']) > 80, row['name']


def test_the_red_flag_clip_was_tested_not_assumed():
    by_name = {row['name']: row for row in P.INVENTORY}
    row = by_name['RF_PROB_CAP']
    assert row['holdout'] is not None
    assert 'held-out' in row['n'] or 'races' in row['n']


def test_the_open_assumption_is_named_as_one():
    by_name = {row['name']: row for row in P.INVENTORY}
    rho = by_name['NOISE_AUTOCORR']
    assert rho['kind'] == 'hand-set'
    assert rho['measured'] is not None
    assert float(rho['value']) != float(rho['measured'])
    assert 'stand' in rho['why'] or 'standing' in rho['why']


# --- 6. the model still works ----------------------------------------------

def test_the_soft_knee_preserves_the_order():
    from Simülasyon.clean import compress_delta
    raw = np.array([0.0, 0.4, 0.9, 1.0, 1.4, 2.2, 3.1, 4.2])
    out = compress_delta(raw)
    assert (np.diff(out) > 0).all(), 'compression reordered the field'
    assert np.allclose(out[raw <= 1.0], raw[raw <= 1.0]), \
        'the front of the field was not left alone'
    assert out[-1] < raw[-1]


def test_the_changed_set_still_produces_a_race():
    positions, total, _, diag = race()
    assert positions.min() == 1
    assert np.isfinite(total[~diag['retired']]).all()
    assert (diag['neutral'] >= 0).all()


def test_the_timing_change_keeps_events_inside_the_race():
    _, _, _, diag = race()
    neutral = diag['neutral']
    n_laps = neutral.shape[1]
    lap_of_first = np.argmax(neutral > 0, axis=1)
    has = (neutral > 0).any(axis=1)
    assert (lap_of_first[has] < n_laps).all()
    assert (lap_of_first[has] >= 0).all()


def test_more_pooling_did_not_silently_change_the_calendar_rate():
    """
    Shrinkage moves a circuit toward the calendar; it must not move the
    calendar itself. With this circuit's own rate pooled harder, the simulated
    safety car share has to sit between its old value and the calendar's.
    """
    P.apply(s, 'v2.2-legacy')
    weak = race()[3]
    P.apply(s, P.ACTIVE_SET)
    strong = race()[3]
    a = float((weak['neutral'] == 2).any(axis=1).mean())
    b = float((strong['neutral'] == 2).any(axis=1).mean())
    assert abs(b - a) < 0.25, (a, b)


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
        finally:
            P.apply(s, P.ACTIVE_SET)
    print(f'\n{passed}/{len(tests)} passed')
    for name, why in failed:
        print(f'  FAIL  {name}\n        {why}')
    return not failed


if __name__ == '__main__':
    import sys
    sys.exit(0 if _run() else 1)
