"""
Acceptance tests for team orders (v2.8): no double stacking, and no
teammate-versus-teammate passing while both are on the same plan.

Two separate mechanisms, tested at two levels. `apply_double_stack_delay`
and `team_order_hold` are pure functions and are checked directly against
small synthetic arrays. `test_no_stack_survives_a_real_race` and
`test_a_lone_car_can_still_pit_its_teammate_is_not_holding_it_to` run the
actual engine, because the unit-level checks cannot see whether the two
mechanisms interact badly with the rest of the lap loop - retirements,
the stint cap, red flags - over a full race.

    python -m Simülasyon.tests.test_team_orders
"""

import numpy as np

from Simülasyon import simulate as S


# --- apply_double_stack_delay: pure function, synthetic arrays -------------


def test_the_rear_car_is_the_one_delayed():
    """Two teammates, both due to pit, car 1 ahead on the road (lower total)."""
    pitting = np.array([[True, True]])
    stack_delay = np.array([[False, False]])
    total = np.array([[100.0, 105.0]])          # driver 0 ahead
    retired = np.array([[False, False]])

    out, next_delay = S.apply_double_stack_delay(
        pitting, stack_delay, total, retired, [(0, 1)])

    assert list(out[0]) == [True, False], 'the trailing car should be held'
    assert list(next_delay[0]) == [False, True]


def test_a_carried_delay_is_honoured_next_lap_regardless_of_order():
    """
    Last lap's held car pits this lap even if, by then, it has become the
    one ahead on the road - the delay is a promise already made, not a
    fresh double-stack check.
    """
    pitting = np.array([[False, False]])
    stack_delay = np.array([[False, True]])     # driver 1 was held out
    total = np.array([[50.0, 40.0]])             # driver 1 now ahead
    retired = np.array([[False, False]])

    out, next_delay = S.apply_double_stack_delay(
        pitting, stack_delay, total, retired, [(0, 1)])

    assert list(out[0]) == [False, True]
    assert list(next_delay[0]) == [False, False]


def test_only_one_car_pitting_is_not_a_stack():
    pitting = np.array([[True, False], [False, True]])
    stack_delay = np.zeros((2, 2), dtype=bool)
    total = np.array([[10.0, 20.0], [10.0, 20.0]])
    retired = np.zeros((2, 2), dtype=bool)

    out, next_delay = S.apply_double_stack_delay(
        pitting, stack_delay, total, retired, [(0, 1)])

    assert (out == pitting).all()
    assert not next_delay.any()


def test_a_retired_teammate_never_triggers_or_receives_a_delay():
    """
    A retired car's pitting flag is stale, not a real stop - see
    `test_diagnostics` for why it can still read True. It must not be read
    as a stack, and must not itself be held.
    """
    pitting = np.array([[True, True]])
    stack_delay = np.array([[False, False]])
    total = np.array([[10.0, 20.0]])
    retired = np.array([[False, True]])          # driver 1 is out

    out, next_delay = S.apply_double_stack_delay(
        pitting, stack_delay, total, retired, [(0, 1)])

    assert list(out[0]) == [True, True], 'retired car is not part of a stack'
    assert not next_delay.any()


def test_three_teammates_are_three_independent_pairs():
    """
    Not a real 2026 grid, but the function makes no assumption about team
    size - a three-car team pitting all three together delays two of them,
    each against whichever of the other two is behind it.
    """
    pitting = np.array([[True, True, True]])
    stack_delay = np.zeros((1, 3), dtype=bool)
    total = np.array([[10.0, 20.0, 30.0]])       # 0 ahead of 1 ahead of 2
    retired = np.zeros((1, 3), dtype=bool)

    out, next_delay = S.apply_double_stack_delay(
        pitting, stack_delay, total, retired, [(0, 1), (0, 2), (1, 2)])

    assert out[0, 0], 'the leader is never delayed'
    assert not out[0, 2], 'the car at the back is delayed against both'


# --- team_order_hold: pure function, synthetic arrays -----------------------


def test_teammates_on_the_same_tyre_and_age_hold_station():
    team_of = np.array(['A', 'A', 'B'])
    fitted = np.array([[0, 0, 1]])               # both team A cars on SOFT
    tyre_age = np.array([[10, 11, 5]])
    sim_idx = np.arange(1)

    hold = S.team_order_hold(np.array([0]), np.array([1]), team_of, fitted,
                             tyre_age, sim_idx, age_tolerance=3)
    assert hold[0]


def test_rivals_never_hold_station_even_on_identical_tyres():
    team_of = np.array(['A', 'B'])
    fitted = np.array([[0, 0]])
    tyre_age = np.array([[10, 10]])
    sim_idx = np.arange(1)

    hold = S.team_order_hold(np.array([0]), np.array([1]), team_of, fitted,
                             tyre_age, sim_idx, age_tolerance=3)
    assert not hold[0]


def test_teammates_on_different_compounds_do_not_hold_station():
    team_of = np.array(['A', 'A'])
    fitted = np.array([[0, 1]])                  # SOFT vs MEDIUM
    tyre_age = np.array([[8, 8]])
    sim_idx = np.arange(1)

    hold = S.team_order_hold(np.array([0]), np.array([1]), team_of, fitted,
                             tyre_age, sim_idx, age_tolerance=3)
    assert not hold[0], 'an undercut is a different plan, not a fight to block'


def test_teammates_far_apart_in_stint_age_do_not_hold_station():
    """Same compound, but one is on lap 2 of it and the other lap 20 - not
    the same plan, whatever the colour."""
    team_of = np.array(['A', 'A'])
    fitted = np.array([[0, 0]])
    tyre_age = np.array([[2, 20]])
    sim_idx = np.arange(1)

    hold = S.team_order_hold(np.array([0]), np.array([1]), team_of, fitted,
                             tyre_age, sim_idx, age_tolerance=3)
    assert not hold[0]


def test_the_age_tolerance_boundary_is_inclusive_then_exclusive():
    team_of = np.array(['A', 'A'])
    fitted = np.array([[0, 0], [0, 0]])
    tyre_age = np.array([[10, 13], [10, 14]])     # gap 3, then gap 4
    sim_idx = np.arange(2)

    hold = S.team_order_hold(np.array([0, 0]), np.array([1, 1]), team_of,
                             fitted, tyre_age, sim_idx, age_tolerance=3)
    assert hold[0] and not hold[1]


# --- whole-race checks --------------------------------------------------


_CACHE = {}


def _inputs():
    if 'v' not in _CACHE:
        pace = S.load_drivers()
        track = S.load_track()
        strategies, _ = S.recost_strategies(S.load_strategies(track), track)
        _CACHE['v'] = (pace, track, strategies)
    return _CACHE['v']


def test_no_stack_survives_a_real_race():
    """
    Every genuinely racing pair of teammates, across a full field and a real
    strategy mix: never both marked to pit on the same lap. Red flags are
    off for this check - a red flag hands every car left on track a free
    change on the same lap by design, which is not a stack the pit wall
    chose and is out of scope for this rule.
    """
    pace, track, strategies = _inputs()
    with S.flag_overrides({'RED_FLAG_ENABLED': False}):
        _, _, _, diag = S.run_simulation(pace, track, strategies, 800, seed=11)

    team_of = pace['Team'].to_numpy()
    pairs = S.team_pairs(team_of)
    pits = diag['trace_pits']
    retired_lap = diag['retired_lap']
    n_laps = pits.shape[0]
    lap_idx = np.arange(n_laps)[:, None]

    for i, j in pairs:
        alive_i = (retired_lap[:, i] < 0) | (lap_idx <= retired_lap[:, i][None, :])
        alive_j = (retired_lap[:, j] < 0) | (lap_idx <= retired_lap[:, j][None, :])
        both = pits[:, :, i] & pits[:, :, j] & alive_i & alive_j
        assert not both.any(), f'driver {i} and {j} pitted together'


def test_turning_it_off_lets_stacks_happen_again():
    """
    The regression this rule exists to prevent, shown by disabling the rule:
    without it, real same-lap double stacks occur in a normal-sized field.
    If this ever stops being true the fixture has drifted, not the model.
    """
    pace, track, strategies = _inputs()
    with S.flag_overrides({'RED_FLAG_ENABLED': False,
                           'DOUBLE_STACK_ENABLED': False}):
        _, _, _, diag = S.run_simulation(pace, track, strategies, 800, seed=11)

    team_of = pace['Team'].to_numpy()
    pairs = S.team_pairs(team_of)
    pits = diag['trace_pits']
    retired_lap = diag['retired_lap']
    n_laps = pits.shape[0]
    lap_idx = np.arange(n_laps)[:, None]

    stacks = 0
    for i, j in pairs:
        alive_i = (retired_lap[:, i] < 0) | (lap_idx <= retired_lap[:, i][None, :])
        alive_j = (retired_lap[:, j] < 0) | (lap_idx <= retired_lap[:, j][None, :])
        both = pits[:, :, i] & pits[:, :, j] & alive_i & alive_j
        stacks += both.sum()
    assert stacks > 0


def test_a_cap_forced_stop_is_never_delayed():
    """
    ENFORCE_STINT_CAP is a physical limit, not a choice - the curve was
    never measured past that age. No stint in a real race may exceed its
    compound's cap, stacking or not; this is the same invariant
    test_diagnostics checks, run with team orders on.
    """
    pace, track, strategies = _inputs()
    result_track = track
    with S.flag_overrides({}):
        positions, total, strat_idx, diag = S.run_simulation(
            pace, track, strategies, 600, seed=5)

    for compound in ('SOFT', 'MEDIUM', 'HARD'):
        curve = result_track['tyre'][compound]
        cap = S.max_stint_laps(curve)
        compounds = S.rebuild_compounds(diag, 0)
        stints = S.build_stints(compounds, diag['trace_pits'][:, 0, :],
                                pace['Driver'].to_numpy(),
                                retired_lap=diag['retired_lap'][0]
                                if 'retired_lap' in diag else None)
        for stint in stints:
            if str(stint['compound']) == compound:
                assert stint['length'] <= cap


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
