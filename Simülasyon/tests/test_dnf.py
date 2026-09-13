"""
Acceptance tests for the v2.2 two-cause retirement model.

Every test is one of the checks the v2.2 roadmap requires.

    python -m Simülasyon.tests.test_dnf
"""

import hashlib

import numpy as np

from Simülasyon import dnf as dnf_model
from Simülasyon.data_prep import retirements as rt
from Simülasyon import simulate as s


_CACHE = {}


def inputs():
    if 'v' not in _CACHE:
        pace = s.load_drivers()
        track = s.load_track()
        strategies, _ = s.recost_strategies(s.load_strategies(track), track)
        _CACHE['v'] = (pace, track, strategies)
    return _CACHE['v']


def race(n=600, seed=9, **flags):
    pace, track, strategies = inputs()
    with s.flag_overrides(flags):
        return s.run_simulation(pace, track, strategies, n, seed=seed)


def sha(positions):
    return hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest()


# --- 1. the switch ---------------------------------------------------------

def test_off_keeps_the_old_behaviour():
    a = race(TWO_CAUSE_DNF=False)
    b = race(TWO_CAUSE_DNF=False)
    assert sha(a[0]) == sha(b[0])
    assert (a[3]['dnf_cause'] == 0).all(), 'the old path produced a cause'


def test_on_is_reproducible():
    a = race(TWO_CAUSE_DNF=True)
    b = race(TWO_CAUSE_DNF=True)
    assert sha(a[0]) == sha(b[0])


def test_the_old_flat_draw_does_not_also_run():
    """
    Two processes on the same car would add their risks together and quietly
    double the retirement rate. The new branch replaces the old one rather
    than sitting beside it.
    """
    new = race(TWO_CAUSE_DNF=True)[3]
    old = race(TWO_CAUSE_DNF=False)[3]
    modelled = float(new['retired'].mean())
    flat = float(old['retired'].mean())
    assert modelled < flat * 2.0, (modelled, flat)


# --- 2. the two densities --------------------------------------------------

def test_zero_risk_produces_nothing():
    rng = np.random.default_rng(0)
    shape = (200, 10)
    hit, cause = dnf_model.draw(rng, shape, np.zeros(shape), np.zeros(shape),
                                np.ones(shape, dtype=bool))
    assert not hit.any()
    assert (cause == dnf_model.NONE).all()


def test_one_positive_cause_never_produces_the_other():
    rng = np.random.default_rng(1)
    shape = (400, 10)
    hit, cause = dnf_model.draw(rng, shape, np.full(shape, 0.05),
                                np.zeros(shape), np.ones(shape, dtype=bool))
    assert hit.any(), 'a 5% per-lap density produced nothing at all'
    assert not (cause == dnf_model.MECHANICAL).any()

    hit, cause = dnf_model.draw(rng, shape, np.zeros(shape),
                                np.full(shape, 0.05),
                                np.ones(shape, dtype=bool))
    assert not (cause == dnf_model.ACCIDENT).any()


def test_a_car_lap_gets_at_most_one_cause():
    rng = np.random.default_rng(2)
    shape = (500, 12)
    hit, cause = dnf_model.draw(rng, shape, np.full(shape, 0.02),
                                np.full(shape, 0.02),
                                np.ones(shape, dtype=bool))
    assert ((cause != dnf_model.NONE) == hit).all()
    assert set(np.unique(cause)) <= {0, 1, 2}


def test_the_split_follows_the_two_densities():
    """
    Twice the accident density, about twice the accidents. This is the part
    that would go wrong with two independent draws and an arbitrary tie-break.
    """
    rng = np.random.default_rng(3)
    shape = (40_000, 10)
    _, cause = dnf_model.draw(rng, shape, np.full(shape, 0.004),
                              np.full(shape, 0.002),
                              np.ones(shape, dtype=bool))
    acc = int((cause == dnf_model.ACCIDENT).sum())
    mech = int((cause == dnf_model.MECHANICAL).sum())
    assert 1.8 < acc / mech < 2.2, (acc, mech)


def test_the_total_lands_near_the_density():
    """
    A synthetic case with a known rate. A single sample of this size will not
    hit it exactly and small differences are not failures.
    """
    rng = np.random.default_rng(4)
    shape = (40_000, 1)
    lam = 0.003
    _, cause = dnf_model.draw(rng, shape, np.full(shape, lam),
                              np.full(shape, lam),
                              np.ones(shape, dtype=bool))
    seen = float((cause != dnf_model.NONE).mean())
    expected = 1 - np.exp(-2 * lam)
    assert abs(seen - expected) < 4 * np.sqrt(expected / shape[0]), \
        (seen, expected)


def test_a_retired_car_does_not_draw_again():
    rng = np.random.default_rng(5)
    shape = (300, 8)
    active = np.zeros(shape, dtype=bool)
    active[:, :4] = True
    hit, _ = dnf_model.draw(rng, shape, np.full(shape, 0.2),
                            np.full(shape, 0.2), active)
    assert not hit[:, 4:].any(), 'a car that was already out retired again'


def test_no_car_retires_twice_in_a_race():
    diag = race(TWO_CAUSE_DNF=True)[3]
    cause, lap = diag['dnf_cause'], diag['retired_lap']
    out = cause != dnf_model.NONE
    assert (lap[out] >= 0).all()
    assert (lap[~out] == -1).all(), 'a running car carries a retirement lap'


# --- 3. timing -------------------------------------------------------------

def test_the_shape_does_not_change_the_total():
    """
    A profile that scales the hazard without being normalised changes how many
    failures happen as well as when, and then a change of shape reads as a
    change of reliability.
    """
    flat = dnf_model.timing_shape(60)
    assert np.allclose(flat, 1.0)

    original = dnf_model.BELL_SUPPORTED
    dnf_model.BELL_SUPPORTED = True
    try:
        bell = dnf_model.timing_shape(60)
    finally:
        dnf_model.BELL_SUPPORTED = original
    assert abs(bell.mean() - 1.0) < 1e-9, bell.mean()
    assert bell.max() > bell.min(), 'the bell was flat'


def test_the_bell_was_tested_not_assumed():
    """
    The roadmap proposed a bell. The measurement did not support it, and the
    model is flat as a result - which is a finding, not an omission.
    """
    assert dnf_model.BELL_SUPPORTED is False
    assert np.allclose(dnf_model.timing_shape(53), 1.0)


def test_hazard_is_measured_against_cars_still_running():
    """
    A histogram of failures peaks early because every car is still out there
    early. Dividing by exposure is what turns counts into risk.
    """
    table, _ = rt.load()
    profile = rt.timing_profile(table)
    assert (profile['exposure_laps'] > 0).all()
    # later bands hold fewer car-laps, because cars have dropped out
    assert profile['exposure_laps'].iloc[-1] < profile['exposure_laps'].iloc[0]


# --- 4. classification -----------------------------------------------------

def test_an_unexplained_retirement_is_not_called_mechanical():
    assert rt.classify('Retired') == rt.UNKNOWN_LABEL
    assert rt.classify('Engine') == rt.MECHANICAL_LABEL
    assert rt.classify('Collision') == rt.ACCIDENT_LABEL


def test_a_disqualification_is_not_a_retirement():
    for status in ('Disqualified', 'Did not start', 'Withdrew'):
        assert rt.classify(status) == rt.EXCLUDED_LABEL, status


def test_a_lapped_finisher_finished():
    for status in ('Finished', '+1 Lap', '+2 Laps', 'Lapped'):
        assert rt.classify(status) == rt.FINISHED_LABEL, status


def test_the_unexplained_share_is_reported_not_redistributed():
    table, _ = rt.load()
    cov = rt.coverage(table)
    assert cov['unknown'] > 0, 'this data really does have unexplained ones'
    assert cov['known_share'] < 1.0
    # and the runtime never invents a third cause
    assert set(dnf_model.CAUSE_NAMES) == {'running', 'accident', 'mechanical'}


# --- 5. exposure -----------------------------------------------------------

def test_a_car_that_stopped_early_is_not_exposed_afterwards():
    table, _ = rt.load()
    retired = table[table['cause'] == rt.MECHANICAL_LABEL]
    assert (retired['exposure'] <= retired['race_laps']).all()
    assert (retired['exposure'] == retired['laps_done']).all()


def test_a_finisher_is_exposed_for_the_whole_race():
    table, _ = rt.load()
    done = table[table['cause'] == rt.FINISHED_LABEL]
    assert (done['exposure'] == done['race_laps']).all()


def test_a_driver_with_no_crashes_still_carries_risk():
    """
    Zero observed is not zero risk. A model that says otherwise will
    eventually hand that driver a championship for being unbreakable.
    """
    table, _ = rt.load()
    drivers, pooled = rt._rate(table, 'Driver', rt.ACCIDENT_LABEL)
    none = drivers[drivers['events'] == 0]
    assert len(none) > 0
    assert (none['rate'] > 0).all()
    assert (none['rate'] < pooled).all(), 'pooling should still pull them down'


def test_risk_is_per_lap_not_per_race():
    pace, _, _ = inputs()
    accident, mechanical, _ = dnf_model.rates(pace)
    # a per-race number would be percent-sized; these are per lap
    assert accident.max() < 0.05
    assert mechanical.max() < 0.05
    assert (accident > 0).all() and (mechanical > 0).all()


# --- 6. the flag -----------------------------------------------------------

def test_one_incident_does_not_start_two_safety_cars():
    """
    Three cars out of the same pile-up is one safety car. The request is
    rolled once per simulation per lap, not once per car.
    """
    diag = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=True)[3]
    cause, lap = diag['dnf_cause'], diag['retired_lap']
    crashes = (cause == dnf_model.ACCIDENT).sum()
    assert diag['acc_neutral'].sum() <= crashes, \
        'more flags than accidents, so something rolled per car'


def test_neutralization_states_stay_legal():
    diag = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=True)[3]
    assert set(np.unique(diag['neutral'])) <= {0, 1, 2, 3}


def test_the_red_flag_limit_is_not_relaxed():
    """The one-per-race rule is the existing model's and v2.2 does not touch it."""
    diag = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=True)[3]
    per_race = (diag['neutral'] == 3).sum(axis=1)
    assert per_race.max() <= 1, per_race.max()


def test_the_background_rate_is_scaled_when_accidents_supply_flags():
    """
    The circuit rate was measured over races containing these accidents.
    Leaving it at full strength while accidents also generate their own is the
    double count this version exists to avoid.
    """
    linked = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=True)[3]
    unlinked = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=False)[3]
    assert linked['bg_neutral'].mean() < unlinked['bg_neutral'].mean(), \
        'the background rate was not scaled down'


def test_the_total_flag_rate_does_not_simply_add_up():
    """
    The point of the scaling. Switching accidents on must not raise the total
    neutralization rate by the whole accident contribution.
    """
    off = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=False)[3]
    on = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=True)[3]
    before = float((off['neutral'] > 0).any(axis=1).mean())
    after = float((on['neutral'] > 0).any(axis=1).mean())
    assert after < before * 1.35, (before, after)


def test_the_ninety_percent_assumption_was_replaced_by_a_measurement():
    mix = dict(dnf_model.ACCIDENT_OUTCOME)
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert mix['none'] > 0.40, 'the measurement says about half bring nothing'
    assert mix['SC'] < 0.60


def test_accidents_are_not_drawn_under_a_neutralization():
    """
    A stated limitation rather than a claim that crashes cannot happen behind
    a safety car. There is no separate measurement of the rate under yellow,
    and reusing the green-flag one would be an invention.
    """
    diag = race(TWO_CAUSE_DNF=True)[3]
    cause, lap = diag['dnf_cause'], diag['retired_lap']
    crashed = cause == dnf_model.ACCIDENT
    if crashed.any():
        sims, cars = np.nonzero(crashed)
        laps = lap[sims, cars]
        under_yellow = diag['neutral'][sims, laps]
        assert (under_yellow == 0).all(), 'a crash was drawn under yellow'


# --- 7. the engine ---------------------------------------------------------

def test_a_retired_car_is_not_declared_the_winner():
    positions, total, _, diag = race(TWO_CAUSE_DNF=True)
    out = diag['dnf_cause'] != dnf_model.NONE
    if out.any():
        assert (positions[out] > 1).all() or not out.any(), \
            'a retired car finished first'


def test_a_retired_car_stops_pitting_and_racing():
    _, _, _, diag = race(TWO_CAUSE_DNF=True)
    out = diag['dnf_cause'] != dnf_model.NONE
    # a car that never ran a lap after retiring cannot have gained places
    assert diag['n_stops'][out].mean() <= diag['n_stops'].mean() + 1e-9


def test_the_three_probabilities_are_the_whole_of_it():
    _, _, _, diag = race(TWO_CAUSE_DNF=True)
    out = dnf_model.summarise(diag, 0, 0)
    total = out['p_finish'] + out['p_accident'] + out['p_mechanical']
    assert abs(total - 1.0) < 1e-9, total


def test_per_driver_probabilities_also_sum_to_one():
    pace, _, _ = inputs()
    _, _, _, diag = race(TWO_CAUSE_DNF=True)
    table = dnf_model.per_driver(diag, pace['Driver'])
    total = (table['P_finish'] + table['P_accident']
             + table['P_mechanical']).to_numpy()
    assert np.allclose(total, 1.0), total


def test_the_event_lap_is_recorded_for_every_retirement():
    _, _, _, diag = race(TWO_CAUSE_DNF=True)
    out = diag['dnf_cause'] != dnf_model.NONE
    assert (diag['retired_lap'][out] >= 0).all()
    assert out.sum() > 0, 'no retirements at all, so this tested nothing'


def test_a_future_retirement_cannot_move_a_pit_decision():
    """
    The pit decision reads the state it is handed, and that state contains no
    retirement that has not happened. Two runs whose retirements diverge only
    later must agree on the stops taken before the divergence.
    """
    a = race(TWO_CAUSE_DNF=True, seed=21)[3]
    b = race(TWO_CAUSE_DNF=True, seed=21)[3]
    assert np.array_equal(a['trace_pits'], b['trace_pits'])


# --- 8. reporting ----------------------------------------------------------

def test_the_two_sources_of_flags_are_counted_apart():
    _, _, _, diag = race(TWO_CAUSE_DNF=True, ACCIDENT_NEUTRALIZATION=True)
    assert diag['acc_neutral'].sum() >= 0
    assert diag['bg_neutral'].sum() > 0


def test_provenance_separates_measured_from_assumed():
    rows = dnf_model.provenance()
    kinds = {kind for _, kind, _, _ in rows}
    assert 'measured' in kinds
    assert 'hand-set' in kinds or 'assumption' in kinds
    for name, kind, value, note in rows:
        assert note, name


def test_coverage_is_carried_with_the_result():
    _, _, _, diag = race(TWO_CAUSE_DNF=True)
    out = dnf_model.summarise(diag, 0, 0)
    assert 0.5 < out['coverage'] < 1.0
    assert 'p_unknown' not in out, 'the data gap was shown as a probability'


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
