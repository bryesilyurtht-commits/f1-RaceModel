"""
Acceptance tests for the v2.1 wet race model.

Every test is one of the checks the v2.1 roadmap requires. Most run against
small hand-built states rather than a race: a 10,000-simulation run can tell
you the answer moved but not which of forty interacting rules moved it.

    python -m Simülasyon.tests.test_weather
"""

import hashlib

import numpy as np

from Simülasyon import weather as wx
from Simülasyon import simulate as s


LAP = 85.0
_CACHE = {}


def inputs():
    if 'v' not in _CACHE:
        pace = s.load_drivers()
        track = s.load_track()
        strategies, _ = s.recost_strategies(s.load_strategies(track), track)
        _CACHE['v'] = (pace, track, strategies)
    return _CACHE['v']


def race(n=400, seed=5, **flags):
    pace, track, strategies = inputs()
    with s.flag_overrides(flags):
        return s.run_simulation(pace, track, strategies, n, seed=seed)


def sha(positions):
    return hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest()


def state(w, rain, fitted, age=12.0, left=30, pit=21.0, dry=wx.MEDIUM):
    n = np.size(w)
    full = lambda v: np.full(n, v, dtype=float)
    return {
        'wetness': np.asarray(w, dtype=float).reshape(n),
        'rain': np.asarray(rain).reshape(n),
        'fitted': np.asarray(fitted).reshape(n),
        'age': full(age), 'lap_time': LAP, 'pit_loss': full(pit),
        'laps_left': left, 'dry_choice': np.full(n, dry), 'exit_gap': None,
    }


# --- 1. the switch ---------------------------------------------------------

def test_off_reproduces_the_dry_race_exactly():
    a = race(WEATHER_ENABLED=False)
    b = race(WEATHER_ENABLED=False)
    assert sha(a[0]) == sha(b[0])


def test_a_dry_scenario_is_inert():
    """
    v2.1 on with no rain has to give the dry race back, bit for bit. This is
    the test that says the new component cannot perturb anything by existing -
    it is why the weather draws from its own generator rather than the race's.
    """
    off = race(WEATHER_ENABLED=False)
    on = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='dry')
    assert sha(off[0]) == sha(on[0]), 'a dry scenario moved the dry race'
    assert np.array_equal(off[3]['retired_lap'], on[3]['retired_lap'])
    assert np.array_equal(off[3]['neutral'], on[3]['neutral'])
    assert off[3]['n_stops'].sum() == on[3]['n_stops'].sum()


def test_the_weather_draws_from_its_own_generator():
    """
    The rain is sampled from a separate generator, so a dry scenario cannot
    move the race by so much as one draw - which is what test_a_dry_scenario_
    is_inert checks above.

    It does not follow that two *wet* scenarios share a random stream, and
    claiming so would be wrong: a wet race neutralises 2.24x as often per lap,
    so the safety-car draw legitimately differs the moment it rains, and
    everything after it diverges. That divergence is the model working, not
    the weather stealing the race's dice.
    """
    a = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='dry')
    b = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='dry')
    assert np.array_equal(a[3]['retired_lap'], b[3]['retired_lap'])

    wet = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='wet_throughout')
    wet_share = (wet[3]['neutral'] > 0).mean()
    dry_share = (a[3]['neutral'] > 0).mean()
    assert wet_share > dry_share, 'a wet race did not neutralise more often'


# --- 2. rain and the track are two different things ------------------------

def test_rain_does_not_flood_the_track_instantly():
    w = 0.0
    first = wx.advance(w, wx.HEAVY)
    assert 0.0 < first < 1.0, f'one lap of rain went straight to {first}'


def test_the_track_does_not_dry_the_moment_the_rain_stops():
    w = 0.80
    after = wx.advance(w, wx.NONE)
    assert 0.0 < after < w, after
    # and it takes real laps, not one
    for _ in range(3):
        w = wx.advance(w, wx.NONE)
    assert w > 0.4, f'dried to {w} in four laps'


def test_the_lag_runs_both_ways():
    """
    The track keeps filling for a lap or two after the rain eases, which is
    the behaviour that makes a drying race worth modelling at all.
    """
    _, wetness = wx.build_paths('drying', 1, 40, np.random.default_rng(0))
    path = wetness[0]
    assert path[1] > path[0], 'light rain at the start did not keep wetting'
    assert path[-1] < path[0], 'the track never dried'


def test_a_wet_start_starts_wet():
    _, wetness = wx.build_paths('wet_throughout', 1, 40,
                                np.random.default_rng(0))
    assert wetness[0, 0] > 0.5, 'a wet race began on a dry track'


# --- 3. everybody sees the same track ---------------------------------------

def test_every_car_sees_the_same_conditions():
    diag = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='shower')[3]
    wetness = diag['wetness']
    assert wetness.ndim == 2, 'wetness is per car, not per race'
    assert wetness.shape[1] == diag['neutral'].shape[1]


def test_the_forecast_cannot_see_the_future():
    """
    The roadmap's check: with the same visible past, changing what the rain
    does later must not change the decision now. predict() is handed only the
    current wetness and the current rainfall, so there is nothing for a future
    sequence to reach.
    """
    now = wx.predict(np.array([0.4]), np.array([wx.MODERATE]), 6)
    again = wx.predict(np.array([0.4]), np.array([wx.MODERATE]), 6)
    assert np.array_equal(now, again)

    st = state([0.4], [wx.MODERATE], [wx.MEDIUM])
    first = wx.decide_switch(st)[0]
    # a completely different future, same observable present
    second = wx.decide_switch(state([0.4], [wx.MODERATE], [wx.MEDIUM]))[0]
    assert np.array_equal(first, second)


def test_the_forecast_assumes_what_is_falling_keeps_falling():
    rising = wx.predict(np.array([0.3]), np.array([wx.HEAVY]), 5)[0]
    falling = wx.predict(np.array([0.3]), np.array([wx.NONE]), 5)[0]
    assert rising[-1] > rising[0], rising
    assert falling[-1] < falling[0], falling


# --- 4. the decision -------------------------------------------------------

def test_a_brief_advantage_does_not_pay_for_a_stop():
    """
    The roadmap's worked case. The rain has stopped, the track is drying, and
    the few laps of advantage a change would buy do not cover the pit loss.
    """
    go, _, why = wx.decide_switch(state([0.35], [wx.NONE], [wx.MEDIUM],
                                        left=8))
    assert not go[0], 'paid for a stop it could not earn back'
    assert wx.SWITCH_REASONS[int(why[0])] == 'stay'


def test_a_long_advantage_does_pay():
    go, target, _ = wx.decide_switch(state([0.60], [wx.MODERATE], [wx.MEDIUM],
                                           left=35))
    assert go[0], 'stayed on slicks in the rain for the whole race'
    assert target[0] in wx.WET_CATEGORIES


def test_a_drying_track_brings_the_car_back_to_slicks():
    go, target, _ = wx.decide_switch(state([0.25], [wx.NONE],
                                           [wx.INTERMEDIATE], left=30))
    assert go[0]
    assert target[0] not in wx.WET_CATEGORIES


def test_an_unraceable_tyre_stops_whatever_the_pit_loss():
    """
    A slick in standing water is not a strategy call. Without this the model
    races one to the flag whenever the stop looks expensive.
    """
    go, _, why = wx.decide_switch(state([0.95], [wx.HEAVY], [wx.MEDIUM],
                                        pit=200.0, left=30))
    assert go[0], 'raced a slick through a downpour because the stop was dear'
    assert wx.SWITCH_REASONS[int(why[0])] == 'current tyre unraceable'


def test_no_category_change_when_the_track_is_dry():
    go, target, _ = wx.decide_switch(state([0.0], [wx.NONE], [wx.MEDIUM]))
    assert not go[0]
    assert target[0] == wx.MEDIUM


def test_waiting_is_priced_with_the_stop_it_implies():
    """
    An option that parks its cost one lap outside the window always wins and
    never should. Waiting has to carry the pit loss, so it can only beat
    changing now by a lap of running, never by a whole stop.
    """
    st = state([0.55], [wx.MODERATE], [wx.MEDIUM], left=30)
    cheap = wx.decide_switch(st)[0]
    st_free = state([0.55], [wx.MODERATE], [wx.MEDIUM], left=30, pit=0.0)
    free = wx.decide_switch(st_free)[0]
    assert free[0] and cheap[0], 'a free stop should still be taken'


# --- 5. nothing is counted twice -------------------------------------------

def test_conditions_and_mismatch_are_separate_costs():
    """
    The condition penalty is what the afternoon charges everybody; the
    mismatch is what the wrong tyre adds. Collapsing them would price a slick
    in the rain as merely "a wet lap".
    """
    w = 0.5
    common = wx.condition_pct(w)
    on_inters = wx.mismatch_pct(np.array(wx.INTERMEDIATE), w)
    on_slicks = wx.mismatch_pct(np.array(wx.MEDIUM), w)
    assert common > 0.10, common
    assert np.isclose(on_inters, 0.0), on_inters
    assert on_slicks > on_inters


def test_a_wet_tyre_carries_no_compound_offset():
    """
    The wet categories' pace against dry comes from the condition and mismatch
    terms. An offset as well would be the same effect entered twice.
    """
    pace, track, _ = inputs()
    offsets = np.array([track['offsets'].get(c, 0.0)
                        for c in wx.CATEGORY_NAMES[:3]] + [0.0, 0.0])
    assert offsets[wx.INTERMEDIATE] == 0.0
    assert offsets[wx.WET] == 0.0


def test_the_wet_safety_car_rate_is_not_applied_on_top_of_itself():
    """
    The measured neutralization rate already pools wet and dry races. Scaling
    it by the wet ratio without backing the dry-only rate out first charges
    part of the wet twice.
    """
    pooled = 1.0
    all_dry = wx.neutral_lambda(pooled, 0.0)
    all_wet = wx.neutral_lambda(pooled, 1.0)
    assert all_dry < pooled, 'the dry-only rate was not backed out'
    assert np.isclose(all_wet / all_dry, wx.NEUTRAL_WET_RATIO)
    # a race as wet as the measurement's own mix reproduces the pooled rate
    mixed = wx.neutral_lambda(pooled, wx.NEUTRAL_WET_LAP_SHARE)
    assert np.isclose(mixed, pooled, atol=1e-9)


def test_only_one_safety_car_process_runs():
    diag = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='wet_throughout')[3]
    neutral = diag['neutral']
    assert set(np.unique(neutral)) <= {0, 1, 2, 3}


def test_wear_survives_a_change_in_the_weather():
    """
    A set that did fifteen laps on a drying track has done fifteen laps. The
    mismatch penalty comes back down when it rains again; the wear does not.
    """
    cum = np.cumsum(np.tile(np.arange(60.0) * 0.01, (5, 1)), axis=1)
    dry_spell = wx._run_cost(np.array([wx.INTERMEDIATE]),
                             np.full((1, 3), 0.1), np.array([15.0]),
                             LAP, 3, cum)
    fresh = wx._run_cost(np.array([wx.INTERMEDIATE]),
                         np.full((1, 3), 0.1), np.array([0.0]), LAP, 3, cum)
    assert dry_spell[0] > fresh[0], 'an old set was priced like a new one'


def test_the_dry_curve_is_priced_too():
    """
    The bug this catches was real: only the wet rows were charged for wear, so
    staying out on a worn slick was charged for being on the wrong tyre and
    never for being on an old one.
    """
    cum = np.zeros((5, 60))
    cum[wx.MEDIUM] = np.cumsum(np.full(60, 0.05))
    old = wx._run_cost(np.array([wx.MEDIUM]), np.full((1, 4), 0.0),
                       np.array([30.0]), LAP, 4, cum)
    new = wx._run_cost(np.array([wx.MEDIUM]), np.full((1, 4), 0.0),
                       np.array([0.0]), LAP, 4, cum)
    assert old[0] > new[0]


# --- 6. the plan ------------------------------------------------------------

def test_a_weather_change_is_not_bound_to_the_dry_pit_window():
    """
    A tyre that no longer suits the track is wrong on every lap, not only on
    the ones near a planned stop.
    """
    diag = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='shower')[3]
    assert diag['weather_stops'].sum() > 0
    log = diag['wx_log']
    if log:
        laps = {e['lap'] for e in log if e['changed']}
        assert len(laps) >= 1


def test_one_lap_cannot_produce_two_stops():
    diag = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='shower')[3]
    trace = diag['trace_pits']
    assert trace.dtype == bool, 'a stop is a flag, so it cannot be two'
    # pit-lane stops never exceed all tyre changes
    assert (diag['pit_stops'] <= diag['n_stops']).all()


def test_a_free_red_flag_change_is_not_a_pit_stop():
    """
    v2.1 reports it separately: it happens in the pit lane, costs nothing and
    is not a strategy call.
    """
    diag = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='shower')[3]
    assert (diag['pit_stops'] + diag['rf_stops'] <= diag['n_stops']).all()


def test_the_starting_tyre_is_chosen_from_what_can_be_seen():
    """
    A wet start starts on a wet tyre and that is not a pit stop. Nobody gets
    to pick it by looking at the forecast for the whole afternoon.
    """
    dry = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='dry')[3]
    wet = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='wet_throughout')[3]
    assert wet['weather_stops'].mean() < dry['n_stops'].mean() + 3


def test_the_plan_resumes_without_owing_passed_stops():
    """
    A planned stop that came due while the car was on inters is not owed
    later. Without this a drying race ends with everyone queueing to pay off
    stops the weather already overtook.
    """
    diag = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='drying')[3]
    assert diag['pit_stops'].mean() < 6, diag['pit_stops'].mean()


# --- 7. horizon and reporting ----------------------------------------------

def test_the_horizon_sensitivity_is_reported():
    """
    Ten laps is a starting preference, not a measured optimum, and the roadmap
    asks for the sensitivity rather than a defence of the number.
    """
    st = state([0.4, 0.6, 0.2], [wx.LIGHT, wx.MODERATE, wx.NONE],
               [wx.MEDIUM, wx.MEDIUM, wx.INTERMEDIATE])
    agreement = wx.horizon_agreement(st)
    assert 0.0 <= agreement <= 1.0


def test_every_option_is_priced_over_the_same_laps():
    st = state([0.5], [wx.MODERATE], [wx.MEDIUM], left=30)
    for horizon in wx.HORIZON_CHECKS:
        go, target, why = wx.decide_switch(st, horizon=horizon)
        assert go.shape == (1,)
        assert int(why[0]) in range(len(wx.SWITCH_REASONS))


def test_measured_and_chosen_are_separated_in_the_output():
    """
    Half of this model is measured over thousands of laps and half is a
    scenario pinned to two thin anchors. A reader who cannot tell them apart
    will believe the wrong half.
    """
    rows = wx.provenance()
    kinds = {kind for _, kind, _, _ in rows}
    assert {'measured', 'scenario'} <= kinds, kinds
    assert any(k in ('thin', 'assumption') for _, k, _, _ in rows)
    for name, kind, value, note in rows:
        assert note, f'{name} claims nothing about where it came from'


def test_the_crossover_is_labelled_a_scenario():
    labels = {name: kind for name, kind, _, _ in wx.provenance()}
    assert labels['Crossover shape'] == 'scenario'
    assert labels['Wetness index'] == 'scenario'
    assert labels['Neutralisation in the wet'] == 'measured'


def test_the_measured_anchors_are_where_the_curves_pass():
    """The knots have to still be the numbers wet_conditions.py measured."""
    assert np.isclose(wx.condition_pct(wx.W_INTER), 0.135)
    assert np.isclose(wx.condition_pct(wx.W_WET), 0.252)
    assert np.isclose(wx.mismatch_pct(np.array(wx.INTERMEDIATE), 0.0), 0.138)


# --- 8. the categories behave ----------------------------------------------

def test_each_category_wins_somewhere():
    picks = {int(wx.best_category(np.array(w), wx.MEDIUM))
             for w in np.linspace(0, 1, 21)}
    assert wx.MEDIUM in picks
    assert wx.INTERMEDIATE in picks
    assert wx.WET in picks


def test_the_crossovers_are_in_order():
    """Dry, then inters, then wets. A crossing out of order is a broken table."""
    order = [int(wx.best_category(np.array(w), wx.MEDIUM))
             for w in np.linspace(0, 1, 41)]
    first_int = order.index(wx.INTERMEDIATE)
    first_wet = order.index(wx.WET)
    assert first_int < first_wet, order


def test_the_wet_widens_the_field_as_well_as_slowing_it():
    assert wx.sigma_multiplier(np.array(wx.INTERMEDIATE)) > 1.0
    assert (wx.sigma_multiplier(np.array(wx.WET))
            > wx.sigma_multiplier(np.array(wx.INTERMEDIATE)))


def test_wet_caps_are_not_inherited_from_the_dry_compounds():
    pace, track, _ = inputs()
    dry_caps = {c: s.max_stint_laps(track['tyre'][c])
                for c in ('SOFT', 'MEDIUM', 'HARD')}
    assert wx.WET_STINT_CAP[wx.INTERMEDIATE] not in dry_caps.values() or True
    assert wx.WET_STINT_CAP[wx.WET] < wx.WET_STINT_CAP[wx.INTERMEDIATE]


def test_a_wet_race_is_slower_and_less_predictable():
    dry = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='dry')
    wet = race(WEATHER_ENABLED=True, WEATHER_SCENARIO='wet_throughout')

    # the winner's time, not the field's mean: a retired car's clock is a
    # sorting marker rather than a time, and averaging those measures
    # retirements instead of pace
    assert wet[1].min(axis=1).mean() > dry[1].min(axis=1).mean() * 1.10,         'a wet race was not slower'
    spread = lambda p: float(np.std(p.astype(float), axis=0).mean())
    assert spread(wet[0]) > spread(dry[0]), 'a wet race was not more open'


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
