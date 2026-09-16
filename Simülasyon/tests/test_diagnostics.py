"""
Acceptance tests for build_stints - the pit-boundary fix and its regression.

A stint chart is read by eye, so a chart that quietly draws two stints as one
does not raise anything. It draws a bar, the bar looks like a bar, and it is
wrong only to a reader who knows the tyre could not have lasted that long. The
Spanish Grand Prix caught this directly: a driver's tyre was shown running 49
laps into a compound the curve was fitted to for 22, which is the model
looking broken on the very first read. The tyre never ran 49 laps - the car
pitted at lap 25 and fitted the same compound again, and the chart had no
boundary for a pit stop that did not change colour.

    python -m Simülasyon.tests.test_diagnostics
"""

import numpy as np

from Simülasyon.diagnostics import build_stints


def _race(compound_plan, pit_laps, n_laps=40, n_drivers=1):
    """
    A synthetic single-car race.

    compound_plan: list of (compound, length) run in order.
    pit_laps: the human lap numbers (1-based) on which the car pits - must
    match the boundaries compound_plan implies, since a real race never
    changes tyre without a stop.
    """
    compounds = np.empty((n_laps, n_drivers), dtype=object)
    pits = np.zeros((n_laps, n_drivers), dtype=bool)

    lap = 0
    for compound, length in compound_plan:
        compounds[lap:lap + length, 0] = compound
        lap += length
    compounds[lap:, 0] = compound_plan[-1][0] if lap < n_laps else compounds[lap - 1, 0]

    for human_lap in pit_laps:
        pits[human_lap - 1, 0] = True

    return compounds, pits


def test_a_same_compound_pit_stop_produces_two_stints():
    """
    The exact Spanish Grand Prix case: MEDIUM for 25 laps, a stop, MEDIUM
    again for the rest. Before the fix this was one 49-lap bar; the compound
    never changed, so nothing ended it.
    """
    compounds, pits = _race([('MEDIUM', 25), ('MEDIUM', 24)], pit_laps=[25],
                            n_laps=49)
    stints = build_stints(compounds, pits, codes=['XXX'])

    assert len(stints) == 2, stints
    assert [s['length'] for s in stints] == [25, 24]
    assert [s['compound'] for s in stints] == ['MEDIUM', 'MEDIUM']
    assert stints[0]['end'] == 25
    assert stints[1]['start'] == 26


def test_no_stint_exceeds_what_its_compound_was_measured_for():
    """
    The regression this whole fix exists to prevent: a stint reading longer
    than the curve's own cap is a chart bug, not a strategy - the simulator
    physically refuses to run a tyre past ENFORCE_STINT_CAP, so if one shows
    up in the chart it was never a real stint, only two merged into one.
    """
    from Simülasyon import race_select as RS

    built, why = RS.activate(7)      # Spanish Grand Prix
    assert built, why
    try:
        module = built['module']
        module.N_SIMS = 600
        result = module.run()
        caps = {compound: module.max_stint_laps(result['track']['tyre'][compound])
                for compound in ('SOFT', 'MEDIUM', 'HARD')}
        for stint in result['stints']:
            compound = str(stint['compound'])
            if compound in caps:
                assert stint['length'] <= caps[compound], \
                    f'{stint["driver"]} ran {compound} for {stint["length"]} ' \
                    f'laps, cap is {caps[compound]}'
    finally:
        RS.release(built)


def test_every_stint_boundary_is_a_real_pit_lap():
    """Every stint but the last one ends where trace_pits actually says so."""
    from Simülasyon import simulate as S

    S.N_SIMS = 500
    result = S.run()
    pits = np.asarray(result['trace']['pits'])

    by_driver = {}
    for stint in result['stints']:
        by_driver.setdefault(stint['driver_idx'], []).append(stint)

    for driver, stints in by_driver.items():
        stints.sort(key=lambda s: s['start'])
        stop_laps = set((np.flatnonzero(pits[:, driver]) + 1).tolist())
        for stint in stints[:-1]:
            assert stint['end'] in stop_laps, (driver, stint)


def test_consecutive_stints_never_share_a_compound_without_a_pit_between():
    """
    Restated the other way round: if two adjacent stints on the same car are
    the same compound, there must be a pit lap exactly at the boundary -
    otherwise they are the same stint, wrongly split.
    """
    compounds, pits = _race([('SOFT', 10)], pit_laps=[], n_laps=10)
    stints = build_stints(compounds, pits, codes=['XXX'])
    assert len(stints) == 1
    assert stints[0]['length'] == 10


def test_a_compound_change_without_a_recorded_pit_still_ends_the_stint():
    """
    Belt and braces: the boundary condition is `or`, not a replacement, so a
    compound change alone still closes a stint even if pits_by_lap were ever
    wrong or incomplete for some reason.
    """
    compounds, pits = _race([('SOFT', 15), ('HARD', 10)], pit_laps=[],
                            n_laps=25)
    stints = build_stints(compounds, pits, codes=['XXX'])
    assert len(stints) == 2
    assert stints[0]['compound'] == 'SOFT'
    assert stints[1]['compound'] == 'HARD'


def test_pit_lap_is_set_for_a_same_compound_stop():
    """
    pit_lap already existed for compound changes; the fix must not leave it
    None for the case it was built to catch.
    """
    compounds, pits = _race([('HARD', 20), ('HARD', 20)], pit_laps=[20],
                            n_laps=40)
    stints = build_stints(compounds, pits, codes=['XXX'])
    assert stints[0]['pit_lap'] == 20
    assert stints[1]['pit_lap'] is None      # the race simply ends


def test_a_retirement_still_ends_the_race_even_mid_stint():
    """The retirement path is untouched by the pit-boundary change."""
    compounds, pits = _race([('SOFT', 30)], pit_laps=[], n_laps=30)
    stints = build_stints(compounds, pits, codes=['XXX'],
                          retired_lap=np.array([14]))
    assert len(stints) == 1
    assert stints[0]['end'] == 15
    assert stints[0]['retired'] is True


def test_three_stints_two_pits_both_same_compound():
    """A one-stop plan that becomes two stops because of a repeat compound."""
    compounds, pits = _race(
        [('MEDIUM', 15), ('MEDIUM', 15), ('MEDIUM', 16)],
        pit_laps=[15, 30], n_laps=46)
    stints = build_stints(compounds, pits, codes=['XXX'])
    assert [s['length'] for s in stints] == [15, 15, 16]
    assert all(s['compound'] == 'MEDIUM' for s in stints)


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
