"""
Acceptance tests for the v2.7 start.

The brief's checks, and the ones that would otherwise pass silently. A start
model that is subtly wrong does not crash - it returns an order, the order
looks like a grid, and the error only shows up as a bias nobody is looking for.
So these test the things that must be true of any valid start: everyone is in
it once, the times agree with the order, and which car the code happens to
process first does not decide who gets ahead.

    python -m Simülasyon.tests.test_starts
"""

import numpy as np
import pandas as pd

from Simülasyon import simulate as S
from Simülasyon import starts as ST


def _track(gap=0.5, sigma=1.9):
    return {'grid_gap': gap, 'start_sigma': sigma}


def _draw(n_sims=4000, n_drivers=20, seed=5, lock=False, track=None):
    rng = np.random.default_rng(seed)
    shape = (n_sims, n_drivers)
    grid = np.arange(1, n_drivers + 1, dtype=float)
    gaps = S.build_start_gaps(np.broadcast_to(grid, shape), n_drivers,
                              track or _track(), rng, shape, lock_order=lock)
    order = np.argsort(np.argsort(gaps, axis=1), axis=1) + 1
    return gaps, order


# --- the brief's structural checks ------------------------------------------

def test_every_car_appears_exactly_once():
    """A sort cannot promote two cars into the same slot. This proves it."""
    _, order = _draw()
    expected = np.arange(1, order.shape[1] + 1)
    for row in order[:200]:
        assert np.array_equal(np.sort(row), expected)


def test_no_two_cars_share_a_position():
    _, order = _draw()
    assert (np.diff(np.sort(order, axis=1), axis=1) == 1).all()


def test_the_order_agrees_with_the_times():
    """
    Position and cumulative time must not disagree. A car shown ahead while
    carrying the larger clock is the bug that makes every later gap nonsense.
    """
    gaps, order = _draw()
    for i in range(200):
        by_time = np.argsort(gaps[i], kind='stable')
        by_position = np.argsort(order[i], kind='stable')
        assert np.array_equal(by_time, by_position)


def test_no_gap_is_negative():
    """Everything is measured from the leader, so the leader is zero."""
    gaps, _ = _draw()
    assert (gaps >= 0).all()
    assert np.isclose(gaps.min(axis=1), 0).all()


def test_processing_order_gives_nobody_an_advantage():
    """
    The same grid, with the cars handed to the code in a different order, must
    produce the same distribution. If position in the array mattered, a car
    would gain places by being early in it.
    """
    n = 20
    forward = _draw(n_drivers=n, seed=11)[1]
    rng = np.random.default_rng(11)
    shuffle = rng.permutation(n)
    shape = (4000, n)
    grid = np.arange(1, n + 1, dtype=float)[shuffle]
    gaps = S.build_start_gaps(np.broadcast_to(grid, shape), n, _track(),
                              np.random.default_rng(11), shape, lock_order=False)
    shuffled = np.argsort(np.argsort(gaps, axis=1), axis=1) + 1

    # Undo the shuffle and compare per-slot mean change, not per-draw results:
    # the two use the same stream for different cars, so only the distribution
    # can agree.
    back = np.empty_like(shuffled)
    back[:, shuffle] = shuffled
    for slot in range(n):
        a = (slot + 1) - forward[:, slot]
        b = (slot + 1) - back[:, slot]
        assert abs(a.mean() - b.mean()) < 0.25, (slot, a.mean(), b.mean())


# --- the calibration --------------------------------------------------------

def test_the_start_actually_changes_the_order():
    """The point of v2.7. Locked, this is exactly zero."""
    _, order = _draw(lock=False)
    grid = np.arange(1, order.shape[1] + 1)
    assert (order != grid).any(axis=1).mean() > 0.9


def test_locking_restores_the_old_behaviour_exactly():
    _, order = _draw(lock=True)
    grid = np.arange(1, order.shape[1] + 1)
    assert (order == grid).all()


def test_the_front_of_the_grid_is_calmer_than_the_back():
    """
    The measured fact the chaos ramp exists for: pole holds 78% against 25% at
    slot ten.
    """
    _, order = _draw()
    grid = np.arange(1, order.shape[1] + 1)
    change = grid - order
    assert change[:, 0].std() < change[:, 9].std()
    assert (change[:, 0] == 0).mean() > (change[:, 9] == 0).mean()


def test_the_spread_matches_what_was_measured():
    """
    Calibration, against the table starts.py wrote. Not a tight fit - it is two
    parameters against twenty slots - but the model must land in the right
    region rather than merely in the right direction.
    """
    profile = pd.read_csv(ST.PROFILE_OUT)
    profile = profile[~profile['thin']]
    _, order = _draw(n_sims=20000, n_drivers=20, seed=3)
    grid = np.arange(1, 21)
    change = grid - order

    errors = []
    for _, row in profile.iterrows():
        slot = int(row['slot'])
        if slot > 20:
            continue
        errors.append(abs(change[:, slot - 1].std() - row['sd_change']))
    assert np.mean(errors) < 0.55, np.mean(errors)


def test_pole_retention_is_in_the_measured_region():
    """Measured 78%. The model is allowed to miss, but not by a lot."""
    _, order = _draw(n_sims=20000, seed=4)
    assert 0.55 < (order[:, 0] == 1).mean() < 0.90


def test_the_mean_change_is_near_zero_everywhere():
    """Places gained by one car are lost by another. A sort guarantees it."""
    _, order = _draw(n_sims=20000)
    grid = np.arange(1, order.shape[1] + 1)
    assert abs((grid - order).mean()) < 1e-9


# --- the measurement itself -------------------------------------------------

def test_a_car_that_pitted_on_lap_one_is_not_a_start_observation():
    frame = ST.classify(_observations())
    row = frame[frame['Driver'] == 'PIT'].iloc[0]
    assert row['category'] == 'lap-one pit or pit-lane start'


def test_a_car_that_retired_on_lap_one_is_not_a_start_observation():
    frame = ST.classify(_observations())
    row = frame[frame['Driver'] == 'OUT'].iloc[0]
    assert row['category'] == 'lap-one retirement'


def test_a_neutralised_lap_one_is_its_own_category():
    frame = ST.classify(_observations())
    row = frame[frame['Driver'] == 'SC'].iloc[0]
    assert row['category'] == 'lap one neutralised'


def test_every_row_gets_exactly_one_category():
    frame = ST.classify(_observations())
    assert frame['category'].notna().all()
    assert frame['category'].isin([
        'raced', 'lap-one retirement', 'lap-one pit or pit-lane start',
        'lap one neutralised', 'incomplete record']).all()


def test_places_inherited_from_a_retirement_are_not_counted_as_a_pass():
    """
    The trap the whole module exists for. Four cars; the one in front retires
    on lap one. The car behind is second on the road and must still be scored
    as having gained nothing, because it passed nobody.
    """
    frame = ST.classify(_observations())
    raced = ST.racing_change(frame)
    chaser = raced[raced['Driver'] == 'B']
    assert len(chaser) == 1
    assert chaser['change'].iloc[0] == 0, chaser['change'].iloc[0]


def test_the_racing_set_is_ranked_only_among_itself():
    frame = ST.classify(_observations())
    raced = ST.racing_change(frame)
    assert set(raced['grid_rank']) == set(range(1, len(raced) + 1))


def _observations():
    """
    One race: two cars racing, one retiring on lap one, one pitting, one
    neutralised - built so each category has a member that must not leak into
    the others.
    """
    rows = [
        # A leads from pole and stays there; B starts third behind the car
        # that retires, so its road position improves without a pass.
        dict(Driver='A', grid_pos=1, Position=1, last_lap=50, TrackStatus='1.0',
             PitInTime_s=np.nan, PitOutTime_s=np.nan),
        dict(Driver='OUT', grid_pos=2, Position=2, last_lap=1, TrackStatus='1.0',
             PitInTime_s=np.nan, PitOutTime_s=np.nan),
        dict(Driver='B', grid_pos=3, Position=3, last_lap=50, TrackStatus='1.0',
             PitInTime_s=np.nan, PitOutTime_s=np.nan),
        dict(Driver='PIT', grid_pos=4, Position=4, last_lap=50,
             TrackStatus='1.0', PitInTime_s=12.0, PitOutTime_s=np.nan),
        dict(Driver='SC', grid_pos=5, Position=5, last_lap=50,
             TrackStatus='124.0', PitInTime_s=np.nan, PitOutTime_s=np.nan),
    ]
    frame = pd.DataFrame(rows)
    frame['Season'], frame['RoundNumber'], frame['Race'] = 2024, 1, 'Test GP'
    return frame


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
