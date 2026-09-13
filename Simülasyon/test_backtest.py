"""
Acceptance tests for the v2.6 measuring instrument.

The checks the v2.6 brief lists, plus the ones that caught a mistake while it
was being written. A scoring function that is wrong does not fail loudly - it
returns a number, and the number goes in the report - so these test the cases
where the right answer is known in advance: a perfect prediction, a coin flip,
and a prediction that is exactly the baseline.

    python -m Simülasyon.test_backtest
"""

import numpy as np
import pandas as pd

from Simülasyon import backtest as B


# --- fixtures ---------------------------------------------------------------

def _results():
    """Two tiny seasons, enough to exercise ordering and the walk-forward cut."""
    rows = []
    for season in (2024, 2025):
        for rnd in (1, 2):
            for i, driver in enumerate(['AAA', 'BBB', 'CCC', 'DDD']):
                pos = (i + rnd) % 4 + 1
                rows.append(dict(
                    Driver=driver, Team=f'T{i // 2}', GridPosition=i + 1,
                    Position=pos, ClassifiedPosition=str(pos),
                    Status='Finished', Points=10.0 if pos <= 2 else 0.0,
                    Season=season, Race=f'R{rnd}', RoundNumber=rnd))
    frame = pd.DataFrame(rows)
    frame['finish_pos'] = pd.to_numeric(frame['ClassifiedPosition'],
                                        errors='coerce')
    frame['grid_pos'] = pd.to_numeric(frame['GridPosition'], errors='coerce')
    return frame


def _truth(positions, finished=None):
    drivers = list(positions)
    pos = [positions[d] for d in drivers]
    fin = [True] * len(drivers) if finished is None else \
        [finished[d] for d in drivers]
    return pd.DataFrame({
        'Driver': drivers, 'actual_pos': pos,
        'won': [p == 1 and f for p, f in zip(pos, fin)],
        'podium': [p <= 3 and f for p, f in zip(pos, fin)],
        'points': [p <= 10 and f for p, f in zip(pos, fin)],
        'finished': fin})


def _certain(positions):
    """A Prediction that puts all its mass on the given positions."""
    drivers = list(positions)
    n = len(drivers)
    dist = np.zeros((n, n))
    for i, d in enumerate(drivers):
        dist[i, positions[d] - 1] = 1.0
    pos = [positions[d] for d in drivers]
    return B.Prediction(drivers, pos,
                        [1.0 if p == 1 else 0.0 for p in pos],
                        [1.0 if p <= 3 else 0.0 for p in pos],
                        [1.0 if p <= 10 else 0.0 for p in pos],
                        dist=dist, label='oracle')


# --- the brief's stated checks ----------------------------------------------

def test_a_perfect_event_prediction_scores_zero_brier():
    order = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4}
    scores = B.event_brier(_certain(order), _truth(order))
    for event, value in scores.items():
        assert abs(value) < 1e-12, f'{event} was {value}, expected 0'


def test_a_coin_flip_contributes_a_quarter():
    """The brief's second worked number: 0.5 against any outcome is 0.25."""
    assert abs(B.brier([0.5, 0.5, 0.5], [1, 0, 1]) - 0.25) < 1e-12


def test_the_briefs_podium_example():
    """70% given, podium happens -> 0.09; it does not -> 0.49."""
    assert abs(B.brier([0.7], [1]) - 0.09) < 1e-12
    assert abs(B.brier([0.7], [0]) - 0.49) < 1e-12


def test_a_perfect_position_distribution_scores_zero_rps():
    order = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4}
    assert abs(B.ranked_probability_score(_certain(order), _truth(order))) < 1e-12


def test_a_confident_miss_costs_more_than_a_hedged_one():
    """The property RPS exists for, and that position error cannot see."""
    truth = _truth({'AAA': 1, 'BBB': 2})
    confident = B.Prediction(['AAA', 'BBB'], [2, 1], [0, 1], [0, 1], [1, 1],
                             dist=np.array([[0.0, 1.0], [1.0, 0.0]]))
    hedged = B.Prediction(['AAA', 'BBB'], [1.5, 1.5], [.5, .5], [1, 1], [1, 1],
                          dist=np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert B.ranked_probability_score(confident, truth) > \
           B.ranked_probability_score(hedged, truth)


def test_rps_is_on_the_same_scale_for_different_field_sizes():
    """
    Normalising by N-1 is what lets a 19-car race sit beside a 24-car one.

    Scored on the same error - a completely reversed order - across every field
    size that occurs. The archive holds 19 and 20; 2026 enters 23, and a race
    can lose cars before the flag, so the range is wider than the archive.

    The invariance is asymptotic, not exact: a reversal in a four-car field is
    proportionally a larger error and scores 0.67 against 0.53. That matters
    for a toy example and not for a Grand Prix, so the tolerance here is the
    one that applies to real fields rather than the loosest one that passes.
    """
    scores = []
    for n in (18, 19, 20, 22, 23, 24):
        drivers = [f'D{i}' for i in range(n)]
        order = {d: i + 1 for i, d in enumerate(drivers)}
        reversed_order = {d: n - i for i, d in enumerate(drivers)}
        scores.append(B.ranked_probability_score(_certain(reversed_order),
                                                 _truth(order)))
    assert max(scores) - min(scores) < 0.01, scores


# --- identity and ground truth ----------------------------------------------

def test_the_driver_that_did_not_start_is_not_scored():
    results = _results()
    results.loc[(results.Season == 2025) & (results.RoundNumber == 1)
                & (results.Driver == 'DDD'), 'grid_pos'] = np.nan
    truth = B.outcome(results, 2025, 1)
    assert 'DDD' not in set(truth['Driver'])
    assert len(truth) == 3


def test_a_retirement_is_scored_behind_every_classified_car():
    results = _results()
    mask = ((results.Season == 2025) & (results.RoundNumber == 1)
            & (results.Driver == 'AAA'))
    results.loc[mask, ['finish_pos', 'ClassifiedPosition']] = [np.nan, 'R']
    truth = B.outcome(results, 2025, 1).set_index('Driver')
    assert truth.loc['AAA', 'actual_pos'] > truth.drop('AAA')['actual_pos'].max()
    assert not truth.loc['AAA', 'finished']
    assert not truth.loc['AAA', 'points']


def test_retirements_are_not_dropped_from_the_sample():
    """Scoring only the finishers would grade the model on the easy half."""
    results = _results()
    mask = ((results.Season == 2025) & (results.RoundNumber == 2)
            & results.Driver.isin(['AAA', 'BBB']))
    results.loc[mask, ['finish_pos', 'ClassifiedPosition']] = [np.nan, 'R']
    truth = B.outcome(results, 2025, 2)
    assert len(truth) == 4
    assert truth['finished'].sum() == 2


def test_scores_line_up_by_driver_name_not_by_row_order():
    """The mistake that would be invisible in every summary table."""
    order = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4}
    prediction = _certain(order)
    shuffled = _truth(order).iloc[::-1].reset_index(drop=True)
    assert abs(B.position_errors(prediction, shuffled)['mae_expected']) < 1e-12
    assert abs(B.ranked_probability_score(prediction, shuffled)) < 1e-12


def test_a_missing_driver_shrinks_the_sample_rather_than_scoring_zero():
    order = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4}
    truth = _truth(order)
    truth = truth[truth.Driver != 'DDD']
    errors = B.position_errors(_certain(order), truth)
    assert errors['n_drivers'] == 3


# --- the walk-forward rule --------------------------------------------------

def test_history_never_contains_the_target_race():
    results = _results()
    past = B.history_before(results, 2025, 2)
    assert past[(past.Season == 2025) & (past.RoundNumber == 2)].empty


def test_history_never_contains_a_later_race():
    results = _results()
    past = B.history_before(results, 2025, 1)
    assert past[(past.Season == 2025) & (past.RoundNumber >= 1)].empty
    assert not past[past.Season == 2024].empty


def test_history_for_the_first_race_of_the_archive_is_empty():
    results = _results()
    assert B.history_before(results, 2024, 1).empty


def test_the_universe_excludes_the_old_tyre_era_by_rule():
    frame = B.universe(B.load_results())
    old = frame[frame.Season < B.ERA_FIRST_SEASON]
    assert (old.state == 'excluded').all()
    assert old.reason.str.contains('13-inch').all()


def test_every_included_race_clears_the_training_floor():
    frame = B.universe(B.load_results())
    included = frame[frame.state == 'included']
    assert (included.training_races >= B.MIN_TRAINING_RACES).all()
    assert (included.Season >= B.ERA_FIRST_SEASON).all()


def test_the_universe_is_decided_without_any_prediction():
    """No argument carrying a score can reach it, which is the point."""
    import inspect
    assert set(inspect.signature(B.universe).parameters) == {'results'}


# --- baselines --------------------------------------------------------------

def test_the_grid_baseline_predicts_the_grid():
    entries = pd.DataFrame({'Driver': ['AAA', 'BBB', 'CCC'],
                            'grid_pos': [3.0, 1.0, 2.0]})
    prediction = B.grid_baseline(entries)
    assert list(prediction.mean_pos) == [3, 1, 2]
    assert list(prediction.rank) == [3, 1, 2]


def test_the_grid_baseline_offers_no_probabilities():
    """It has no claim to make about how often anything happens."""
    entries = pd.DataFrame({'Driver': ['AAA', 'BBB'], 'grid_pos': [1.0, 2.0]})
    prediction = B.grid_baseline(entries)
    assert np.isnan(prediction.p_win).all()
    assert B.event_brier(prediction, _truth({'AAA': 1, 'BBB': 2}))['win'] is None


def test_the_rate_baseline_reads_only_the_past():
    results = B.load_results()
    entries = pd.DataFrame({'Driver': ['X'], 'grid_pos': [1.0]})
    early = B.grid_rate_baseline(B.history_before(results, 2023, 1), entries)
    late = B.grid_rate_baseline(B.history_before(results, 2025, 20), entries)
    assert early.p_win[0] != late.p_win[0]


def test_the_rate_baseline_ranks_pole_ahead_of_the_back_row():
    results = B.load_results()
    entries = pd.DataFrame({'Driver': ['P', 'B'], 'grid_pos': [1.0, 18.0]})
    prediction = B.grid_rate_baseline(B.history_before(results, 2025, 1), entries)
    assert prediction.p_win[0] > prediction.p_win[1]
    assert prediction.p_points[0] > prediction.p_points[1]
    assert prediction.mean_pos[0] < prediction.mean_pos[1]


def test_shrinkage_pulls_a_thin_slot_toward_the_field():
    """Three races cannot establish a 67% win rate, and must not be allowed to."""
    rows = []
    for rnd in (1, 2, 3):
        for slot in range(1, 21):
            pos = 1 if (slot == 2 and rnd < 3) else slot
            rows.append(dict(Driver=f'D{slot}', GridPosition=slot,
                             ClassifiedPosition=str(pos), Season=2024,
                             RoundNumber=rnd, Race=f'R{rnd}'))
    history = pd.DataFrame(rows)
    history['grid_pos'] = history['GridPosition'].astype(float)
    entries = pd.DataFrame({'Driver': ['X'], 'grid_pos': [2.0]})
    raw = 2 / 3
    shrunk = B.grid_rate_baseline(history, entries).p_win[0]
    assert shrunk < raw / 2, f'{shrunk} was not pulled far from {raw}'


def test_an_unseen_grid_slot_falls_back_to_the_pooled_rate():
    results = B.load_results()
    entries = pd.DataFrame({'Driver': ['X'], 'grid_pos': [40.0]})
    prediction = B.grid_rate_baseline(results, entries)
    assert 0 < prediction.p_win[0] < 1
    assert not np.isnan(prediction.mean_pos[0])


# --- calibration and uncertainty --------------------------------------------

def test_a_calibrated_set_of_probabilities_matches_its_bands():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(float)
    table = B.calibration(p, y)
    thick = table[~table.thin]
    assert len(thick) >= 4
    assert (abs(thick.predicted - thick.observed) < 0.06).all(), table


def test_an_overconfident_set_is_visible_as_such():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.7, 1.0, 2000)
    y = (rng.uniform(0, 1, 2000) < 0.4).astype(float)
    table = B.calibration(p, y)
    assert (table.observed < table.predicted).all()


def test_thin_bands_are_marked_rather_than_quietly_averaged():
    p = np.concatenate([np.full(3, 0.9), np.full(500, 0.1)])
    y = np.zeros(503)
    table = B.calibration(p, y)
    assert table.thin.any()
    assert bool(table[table.n == 3].thin.iloc[0])


def test_the_bootstrap_resamples_races_not_drivers():
    """
    Twenty drivers in one race are not twenty observations. Feeding the same
    mean as 8 races and as 160 driver rows must not give the same interval.
    """
    rng = np.random.default_rng(3)
    per_race = rng.normal(0.2, 1.0, 8)
    races = B.bootstrap_difference(per_race)
    drivers = B.bootstrap_difference(np.repeat(per_race, 20))
    assert races['n_races'] == 8
    assert (races['hi'] - races['lo']) > 3 * (drivers['hi'] - drivers['lo'])


def test_the_interval_covers_the_mean_and_reports_its_sample():
    values = [0.5, -0.2, 0.9, 0.1, -0.4, 0.3, 0.7, 0.0]
    out = B.bootstrap_difference(values)
    assert out['lo'] < out['mean'] < out['hi']
    assert out['n_races'] == len(values)


def test_a_single_race_gives_no_interval_rather_than_a_fake_one():
    assert B.bootstrap_difference([0.4]) is None


def test_the_bootstrap_ignores_races_that_failed():
    values = [0.5, None, 0.3, float('nan'), 0.4]
    assert B.bootstrap_difference(values)['n_races'] == 3


# --- the prediction object --------------------------------------------------

def test_a_simulation_result_keeps_each_driver_with_its_own_distribution():
    """
    positions is in dnf_by_driver order and summary is sorted by win
    probability. Reading them off by row would attribute every distribution to
    the wrong driver, and every score would still look plausible.
    """
    n_sims, drivers = 500, ['AAA', 'BBB', 'CCC']
    rng = np.random.default_rng(1)
    positions = np.array([rng.permutation([1, 2, 3]) for _ in range(n_sims)])
    summary = pd.DataFrame({
        'Driver': ['CCC', 'AAA', 'BBB'],
        'mean_pos': [positions[:, 2].mean(), positions[:, 0].mean(),
                     positions[:, 1].mean()],
        'P_win': [(positions[:, 2] == 1).mean(), (positions[:, 0] == 1).mean(),
                  (positions[:, 1] == 1).mean()],
        'P_podium': [1.0, 1.0, 1.0], 'P_points': [1.0, 1.0, 1.0]})
    result = {'positions': positions, 'summary': summary,
              'dnf_by_driver': pd.DataFrame({'Driver': drivers})}

    prediction = B.Prediction.from_simulation(result)
    for i, driver in enumerate(prediction.drivers):
        column = drivers.index(driver)
        assert abs(prediction.mean_pos[i] - positions[:, column].mean()) < 1e-9
        assert abs(prediction.dist[i].sum() - 1.0) < 1e-9
        assert abs(prediction.dist[i][0] - (positions[:, column] == 1).mean()) < 1e-9


def test_the_rank_is_a_permutation():
    prediction = B.Prediction(['A', 'B', 'C'], [5.5, 1.2, 3.0],
                              [0, 0, 0], [0, 0, 0], [0, 0, 0])
    assert sorted(prediction.rank) == [1, 2, 3]
    assert list(prediction.rank) == [3, 1, 2]


def test_expected_and_ranked_error_are_reported_separately():
    """They disagree, and the report must never quote whichever is lower."""
    truth = _truth({'A': 1, 'B': 2, 'C': 3})
    prediction = B.Prediction(['A', 'B', 'C'], [1.4, 2.4, 2.6],
                              [0, 0, 0], [0, 0, 0], [0, 0, 0])
    errors = position_errors = B.position_errors(prediction, truth)
    assert errors['mae_expected'] != errors['mae_ranked']
    assert set(position_errors) == {'n_drivers', 'mae_expected', 'mae_ranked'}


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
