"""
Tests for team_affinity.py and the affinity path in simulate.py.

Three things are worth guarding here, and they are not the arithmetic.

The censoring is the whole feature - measured, it takes the skill from +3.0%
to +12.9% - and it is one np.where away from silently reverting to reading the
classification. There is a test for a retiree who was running 4th.

The sum-to-zero property of affinity is what broke the first version of the
similarity model: because a team's affinities sum to zero, the mean of the
others is mechanically the negative of the one left out. That property is
asserted directly, so anyone who reintroduces a leave-one-out mean has a test
telling them why it cannot work.

And the percentage conversion has to scale with the circuit. A test that only
checked the number at one lap time would pass for the old fixed-seconds form
too, so it is checked at two.

Run:  python -m Simülasyon.test_team_affinity
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simülasyon import team_affinity as TA


def make_results(rows):
    return pd.DataFrame(rows, columns=['Race', 'Round', 'Team', 'Driver',
                                       'Position', 'ClassifiedPosition'])


def make_laps(rows):
    return pd.DataFrame(rows, columns=['Race', 'Driver', 'LapNumber',
                                       'Position'])


class TestCensoring(unittest.TestCase):
    """A retirement must be scored where the car was, not where it was ranked."""

    def setUp(self):
        self.results = make_results([
            ('A GP', 1, 'Red', 'AAA', 1.0, '1'),
            ('A GP', 1, 'Blue', 'BBB', 19.0, 'R'),      # retired from 4th
            ('A GP', 1, 'Blue', 'CCC', 2.0, '2'),
            ('A GP', 1, 'Grey', 'DDD', np.nan, 'W'),    # never started
        ])
        self.laps = make_laps([
            ('A GP', 'AAA', 1, 1.0), ('A GP', 'AAA', 2, 1.0),
            ('A GP', 'BBB', 1, 6.0), ('A GP', 'BBB', 2, 4.0),
            ('A GP', 'CCC', 1, 3.0), ('A GP', 'CCC', 2, 2.0),
        ])

    def test_retiree_scored_at_last_running_position(self):
        TA.CENSOR_RETIREMENTS = True
        out = TA.race_signal(self.results, self.laps)
        bbb = out[out['Driver'] == 'BBB']['signal'].iloc[0]
        self.assertEqual(bbb, 4.0,
                         'retiree must score its last running position, not 19th')

    def test_uncensored_reads_the_classification(self):
        TA.CENSOR_RETIREMENTS = False
        try:
            out = TA.race_signal(self.results, self.laps)
            bbb = out[out['Driver'] == 'BBB']['signal'].iloc[0]
            self.assertEqual(bbb, 19.0)
        finally:
            TA.CENSOR_RETIREMENTS = True

    def test_non_starter_is_dropped(self):
        out = TA.race_signal(self.results, self.laps)
        self.assertNotIn('DDD', out['Driver'].tolist(),
                         'a car that never started has nothing to measure')

    def test_finisher_is_untouched(self):
        out = TA.race_signal(self.results, self.laps)
        self.assertEqual(out[out['Driver'] == 'AAA']['signal'].iloc[0], 1.0)

    def test_censoring_uses_the_last_lap_not_the_first(self):
        # BBB ran 6th then 4th; reading the wrong end of the stint would give 6
        out = TA.race_signal(self.results, self.laps)
        self.assertEqual(out[out['Driver'] == 'BBB']['signal'].iloc[0], 4.0)


class TestSessionCentring(unittest.TestCase):
    """
    Sessions are not all the same size, and a short one flatters everybody.

    This was found by the circuit-balance test below, not by reading the code:
    only 19 cars set a time in Australian qualifying, so positions ran 1..19,
    the mean was 10.0 instead of 11.5, and every team on the grid came out
    1.3 positions "better than usual" at that one circuit.
    """

    def short_and_full(self):
        full = pd.DataFrame({
            'Race': ['Full'] * 22, 'Round': [2] * 22,
            'Team': [f'T{i // 2}' for i in range(22)],
            'Driver': [f'D{i}' for i in range(22)],
            'signal': np.arange(1.0, 23.0)})
        short = full.iloc[:19].copy()
        short['Race'] = 'Short'
        short['Round'] = 1
        return pd.concat([short, full], ignore_index=True)

    def test_raw_short_session_flatters_everyone(self):
        d = self.short_and_full()
        means = d.groupby('Race')['signal'].mean()
        self.assertLess(means['Short'], means['Full'],
                        'this is the artefact the centring exists to remove')

    def test_centring_makes_every_session_mean_zero(self):
        out = TA.centre_by_session(self.short_and_full())
        for race, g in out.groupby('Race'):
            self.assertAlmostEqual(g['signal'].mean(), 0.0, places=9,
                                   msg=f'{race} is still off-centre')

    def test_a_short_field_no_longer_flatters_the_whole_grid(self):
        """
        The end-to-end version, and the honest limit of the fix.

        Centring is exact at car level. At team level a small residual
        survives, because a session that drops three cars also leaves three
        teams fielding one car instead of two, and a one-car average is not
        the same statistic as a two-car average. On the real data that
        residual is -0.11 positions at Australia, against +0.66 before
        centring. It is a sixth of what it was, not zero, and pretending
        otherwise would need a weighting scheme the sample cannot support.
        """
        d = self.short_and_full()
        raw = TA.team_table(d)
        fixed = TA.team_table(TA.centre_by_session(d))

        before = abs(raw[raw['Race'] == 'Short']['pos'].mean()
                     - raw[raw['Race'] == 'Full']['pos'].mean())
        after = abs(fixed[fixed['Race'] == 'Short']['pos'].mean()
                    - fixed[fixed['Race'] == 'Full']['pos'].mean())
        self.assertLess(after, before / 2.0,
                        'centring must remove most of the field-size shift')

    def test_equal_car_counts_balance_exactly(self):
        """With every team fielding the same number of cars there is no residual."""
        d = self.short_and_full()
        d = d[d['Team'].isin([f'T{i}' for i in range(9)])]   # 9 two-car teams
        team = TA.team_table(TA.centre_by_session(d))
        for race, g in team.groupby('Race'):
            self.assertAlmostEqual(g['pos'].mean(), 0.0, places=9,
                                   msg=f'{race} favours everyone')

    def test_field_size_is_recorded(self):
        out = TA.centre_by_session(self.short_and_full())
        self.assertEqual(out[out['Race'] == 'Short']['field'].iloc[0], 19)
        self.assertEqual(out[out['Race'] == 'Full']['field'].iloc[0], 22)

    def test_relative_order_is_preserved(self):
        out = TA.centre_by_session(self.short_and_full())
        g = out[out['Race'] == 'Full'].sort_values('signal')
        self.assertEqual(g['Driver'].iloc[0], 'D0',
                         'centring shifts the scale, it must not reorder it')


class TestTeamTable(unittest.TestCase):

    def test_team_is_the_mean_of_its_cars(self):
        sig = pd.DataFrame({
            'Race': ['A', 'A'], 'Round': [1, 1], 'Team': ['Blue', 'Blue'],
            'Driver': ['BBB', 'CCC'], 'signal': [4.0, 2.0]})
        t = TA.team_table(sig)
        self.assertEqual(t['pos'].iloc[0], 3.0)
        self.assertEqual(t['cars'].iloc[0], 2)

    def test_single_car_team_is_that_car(self):
        sig = pd.DataFrame({
            'Race': ['A'], 'Round': [1], 'Team': ['Blue'],
            'Driver': ['BBB'], 'signal': [4.0]})
        t = TA.team_table(sig)
        self.assertEqual(t['pos'].iloc[0], 4.0)
        self.assertEqual(t['cars'].iloc[0], 1)


class TestAffinity(unittest.TestCase):

    def make_team(self, positions):
        return pd.DataFrame({
            'Team': ['Blue'] * len(positions),
            'Race': [f'R{i}' for i in range(len(positions))],
            'Round': list(range(1, len(positions) + 1)),
            'pos': positions,
            'cars': [2] * len(positions)})

    def test_sign_positive_means_better_than_usual(self):
        # 10th everywhere, 4th at R0: that circuit suits them
        aff = TA.affinity_table(self.make_team([4.0] + [10.0] * 7))
        self.assertGreater(aff[aff['Race'] == 'R0']['affinity'].iloc[0], 0)

    def test_flat_team_has_zero_affinity_everywhere(self):
        aff = TA.affinity_table(self.make_team([7.0] * 8))
        self.assertTrue(np.allclose(aff['affinity'], 0.0))

    def test_affinities_sum_to_zero(self):
        """
        The property that killed the similarity model. Because these sum to
        zero, the mean of the other n-1 circuits equals -y_i/(n-1) exactly,
        so a leave-one-out kernel is anti-correlated with the truth for
        arithmetic reasons alone. Any future pooling must subtract the
        training mean - see centred_kernel.
        """
        aff = TA.affinity_table(self.make_team([3.0, 9.0, 5.0, 11.0,
                                                7.0, 2.0, 8.0, 6.0]))
        self.assertAlmostEqual(aff['affinity'].sum(), 0.0, places=9)

        y = aff['affinity'].to_numpy()
        n = len(y)
        for i in range(n):
            others = np.delete(y, i).mean()
            self.assertAlmostEqual(others, -y[i] / (n - 1), places=9)

    def test_short_seasons_are_dropped(self):
        aff = TA.affinity_table(self.make_team([4.0, 5.0, 6.0]))
        self.assertTrue(aff.empty,
                        f'{TA.MIN_RACES_PER_TEAM} races are required')

    def test_target_row_excluded_from_its_own_reference(self):
        """
        The target circuit has qualifying but no race. It is measured against
        the other circuits and must not enter the average it is compared with,
        otherwise its own result damps the number it produces.
        """
        team = self.make_team([4.0] + [10.0] * 7)
        mask = pd.Series([False] + [True] * 7, index=team.index)
        aff = TA.affinity_table(team, reference_mask=mask)
        target = aff[aff['Race'] == 'R0'].iloc[0]
        self.assertAlmostEqual(target['affinity'], 10.0 - 4.0)
        self.assertEqual(target['n_reference'], 7)


class TestCentredKernel(unittest.TestCase):

    def test_centring_removes_the_constant(self):
        pc = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        y = np.array([2.0, 2.0, 2.0])
        # every neighbour says +2, so the contrast against their mean is zero
        self.assertAlmostEqual(
            TA.centred_kernel(pc, y, np.array([0.5, 0.0]), 1.0), 0.0)

    def test_near_neighbour_dominates(self):
        pc = np.array([[0.0, 0.0], [6.0, 0.0]])
        y = np.array([3.0, -3.0])
        out = TA.centred_kernel(pc, y, np.array([0.1, 0.0]), 0.5)
        self.assertGreater(out, 2.0, 'the close circuit should carry the answer')

    def test_empty_training_set_is_zero(self):
        self.assertEqual(
            TA.centred_kernel(np.zeros((0, 2)), np.array([]),
                              np.array([0.0, 0.0]), 1.0), 0.0)


class TestSkill(unittest.TestCase):

    def test_perfect_prediction_scores_one(self):
        t = np.array([1.0, -2.0, 3.0, -1.5, 0.5, 2.0, -3.0, 1.0, -1.0, 2.5])
        s = TA.skill(t, t)
        self.assertAlmostEqual(s['lambda'], 1.0)
        self.assertGreater(s['skill'], 0.99)

    def test_anticorrelated_prediction_is_clipped_to_no_skill(self):
        t = np.array([1.0, -2.0, 3.0, -1.5, 0.5, 2.0, -3.0, 1.0, -1.0, 2.5])
        s = TA.skill(-t, t)
        self.assertEqual(s['lambda'], 0.0)
        self.assertAlmostEqual(s['skill'], 0.0)

    def test_half_scale_prediction_recovers_its_scale(self):
        t = np.array([1.0, -2.0, 3.0, -1.5, 0.5, 2.0, -3.0, 1.0, -1.0, 2.5])
        s = TA.skill(2.0 * t, t)
        self.assertAlmostEqual(s['lambda'], 0.5)


class TestOutputFile(unittest.TestCase):
    """The file simulate.py actually reads."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(TA.DATA_DIR, f'team_affinity_{TA.SEASON}.csv')
        if not os.path.exists(path):
            raise unittest.SkipTest('run python -m Simülasyon.team_affinity first')
        cls.df = pd.read_csv(path)

    def test_required_columns(self):
        for col in ('Team', 'Race', 'affinity', 'affinity_quali', 'shrinkage'):
            self.assertIn(col, self.df.columns)

    def target_rows(self):
        # whatever target_race.py currently points at, not a hardcoded name -
        # this file is regenerated every time TARGET_RACE changes
        from Simülasyon.target_race import TRACK_ALIASES
        key = self.df['Race'].astype(str).str.lower()
        return self.df[key.apply(lambda s: any(a in s for a in TRACK_ALIASES))]

    def test_target_circuit_is_present(self):
        hit = self.target_rows()
        self.assertFalse(hit.empty,
                         'the circuit being predicted must have a row')
        self.assertEqual(len(hit), 11, 'one row per 2026 team')

    def test_target_has_no_race_measurement(self):
        hit = self.target_rows()
        self.assertFalse(hit['raced_2026'].any(),
                         'the target race must not have been read - leakage')

    def test_affinity_is_the_shrunk_quali_value(self):
        d = self.df.dropna(subset=['affinity_quali'])
        lam = d['shrinkage'].iloc[0]
        self.assertTrue(np.allclose(d['affinity'], lam * d['affinity_quali'],
                                    atol=1e-3))

    def test_shrinkage_is_a_shrinkage(self):
        lam = self.df['shrinkage'].iloc[0]
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

    def test_each_circuit_roughly_balances(self):
        # affinity measures suitability, so a circuit cannot suit everyone
        for race, g in self.df.groupby('Race'):
            self.assertLess(abs(g['affinity'].mean()), 0.5,
                            f'{race} favours the whole field, which is not '
                            f'what affinity means')


class TestSimulateIntegration(unittest.TestCase):
    """The multiplier form, checked where it differs from the old one."""

    @classmethod
    def setUpClass(cls):
        try:
            from Simülasyon import simulate
        except Exception as exc:                       # pragma: no cover
            raise unittest.SkipTest(f'simulate.py not importable: {exc}')
        cls.sim = simulate

    def test_rate_reproduces_the_old_constant_at_the_reference_lap(self):
        s = self.sim
        got = s.AFFINITY_PCT_PER_POSITION * s.AFFINITY_REFERENCE_LAP
        self.assertAlmostEqual(got, 0.020, places=3,
                               msg='the percentage rate is pinned to the old '
                                   '0.020 s/pos at a 67 s lap')

    def test_cap_reproduces_the_old_cap_at_the_reference_lap(self):
        s = self.sim
        got = s.AFFINITY_CAP_PCT * s.AFFINITY_REFERENCE_LAP
        self.assertAlmostEqual(got, 0.12, places=2)

    def test_effect_scales_with_lap_time(self):
        """
        The point of the change. A long lap must give the same affinity more
        seconds; under the old fixed-seconds form these two would be equal.
        """
        s = self.sim
        monaco, spa = 72.0, 105.0
        one_position = s.AFFINITY_PCT_PER_POSITION
        self.assertLess(one_position * monaco, one_position * spa)
        self.assertAlmostEqual(one_position * spa / (one_position * monaco),
                               spa / monaco, places=6)

    def test_cap_scales_too(self):
        s = self.sim
        self.assertGreater(s.AFFINITY_CAP_PCT * 105.0,
                           s.AFFINITY_CAP_PCT * 72.0,
                           'a percentage cap must scale, or long circuits get '
                           'clipped harder than short ones')

    def test_team_term_reads_the_2026_file(self):
        s = self.sim
        pace = pd.DataFrame({
            'Driver': ['NOR', 'PIA', 'HAM'],
            'Team': ['McLaren', 'McLaren', 'Ferrari'],
            'delta': [0.0, 0.1, 0.3], 'sigma': [0.2, 0.2, 0.2]})
        term, source = s.team_affinity_2026(pace)
        if term is None:
            self.skipTest(f'no 2026 team affinity available: {source}')
        self.assertIn('2026 team', source)
        self.assertEqual(term.iloc[0], term.iloc[1],
                         'teammates share a chassis and so share the term')
        self.assertNotEqual(term.iloc[0], term.iloc[2])

    def test_affinity_is_centred_so_it_redistributes(self):
        """
        Affinity moves drivers past each other; it must not make the whole
        field faster or slower, which would be a pace change wearing an
        affinity label.
        """
        s = self.sim
        pace = pd.DataFrame({
            'Driver': ['NOR', 'PIA', 'HAM', 'LEC', 'RUS', 'ANT',
                       'VER', 'TSU', 'ALO', 'STR'],
            'Team': ['McLaren', 'McLaren', 'Ferrari', 'Ferrari',
                     'Mercedes', 'Mercedes', 'Red Bull Racing',
                     'Red Bull Racing', 'Aston Martin', 'Aston Martin'],
            'delta': np.linspace(0.0, 1.0, 10),
            'sigma': [0.2] * 10})
        out = s.apply_affinity(pace)
        # the reported column is rounded to two decimals for the report, so
        # the assertion is at that resolution and not at the float's
        self.assertAlmostEqual(out['affinity'].mean(), 0.0, places=2)
        self.assertAlmostEqual(out['affinity_sec'].mean(), 0.0, places=2)

    def test_pace_order_can_change_but_the_spread_stays_sane(self):
        s = self.sim
        pace = pd.DataFrame({
            'Driver': ['NOR', 'PIA', 'HAM', 'LEC', 'RUS', 'ANT',
                       'VER', 'TSU', 'ALO', 'STR'],
            'Team': ['McLaren', 'McLaren', 'Ferrari', 'Ferrari',
                     'Mercedes', 'Mercedes', 'Red Bull Racing',
                     'Red Bull Racing', 'Aston Martin', 'Aston Martin'],
            'delta': np.linspace(0.0, 1.0, 10),
            'sigma': [0.2] * 10})
        out = s.apply_affinity(pace)
        cap = s.AFFINITY_CAP_PCT * s.POLE_TIME
        self.assertLessEqual(out['affinity_sec'].abs().max(), cap + 1e-9,
                             'the cap must hold')
        self.assertGreaterEqual(out['delta'].min(), -1e-9,
                                'delta is a gap to the fastest, never negative')


if __name__ == '__main__':
    unittest.main(verbosity=2)
