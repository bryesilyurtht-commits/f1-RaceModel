"""
Tests for the v1.5 pass model: the gap term and the PC1 team term.

Two of these exist because the mistakes already happened.

The gap coefficient was fitted on laps where a car inside a second had at
least 0.15 s/lap in hand. Applied to every lap that reaches the attacking
branch it is an extrapolation, and over half of those laps sit outside the
fitted region - the simulation carries an unsorted cumulative-time array, so
its gap goes negative whenever a car has gained time it has not yet been
given the place for. Dropped in without the domain guard the model produced
118 overtakes at a circuit whose own history says forty. There is a test for
the guard and a test for the negative gap.

The other is HELD_GAP. A car that failed to pass used to be pinned at
MIN_GAP = 0.35 s, where 7% of real blocked cars are. That was harmless while
pass probability ignored the gap and fed only dirty air; the moment the gap
started pricing a lap it became a +1.02 logit bonus handed out every lap to
every stuck car.

Run:  python -m Simülasyon.test_pass_model
"""

import os
import sys
import unittest

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simülasyon import simulate as S
from Simülasyon.tracks import ATTACK_GAP, HELD_GAP, MIN_GAP


def model(near=0.271, wide=0.219):
    return S.build_pass_model({'pass_rate': near, 'pass_rate_wide': wide})


def p(advantage, gap=None, shift=0.0, m=None):
    out = S.pass_probability(np.atleast_1d(np.asarray(advantage, float)),
                             m if m is not None else model(),
                             gap=None if gap is None
                             else np.atleast_1d(np.asarray(gap, float)),
                             team_shift=shift)
    return float(np.asarray(out).ravel()[0])


class TestGapTerm(unittest.TestCase):

    def test_closer_is_better(self):
        near = p(0.5, gap=0.25)
        far = p(0.5, gap=0.95)
        self.assertGreater(near, far,
                           'a car 0.25 s back must have a better chance than '
                           'one hanging on at 0.95 s')

    def test_the_difference_is_large(self):
        # measured: 0.69 inside a quarter second against 0.10 approaching one
        self.assertGreater(p(0.5, gap=0.25) / max(p(0.5, gap=0.95), 1e-9), 3.0)

    def test_monotone_across_the_window(self):
        gaps = [0.1, 0.3, 0.5, 0.7, 0.9]
        probs = [p(0.5, gap=g) for g in gaps]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_pace_still_matters_at_a_fixed_gap(self):
        self.assertGreater(p(1.2, gap=0.5), p(0.2, gap=0.5))

    def test_anchor_reproduces_the_measured_rate(self):
        """At the reference point the model must return what was measured."""
        for rate in (0.10, 0.271, 0.45):
            got = p(S.REFERENCE_ADVANTAGE, gap=S.REFERENCE_GAP,
                    m=model(near=rate))
            self.assertAlmostEqual(got, rate, places=6,
                                   msg=f'anchor drifted at base {rate}')


class TestFittedDomain(unittest.TestCase):
    """The guard that separates 47 overtakes from 118."""

    def test_below_min_advantage_falls_back(self):
        """
        A car with no pace in hand must not collect the closeness bonus.

        This is the failure that produced 118 overtakes: at zero advantage
        and 0.2 s back the extrapolated model returned 0.42.
        """
        low = 0.5 * S.GAP_MODEL_MIN_ADVANTAGE
        with_gap = p(low, gap=0.2)
        without = p(low, gap=None)
        self.assertAlmostEqual(with_gap, without, places=9,
                               msg='the gap term reached a lap it was never '
                                   'fitted on')

    def test_at_min_advantage_the_term_engages(self):
        on = p(S.GAP_MODEL_MIN_ADVANTAGE, gap=0.2)
        off = p(S.GAP_MODEL_MIN_ADVANTAGE, gap=None)
        self.assertNotAlmostEqual(on, off, places=6)

    def test_negative_gap_falls_back(self):
        """
        run_simulation's gap goes negative by design - the order is carried,
        not re-sorted, so a car can be ahead on time before it is ahead on
        track. The measurement never saw such a lap.
        """
        self.assertAlmostEqual(p(0.5, gap=-0.4), p(0.5, gap=None), places=9)

    def test_beyond_attack_gap_falls_back(self):
        self.assertAlmostEqual(p(0.5, gap=ATTACK_GAP + 0.3),
                               p(0.5, gap=None), places=9)

    def test_fallback_is_the_old_model(self):
        """Outside the domain nothing about the previous behaviour changed."""
        m = model()
        old = 1.0 / (1.0 + np.exp(-(0.5 - m['threshold']) / S.POOLED_SCALE))
        self.assertAlmostEqual(p(0.5, gap=None), min(old, S.MAX_PASS_PROB),
                               places=9)

    def test_fallback_anchored_on_the_wide_window(self):
        """
        The two anchors are different rates on purpose. Outside the gap fit
        the model is the one that was always calibrated on the full window.
        """
        m = model(near=0.271, wide=0.219)
        self.assertAlmostEqual(p(S.REFERENCE_ADVANTAGE, gap=None, m=m),
                               0.219, places=6)

    def test_mixed_array_splits_correctly(self):
        adv = np.array([0.05, 0.50, 0.50, 0.50])
        gap = np.array([0.20, 0.20, -0.4, 0.20])
        out = np.asarray(S.pass_probability(adv, model(), gap=gap))
        self.assertAlmostEqual(out[0], p(0.05, gap=None), places=9)
        self.assertAlmostEqual(out[2], p(0.50, gap=None), places=9)
        self.assertAlmostEqual(out[1], out[3], places=9)
        self.assertGreater(out[1], out[2])


class TestHeldGap(unittest.TestCase):

    def test_held_gap_is_where_blocked_cars_actually_are(self):
        """Measured median of a car that failed to pass: 0.659 s."""
        self.assertGreater(HELD_GAP, 0.55)
        self.assertLess(HELD_GAP, 0.80)

    def test_held_gap_still_counts_as_attacking(self):
        """Or a car that failed once would never try again."""
        self.assertLess(HELD_GAP, ATTACK_GAP)

    def test_held_gap_is_not_a_bonus(self):
        """
        The old MIN_GAP sat in the part of the window the model now rewards.
        A blocked car must not be handed a better chance than the average one
        simply for having been blocked.
        """
        at_held = p(0.5, gap=HELD_GAP)
        at_reference = p(0.5, gap=S.REFERENCE_GAP)
        self.assertLessEqual(at_held, at_reference + 1e-9)

        at_min = p(0.5, gap=MIN_GAP)
        self.assertGreater(at_min, at_held * 1.5,
                           'this is the bonus the old constant was handing '
                           'out every lap')


class TestTeamTerm(unittest.TestCase):

    def test_higher_pc1_passes_more(self):
        k = S.TEAM_PASS_COEF
        best = p(0.5, gap=0.5, shift=2.66 * k)     # Mercedes
        worst = p(0.5, gap=0.5, shift=-2.17 * k)   # Aston Martin
        self.assertGreater(best, worst)

    def test_the_spread_is_modest(self):
        """
        Raw coefficients put the field 3.7x apart, which is not a credible
        spread between two cars at equal pace. The permutation shrinkage
        brings it to about 1.6x.
        """
        k = S.TEAM_PASS_COEF
        best = p(0.5, gap=0.6, shift=2.66 * k)
        worst = p(0.5, gap=0.6, shift=-2.17 * k)
        self.assertLess(best / worst, 2.2)
        self.assertGreater(best / worst, 1.2)

    def test_zero_pc1_changes_nothing(self):
        self.assertAlmostEqual(p(0.5, gap=0.5, shift=0.0),
                               p(0.5, gap=0.5), places=9)

    def test_shift_is_capped(self):
        huge = p(0.5, gap=0.5, shift=99.0)
        at_cap = p(0.5, gap=0.5, shift=S.TEAM_PASS_CAP)
        self.assertAlmostEqual(huge, at_cap, places=9)

    def test_team_term_reaches_the_fallback_too(self):
        """A team's car is the same car outside the gap fit's domain."""
        k = S.TEAM_PASS_COEF
        self.assertGreater(p(0.05, gap=0.2, shift=2.66 * k),
                           p(0.05, gap=0.2, shift=-2.17 * k))

    def test_profile_table_maps_onto_the_entry_list(self):
        import pandas as pd
        path = os.path.join(S.DATA_DIR, f'overtake_team_profile_{S.SEASON}.csv')
        if not os.path.exists(path):
            self.skipTest('run python -m Simülasyon.overtake_speed first')
        pace = S.read_csv(f'driver_pace_{S.SEASON}.csv')
        shift, source = S.team_pass_shift(pace)
        self.assertEqual(len(shift), len(pace))
        self.assertIn('PC1', source)
        self.assertLessEqual(float(np.abs(shift).max()), S.TEAM_PASS_CAP + 1e-9)
        # PC1 is mean-centred by construction, so this redistributes
        self.assertLess(abs(float(np.mean(shift))), 0.05)


class TestCalibration(unittest.TestCase):

    def test_near_window_rate_exceeds_the_wide_one(self):
        """
        overtaking.py measures from 1.5 s but nothing is rolled beyond 1.0,
        and passes are rarer out in that band. If this ever inverts, the
        measurement has changed shape and the anchoring needs rereading.
        """
        import pandas as pd
        path = os.path.join(S.DATA_DIR, 'overtaking.csv')
        if not os.path.exists(path):
            self.skipTest('overtaking.csv missing')
        df = pd.read_csv(path)
        if 'pass_rate_near' not in df.columns:
            self.skipTest('run python -m Simülasyon.overtaking first')
        d = df.dropna(subset=['pass_rate_near', 'pass_rate'])
        share = (d['pass_rate_near'] >= d['pass_rate']).mean()
        self.assertGreater(share, 0.8,
                           'the narrow window should read higher almost '
                           'everywhere')

    def test_reference_gap_matches_the_measurement(self):
        self.assertGreater(S.REFERENCE_GAP, 0.4)
        self.assertLess(S.REFERENCE_GAP, 0.8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
