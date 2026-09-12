"""
F1 Prediction Simulation - tyre_curve.py
The degradation curve itself. Pure arithmetic: no files, no network, no state.

    D(a) = b1*a + b2*a^2 + gamma * max(0, a - tau)^2

    a      tyre age in laps, measured from the curve's own zero (see below)
    b1     linear degradation           s/lap
    b2     acceleration with age        s/lap^2
    tau    age the cliff begins         laps      (None = no cliff)
    gamma  severity past the cliff      s/lap^2

b0 is fixed at zero, so D(0) = 0 and the curve carries ageing loss only. The
SOFT/MEDIUM/HARD pace difference is NOT in here - the simulation adds that
separately from compound_scaling.csv, and putting it in both places would count
it twice.

Where the zero sits
-------------------
`a` is measured from the profile baseline, not from the tyre's first lap. The
profiles in tyre_age_profile_track.csv are expressed relative to the median of
each stint's opening laps, which puts their zero at FastF1 TyreLife 2. So:

    a = TyreLife - AGE_OFFSET

and D(0) is the pace of a tyre that has already done AGE_OFFSET laps, not a
tyre out of the blanket. The first two laps of wear are outside the curve. That
is a deliberate, documented offset rather than a fitted number: re-anchoring to
the true zero would need a back-extrapolation the data does not support.

Guarantees
----------
Non-decreasing in a, continuous at tau, and identical for identical input on
every run. Nothing here is random, and nothing depends on driver, car,
temperature or traffic - all four are out of scope in this version.
"""

import math

# The profile baseline sits at FastF1 TyreLife 2, so age 0 on this curve is a
# tyre that has completed AGE_OFFSET laps. Changing this invalidates every
# fitted parameter set, so it is versioned alongside them.
AGE_OFFSET = 2

MODEL_NAME = 'two-segment-quadratic'


class TyreCurveError(ValueError):
    """Invalid curve parameters or an out-of-range query."""


class TyreCurve:
    """
    One track x compound curve, with the age range it is allowed to answer for.

    max_age is the observed stint limit for this cell. Past it the curve is not
    evidence any more, so a query is refused rather than extrapolated: nobody
    has run the tyre that long, and a quadratic asked to guess produces a
    confident number with nothing behind it.
    """

    __slots__ = ('track', 'compound', 'b1', 'b2', 'tau', 'gamma',
                 'max_age', 'source', 'notes')

    def __init__(self, track, compound, b1, b2, tau, gamma, max_age,
                 source='unknown', notes=''):
        b1, b2, gamma = float(b1), float(b2), float(gamma)
        max_age = int(max_age)

        if b1 < 0 or b2 < 0 or gamma < 0:
            raise TyreCurveError(
                f'{track}/{compound}: coefficients must be non-negative, '
                f'got b1={b1}, b2={b2}, gamma={gamma}')
        if max_age < 1:
            raise TyreCurveError(f'{track}/{compound}: max_age must be >= 1')

        if tau is None:
            if gamma != 0.0:
                raise TyreCurveError(
                    f'{track}/{compound}: gamma must be 0 when tau is None')
        else:
            tau = float(tau)
            if tau < 0:
                raise TyreCurveError(f'{track}/{compound}: tau must be >= 0')

        self.track = track
        self.compound = compound
        self.b1, self.b2, self.tau, self.gamma = b1, b2, tau, gamma
        self.max_age = max_age
        self.source = source
        self.notes = notes

    # --- queries ------------------------------------------------------------

    def _check_age(self, age):
        if isinstance(age, bool) or not isinstance(age, (int, float)):
            raise TyreCurveError(f'tyre_age must be a number, got {age!r}')
        if isinstance(age, float):
            if math.isnan(age) or math.isinf(age) or not age.is_integer():
                raise TyreCurveError(
                    f'tyre_age must be a whole number of laps, got {age!r}')
            age = int(age)
        if age < 0:
            raise TyreCurveError(f'tyre_age must be >= 0, got {age}')
        if age > self.max_age:
            raise TyreCurveError(
                f'tyre_age {age} is past the observed limit for '
                f'{self.track}/{self.compound} (max_age={self.max_age}); '
                f'no stint that long exists in the data, so the curve will '
                f'not extrapolate')
        return age

    def loss(self, age):
        """Total degradation loss at this age, in seconds."""
        a = self._check_age(age)
        d = self.b1 * a + self.b2 * a * a
        if self.tau is not None and a > self.tau:
            over = a - self.tau
            d += self.gamma * over * over
        return d

    def incremental_loss(self, age):
        """
        Extra loss this lap carries over the previous one: D(a) - D(a-1).

        Zero at age 0 by definition. This is a discrete difference, not the
        derivative - the two differ and the API keeps them apart.
        """
        a = self._check_age(age)
        if a == 0:
            return 0.0
        return self.loss(a) - self.loss(a - 1)

    def slope(self, age):
        """Instantaneous gradient D'(a). Used by the fit, not by the lap loop."""
        a = self._check_age(age)
        d = self.b1 + 2.0 * self.b2 * a
        if self.tau is not None and a > self.tau:
            d += 2.0 * self.gamma * (a - self.tau)
        return d

    def curve(self, max_age=None):
        """
        Ages 0..max_age with their losses, for the whole stint in one call.

        Returns (ages, losses, increments) as plain lists so the API layer can
        serialise them without touching numpy.
        """
        top = self.max_age if max_age is None else int(max_age)
        if top < 0:
            raise TyreCurveError(f'max_age must be >= 0, got {top}')
        if top > self.max_age:
            raise TyreCurveError(
                f'max_age {top} is past the observed limit for '
                f'{self.track}/{self.compound} (max_age={self.max_age})')

        ages = list(range(top + 1))
        losses = [self.loss(a) for a in ages]
        incs = [0.0] + [losses[i] - losses[i - 1] for i in range(1, len(losses))]
        return ages, losses, incs

    # --- serialisation ------------------------------------------------------

    def as_dict(self):
        return {
            'track': self.track,
            'compound': self.compound,
            'beta_1': self.b1,
            'beta_2': self.b2,
            'tau': self.tau,
            'gamma': self.gamma,
            'max_age': self.max_age,
            'source': self.source,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(track=d['track'], compound=d['compound'],
                   b1=d['beta_1'], b2=d['beta_2'], tau=d.get('tau'),
                   gamma=d.get('gamma', 0.0), max_age=d['max_age'],
                   source=d.get('source', 'unknown'), notes=d.get('notes', ''))

    def __repr__(self):
        cliff = 'none' if self.tau is None else f'tau={self.tau:.0f} g={self.gamma:.5f}'
        return (f'<TyreCurve {self.track}/{self.compound} '
                f'b1={self.b1:.4f} b2={self.b2:.5f} {cliff} '
                f'max_age={self.max_age}>')


# --- vectorised helper ------------------------------------------------------


def loss_array(curve, ages):
    """
    D(a) for a sequence of ages. Every age is validated; one bad value fails
    the whole call rather than returning a quietly wrong array.
    """
    return [curve.loss(a) for a in ages]
