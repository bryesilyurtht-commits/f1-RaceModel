"""
F1 Prediction Simulation - v2.2 - dnf.py
Two reasons a car stops, and what race control does about one of them.

What this replaces
------------------
A single flat number. Every car carried the same 6% chance of not finishing,
spread evenly over the race, with no reason attached and no consequence beyond
its own result. The safety car schedule was drawn separately, before the race
started, from a rate that had no idea any of it had happened.

The two causes
--------------
An accident and a failure are not the same risk and do not sit in the same
place. Accident risk belongs to the driver, is broadly flat through the race,
and brings out a safety car about half the time. Mechanical risk belongs to
the car, and where it falls in the race is an empirical question this version
went and asked.

Both are rates per lap at risk, never per race. A driver who crashed twice in
five races did not crash on 40% of laps, and a car that stopped on lap 6 was
not exposed on lap 7. retirements.py builds both denominators properly.

What the data said, and where it disagreed with the plan
--------------------------------------------------------
Three of this version's starting assumptions did not survive contact with the
measurement, and all three are reported rather than quietly kept:

  the bell        The roadmap proposed a hazard rising to mid-race and falling
                  away. Measured on hazard - failures divided by the cars still
                  running, not a histogram of failures - it is very slightly
                  rising and flat wins on AIC by 3.6. So the model is flat, and
                  BELL_SUPPORTED records that this was tested rather than
                  assumed.

  90% of crashes  The project carried "an accident retirement brings out a
                  safety car about 90% of the time". Matched against the
                  neutralizations that actually started within a lap, and
                  deduplicated so one pile-up counts once: 49.6%.

  the total       Six per cent per car was hand-set. The measured accident and
                  mechanical rates together come to about 8.8% over a
                  53-lap race - but they only cover the retirements whose cause
                  the data states, and a third of them say only "Retired".

That last one is a real gap and it is not papered over. The simulation produces
two causes; the unexplained third of history is a fact about the source data,
reported beside the result and never shown as a third probability.

Double counting
---------------
The circuit safety-car rates the simulation already uses were measured over
races that contained these accidents. Switching on accident-triggered safety
cars while leaving that rate alone counts them twice. Of 199 neutralization
starts, 53 match an accident retirement and 146 do not, so the background
process is scaled to 73.4% and the accidents supply the rest. There is still
only one neutralization manager.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

NONE, ACCIDENT, MECHANICAL = 0, 1, 2
CAUSE_NAMES = ('running', 'accident', 'mechanical')

# --- measured, by retirements.py over 2018-2025 -----------------------------
# Per lap at risk. Used when a driver or team has no row of their own, which
# for a 2026 grid is most of them: a rookie and a brand-new team are not
# safer than the field, they are simply unmeasured, and the pooled rate is the
# only honest thing to give them.
POOLED_ACCIDENT = 0.000811
POOLED_MECHANICAL = 0.000854

# What followed an accident retirement, from 123 deduplicated incidents.
# Measured, and a long way from the 90% the project had been carrying.
ACCIDENT_OUTCOME = (('none', 0.504), ('VSC', 0.016), ('SC', 0.382),
                    ('RF', 0.098))

# Share of neutralization starts not matched to an accident retirement. The
# existing circuit rate is scaled to this so the two sources add up to what
# was measured rather than to nearly twice it.
BACKGROUND_SHARE = 0.734

# Tested, not assumed. retirements.py compares a flat per-lap hazard against
# the best bell over a grid, on hazard rather than on an event histogram, and
# flat wins by 3.6 of AIC. Set True only with a measurement that says so.
BELL_SUPPORTED = False
BELL_PEAK = 0.50
BELL_WIDTH = 0.25

# What share of real retirements the two modelled causes actually account for.
# Not used in the arithmetic - it is reported, so that a 9% modelled DNF rate
# is never mistaken for the 14% the results file actually contains.
CAUSE_COVERAGE = 0.661


def _read(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def rates(pace):
    """
    Each car's two per-lap rates, and where they came from.

    A driver or team with no row gets the pooled figure rather than zero.
    Zero would mean a car that cannot fail, and a model containing one will
    eventually hand it a championship.
    """
    n = len(pace)
    accident = np.full(n, POOLED_ACCIDENT)
    mechanical = np.full(n, POOLED_MECHANICAL)
    sources = {'accident': 'pooled', 'mechanical': 'pooled'}

    table = _read('dnf_driver_risk.csv')
    if table is not None and 'rate' in table.columns:
        lookup = table.set_index('Driver')['rate'].to_dict()
        mapped = pace['Driver'].map(lookup)
        accident = mapped.fillna(POOLED_ACCIDENT).to_numpy(float)
        sources['accident'] = (f'measured for {int(mapped.notna().sum())}/{n} '
                               f'drivers, pooled for the rest')

    table = _read('dnf_team_risk.csv')
    if table is not None and 'rate' in table.columns:
        lookup = table.set_index('Team')['rate'].to_dict()
        mapped = pace['Team'].map(lookup)
        mechanical = mapped.fillna(POOLED_MECHANICAL).to_numpy(float)
        sources['mechanical'] = (f'measured for {int(mapped.notna().sum())}/{n} '
                                 f'cars, pooled for the rest')
    return accident, mechanical, sources


def timing_shape(n_laps):
    """
    How mechanical risk is distributed through the race, scaled to leave the
    total alone.

    The scaling is the part that matters. A shape that multiplies the hazard
    without being normalised changes how many failures happen as well as when,
    and then a change of shape looks like a change of reliability. This is
    divided by its own mean, so the shape moves risk around the race and the
    total stays where the measurement put it.
    """
    if not BELL_SUPPORTED:
        return np.ones(n_laps)

    progress = (np.arange(n_laps) + 0.5) / n_laps
    shape = np.exp(-0.5 * ((progress - BELL_PEAK) / BELL_WIDTH) ** 2)
    return shape / shape.mean()


def accident_outcome_table():
    """The measured mix, as cumulative thresholds for one uniform draw."""
    labels = [name for name, _ in ACCIDENT_OUTCOME]
    weights = np.array([w for _, w in ACCIDENT_OUTCOME], dtype=float)
    return labels, np.cumsum(weights / weights.sum())


def draw(rng, shape, accident_rate, mechanical_rate, active, scale=1.0):
    """
    One lap's retirements, at most one per car and with exactly one cause.

    The two rates are added and the event drawn once from the total, then the
    cause is picked in proportion. Two independent yes/no draws would need a
    tie-break when both came up, and whichever cause the tie-break favoured
    would quietly gain risk the measurement never gave it.

    Densities, not probabilities: they are added as densities and converted
    once, so the result cannot leave [0, 1] no matter how high the two go.

    `active` is the cars still running. A retired car does not draw again.
    """
    total = (accident_rate + mechanical_rate) * scale
    p_any = 1.0 - np.exp(-total)
    hit = active & (rng.random(shape) < p_any)

    share = np.divide(accident_rate, np.maximum(total, 1e-15),
                      out=np.zeros_like(total), where=total > 0)
    is_accident = rng.random(shape) < share

    cause = np.where(hit, np.where(is_accident, ACCIDENT, MECHANICAL), NONE)
    return hit, cause.astype(np.int8)


def summarise(diag, n_sims, n_drivers):
    """
    Finishing and the two causes, which together are the whole of it.

    The unexplained third of the historical retirements is not a third
    column here. It is a statement about the source data, carried alongside
    as coverage, because the simulation never produces such an event and
    showing it as a probability would imply that it does.
    """
    cause = diag['dnf_cause']
    finished = float((cause == NONE).mean())
    accident = float((cause == ACCIDENT).mean())
    mechanical = float((cause == MECHANICAL).mean())

    laps = diag['retired_lap']
    out = {
        'p_finish': finished,
        'p_accident': accident,
        'p_mechanical': mechanical,
        'total': accident + mechanical,
        'coverage': CAUSE_COVERAGE,
        'bell_supported': BELL_SUPPORTED,
        'accident_neutralizations': float(diag['acc_neutral'].mean()),
        'background_neutralizations': float(diag['bg_neutral'].mean()),
        'outcome_mix': dict(ACCIDENT_OUTCOME),
    }
    for label, code in (('accident', ACCIDENT), ('mechanical', MECHANICAL)):
        where = cause == code
        out[f'{label}_laps'] = (laps[where].astype(float)
                                if where.any() else np.array([]))
    return out


def per_driver(diag, drivers):
    """Finish, accident and mechanical probability for each car."""
    cause = diag['dnf_cause']
    return pd.DataFrame({
        'Driver': list(drivers),
        'P_finish': (cause == NONE).mean(axis=0),
        'P_accident': (cause == ACCIDENT).mean(axis=0),
        'P_mechanical': (cause == MECHANICAL).mean(axis=0),
    })


def provenance():
    """Which of these numbers were measured and which were chosen."""
    return [
        ('Accident risk, per driver', 'measured',
         f'{POOLED_ACCIDENT * 1000:.3f} per 1,000 laps pooled',
         '153 accident retirements over 2018-2025, on laps at risk'),
        ('Mechanical risk, per team', 'measured',
         f'{POOLED_MECHANICAL * 1000:.3f} per 1,000 laps pooled',
         '161 mechanical retirements, on laps at risk'),
        ('Shrinkage toward the field', 'hand-set', '30 car-races',
         'no out-of-sample exercise behind it'),
        ('Failure timing', 'measured', 'flat through the race',
         'the proposed bell lost to flat by 3.6 of AIC'),
        ('Accident brings out a flag', 'measured', '49.6%',
         '123 deduplicated incidents, not the 90% assumed'),
        ('What kind of flag', 'measured', 'SC 38%, RF 10%, VSC 2%',
         'same incidents'),
        ('Background safety cars', 'measured', f'{BACKGROUND_SHARE:.1%}',
         '146 of 199 starts unmatched to an accident'),
        ('Cause coverage', 'measured', f'{CAUSE_COVERAGE:.1%}',
         'a third of retirements say only "Retired" and are not redistributed'),
        ('Accidents under yellow', 'assumption', 'none drawn',
         'no separate measurement of crash risk behind a safety car'),
    ]
