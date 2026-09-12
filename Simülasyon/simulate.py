"""
F1 Prediction Simulation - v1.1-A - simulate.py
Lap-by-lap Monte Carlo: grid start, tyres, fuel, pit strategy, neutralizations,
overtaking and dirty air.

    lap_time = pole + delta + compound_offset + tyre_penalty(compound, age)
               - fuel_effect*(lap - mid) + dirty_air(gap) + N(0, sigma)
    plus pit_loss on a stop and a lap penalty under neutralization

What changed in v1.2
--------------------
The tyre model is rebuilt. Through v1.1-A degradation was a two-segment curve
in accumulated wear, penalty(a) = k*a until W seconds had been lost and then
k*m per lap, with the cliff pooled per compound. All three compounds' cliff
parameters ended up hand-set, because lap times cannot show a cliff nobody
drives into.

Degradation now comes from Tyre_model/, fitted per track and compound:

    D(a) = b1*a + b2*a^2 + gamma * max(0, a - tau)^2

b1 and b2 are measured from the lap-time profiles. tau and gamma come from
somewhere the previous version never looked: when teams actually stop. A team
pitting off softs at lap 19 knows something about that tyre its lap times were
never allowed to show, and treating the stint's end as an observation - with
the stints that ran to the flag correctly censored - is what identifies the
cliff. At Zandvoort the lap times alone say SOFT wears at a fifth of HARD's
rate; the stopping ages say 19, 23 and 30 laps, which is the physical order.

Three things follow from the scope of that model:

  - It is deterministic. Same track, compound and age, same answer, always.
  - Driver, car, temperature and traffic are out. Every car on the same
    compound at the same age loses the same time, so deg_index no longer
    scales the curve and strategies are no longer priced per driver.
  - No silent extrapolation. Each curve carries the age range it was measured
    over, and ENFORCE_STINT_CAP holds cars inside it.

Curves are loaded once, before the first simulation, and read from memory
thereafter - there is no HTTP call and no fitting inside the lap loop. The
same numbers are available over HTTP from Tyre_model/tyre_api.py for anything
outside this process.

Strategy costs are re-estimated here rather than taken from pit_strategies.csv,
because those were computed against a flat coefficient and cannot see a cliff.
A plan that runs a stint past the drop-off has to be priced as the mistake it
is, or the softmax keeps offering it.

Still missing: red flags and a proper first-lap model.
"""

import contextlib
import os
import sys
import webbrowser

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Simülasyon.diagnostics import build_report, build_stints, find_representative
from Simülasyon.fuel_effect import get_track_params
from Simülasyon.target_race import TARGET_EVENT, TRACK_ALIASES
from Simülasyon.track_deg import get_track_deg
from Simülasyon.tracks import (get_track_racing, DIRTY_AIR_RANGE, MIN_GAP,
                               ATTACK_GAP, HELD_GAP)

from Simülasyon.Tyre_model.tyre_curve import AGE_OFFSET
from Simülasyon.Tyre_model.tyre_store import UnknownProfile
from Simülasyon.Tyre_model.tyre_store import load as load_tyre_store

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'output')

SEASON = 2026
# TARGET_EVENT and TRACK_ALIASES now come from target_race.py - change
# TARGET_RACE there to switch which Grand Prix this predicts, then re-run
# fetch.py so the laps/grid split matches.

N_SIMS = 10_000
RANDOM_SEED = 42

# Pole time/driver are read from data/f1_{SEASON}_poles.csv, matched against
# TRACK_ALIASES the same way everything else is - see resolve_pole() below.
# These two are only the fallback for a circuit whose qualifying has not run
# yet, so the pipeline still has something to fuel a first pass with.
POLE_TIME_FALLBACK = 71.163
POLE_DRIVER_FALLBACK = 'NOR'

# The grid is read from data/f1_2026_grid.csv, written by fetch.py.
# Fill this in only to override it, fastest first.
GRID = []

IS_WET = False

FALLBACK_SIGMA = 0.30
FALLBACK_PIT_LOSS = 21.5
FALLBACK_OFFSETS = {'SOFT': -0.35, 'MEDIUM': 0.0, 'HARD': 0.30}
FALLBACK_SC = {'sc_lambda': 0.6, 'sc_duration': 4.0,
               'vsc_lambda': 0.5, 'vsc_duration': 1.5,
               'rf_lambda': 0.13}     # calendar rate, 12 red flags in 92 races
FALLBACK_GRID_GAP = 0.55     # seconds per grid slot at the end of lap 1
FALLBACK_START_SIGMA = 1.9   # positions gained or lost at the start

# --- tyre model (v1.2) ---
# Where the fitted curves are read from. Loaded once in load_track(), before
# any simulation starts, and held in memory for the rest of the run.
TYRE_PARAMS_PATH = os.path.join(DATA_DIR, 'tyre_curve_params.json')

# Strategy costs are rebuilt from the tyre curve. The numbers in
# pit_strategies.csv were estimated against a flat coefficient, so they price
# a 30-lap soft stint as merely slow rather than ruinous.
RECOST_STRATEGIES = True

# A plan whose stint runs past the age range its curve was measured over is
# dropped before the softmax sees it. The range comes from stint_limits.py via
# the fit, measured over 2022-2025, and the small tolerance allows for a
# slightly longer race distance rather than for optimism about the tyre.
STINT_OVERRUN_TOLERANCE = 1.05   # x the curve's max_age

# The same limit is enforced during the race: a car reaching it pits, whatever
# the plan or the marginal-cost rule says. This is load-bearing, not a safety
# net. The curve is only fitted over ages that were actually run, so asking it
# about lap 45 of a soft stint is asking it to invent a number - and the fit
# refuses. Without this the simulation would hit that refusal mid-race.
ENFORCE_STINT_CAP = True

# The measured start_sigma averages the whole field, and the field is not one
# thing: the front row launches into clean air on a single line, while the
# midfield funnels into the first corner three abreast. Applying the field-wide
# figure to everyone had pole losing the lead about half the time. Front of the
# grid gets this share of the chaos, the back gets all of it.
START_FRONT_STABILITY = 0.35

# The first lap is its own event - launch, slipstream to turn one, contact -
# and none of it is the steady-state pace model used for the other laps. Until
# that gets modelled properly the grid simply holds: cars still leave with the
# time gaps their slots give them, so the race is not restarted from zero, but
# nobody changes place on lap one. Set False to let the start play out under the
# normal overtaking rules.
LOCK_START_ORDER = True

# neutralization behaviour
SC_LAP_PENALTY = 15.0
VSC_LAP_PENALTY = 8.0
SC_PIT_DISCOUNT = 0.50
VSC_PIT_DISCOUNT = 0.65
WET_LAMBDA_MULTIPLIER = 1.8
PIT_WINDOW_TOLERANCE = 8
MIN_STINT_BEFORE_PIT = 5

# Under a safety car or VSC every car runs the same lap: nobody is racing, so
# nobody gains. Until now the flat penalty was added on top of the normal lap,
# which left delta, noise and tyre wear still running - a quick car kept
# banking time behind the safety car and then either passed under yellow or
# slingshotted past the moment the race went green. Freezing the lap time
# freezes the gaps, and no pass can happen without one.
#
NEUTRAL_FREEZES_GAPS = True

# --- SC bunching (v1.2) ---
# Freezing the gaps was only half of what a safety car does. The other half is
# that it closes the field into a queue, which hands back whatever lead the
# man in front had built. A driver 25 s clear is level again at the restart,
# and that is one of the largest sources of race-to-race variance there is.
#
# Only the full safety car bunches. A VSC is a delta-time procedure run at
# racing spacing - the field stays where it is - so nothing here touches it.
SC_BUNCHING = True
SC_QUEUE_GAP = 0.5           # seconds between cars once the queue has formed

# A safety car is not withdrawn after one lap: it has to pick the leader up,
# and the pack has to close before the restart. Three laps is the floor.
MIN_SC_DURATION = 3
MIN_VSC_DURATION = 1         # a VSC really can be one lap, so this is a no-op

# When neutralizations happen. The start lap was drawn uniformly, which spread
# them evenly over the race; in practice they cluster after the opening laps
# have settled and before the run to the flag.
SC_TIMING_MEAN_FRACTION = 0.60   # of race distance
SC_TIMING_STD_FRACTION = 0.20

# --- red flag (v1.2) ---
# The end of the bunching argument. A safety car hands back the leader's gap;
# a red flag deletes it outright, gives everyone a free set of tyres, and
# restarts the race from a standing order. It is the largest single thing that
# can happen to a race result and it was not modelled at all.
#
# One per race at most, drawn as a Bernoulli trial rather than a Poisson count:
# a second red flag in the same race is rare enough to be out of scope, and
# counting them would let one simulation stop the race three times.
RED_FLAG_ENABLED = True
RF_PROB_CAP = 0.15           # per race, no circuit above this
RF_PROB_FLOOR = 0.01         # per race, no circuit exempt either

# Timing reuses the safety car's distribution - a red flag is thrown for the
# same kind of incident, so there is no reason to give it its own shape.

# What a free tyre change is worth depends on when it lands. In the opening
# laps the tyre on the car is nearly new, and at the end there is not enough
# race left to use a different compound; both cases just refit what is already
# fitted. In between, the stop is worth taking as the next step of the plan.
RF_EARLY_LAPS = 5
RF_LATE_LAPS = 10

# The street-circuit split is measured but deliberately not used.
#
# The argument was that a street circuit has no run-off, so an incident that
# is a safety car elsewhere has to stop the race - which would put the
# red-flag rate up. neutralization.py measures the opposite: 2 red flags in 19
# street races against 10 in 73 permanent ones, or 0.77x, and with counts that
# small the two are not distinguishable anyway. The rate here is therefore
# pooled over the whole calendar, the same partial pooling sc_lambda and
# vsc_lambda already get. neutralization.py still prints both categories, so
# if more seasons change the picture this is one edit away from coming back.

# Zandvoort's SC rate is measured from four races. At lambda 1.33 that puts a
# safety car in three quarters of all simulations, which is the small sample
# talking rather than the circuit. The measured rate is pulled toward the
# calendar average with this many pseudo-races of weight, so a track with four
# real observations keeps about 40% of its own number. Set to 0 to use the raw
# measurement.
SC_SHRINK_RACES = 6.0
SC_DEFAULT_RACES = 4.0       # assumed when the CSV does not say

# The same treatment for the compound pace offsets, which are measured from
# the same handful of dry races and come out just as noisy. See
# shrink_offsets. Set to 0 to use each circuit's raw measurement.
OFFSET_SHRINK_RACES = 6.0

# Which compounds get pulled toward the calendar. MEDIUM is the reference and
# cannot move.
#
# HARD is the clear case: three dry races at Zandvoort put it 0.159 s/lap
# quicker than the medium, and a hard tyre is not quicker than a medium.
#
# SOFT is not, and this is the dial to reach for first. Its calendar median is
# -0.019 s/lap, which would have the soft barely quicker than the medium - far
# smaller than the real gap - so shrinking toward it damps a number that was
# not obviously wrong. Worse, the soft is already paying for its speed through
# the panel wear curve, so shrinking the pace as well charges it twice: with
# both applied the strategy space collapses onto a single plan (99.9% on
# medium-hard), where the pace left alone keeps a genuine contest (82%, with
# the soft three-stopper 1.7 s behind).
OFFSET_SHRINK_COMPOUNDS = ('HARD',)

# --- lap-to-lap noise ---
# White noise was wrong. Independent draws over 72 laps average out almost
# completely, which is why the leader kept coming out at 98% to win: the model
# had no way to have a bad afternoon. Real form is sticky - a driver off the
# pace in one stint tends to stay off it - so the noise carries over:
#
#     e_t = rho * e_{t-1} + sqrt(1 - rho^2) * sigma * z
#
# The scaling keeps each lap's spread at sigma while the race total widens by
# roughly sqrt((1+rho)/(1-rho)), about double at rho = 0.6.
NOISE_AUTOCORR = 0.60

# --- retirements ---
# Every car finishing is a real omission: a retirement rearranges the whole
# order behind it and is a large share of what makes race outcomes uncertain.
# A flat per-car rate, deliberately low for now; reliability differences between
# teams are a separate model.
DNF_ENABLED = True
DNF_RATE_PER_RACE = 0.06     # chance a given car fails to see the flag

# --- track evolution ---
# Measured by track_evolution.py and until now unused. Centred on the race
# midpoint for the same reason as fuel: delta was measured from race medians
# that already contain the average state of the track, so an uncentred term
# would count it twice.
EVOLUTION_ENABLED = True

AUTO_PASS_MARGIN = 1.20      # s of time advantage that no defence can hold off
ORDER_SWEEPS = 12            # passes over the order, so a pit stop can drop a car far
POOLED_THRESHOLD = 1.60      # s/lap advantage at 50% per-lap pass odds
POOLED_SCALE = 0.52          # how sharp that transition is
REFERENCE_ADVANTAGE = 0.50   # advantage the measured base rate corresponds to
MAX_PASS_PROB = 0.85

# --- how close the follower is (v1.5) ---
# Until now a car 0.2 s behind and a car 0.95 s behind rolled the same dice,
# because pass_probability read only the pace advantage. Inside the very
# window this model rolls in, the measured rate runs:
#
#     gap < 0.25 s   0.691        0.50 - 0.75 s   0.154
#     0.25 - 0.50    0.335        0.75 - 1.00 s   0.097
#
# and the gap is almost uncorrelated with the pace advantage (-0.12), so it
# is separate information rather than a restatement of it. Adding it to the
# fit takes the pseudo-R2 from 0.050 to 0.164 on 4,490 laps of 2022-25 -
# more than twice what the advantage explains on its own, with a likelihood
# ratio of 546 on one degree of freedom.
#
# This is also what makes the per-lap roll honest about pursuits. A car that
# hangs on at 0.9 s for ten laps was previously handed ten full-strength
# chances; now each of those laps is priced at what 0.9 s is actually worth,
# and only closing up buys a real one.
#
# Measured by overtaking.py, which prints both numbers.
# Both coefficients come from the SAME fit and have to be used together.
# Taking the gap term from the joint fit while leaving the advantage slope at
# 1/POOLED_SCALE = 1.92 mixes two different models, and the mixture is not a
# model of anything: a car with no pace advantage at all, sitting 0.2 s back,
# came out at a 42% chance of passing, and the race produced 106 overtakes
# where the circuit's own history says 40.
#
#     joint fit on 4,490 laps:  advantage +1.325,  gap -4.074
#
# The advantage response is softer than POOLED_SCALE implied, which is what
# happens when the thing it was silently standing in for - being close -
# gets its own term.
GAP_IN_PASS_MODEL = True
GAP_ADV_COEF = 1.325         # logit per s/lap of pace advantage
GAP_COEF = 4.074             # logit per second of gap
REFERENCE_GAP = 0.60         # mean gap inside ATTACK_GAP, the anchoring point

# The fit only ever saw a following car inside a second with at least
# MIN_ADVANTAGE of pace in hand. Applied everywhere, it is an extrapolation,
# and a wild one - this model's gap is not the measurement's gap.
#
# overtaking.py reads a gap between consecutive cars in the running order, so
# it is positive by construction. run_simulation carries an unsorted
# cumulative-time array on purpose, so its gap goes negative whenever a car
# has gained time it has not yet been given the place for. Over half of the
# laps that reach the attacking branch have less than MIN_ADVANTAGE of pace,
# and the median gap there is negative - a population the coefficients were
# never fitted on. Handing those laps the gap term produced 118 overtakes at
# a circuit whose own history says forty.
#
# Inside the fitted domain the term is used; outside it the model falls back
# to the advantage-only form it had before, anchored on the wide-window rate
# it was always anchored on. That keeps the new term to the situations it
# was measured in and leaves everything else exactly as it was.
GAP_MODEL_MIN_ADVANTAGE = 0.15   # the conditioning overtaking.py fitted under

# The base rate the threshold is anchored to has to be measured on the same
# window. overtaking.py counts a chance from CLOSE_GAP = 1.5 s, but nothing
# is ever rolled beyond ATTACK_GAP = 1.0, and 28% of those laps sit in the
# band between. Passes are much rarer out there, so the wide-window rate
# understates the narrow window by about a quarter - 0.172 against 0.222
# pooled, a median factor of 1.27 per circuit. pass_rate_near is the same
# measurement restricted to the window this model uses.
USE_NEAR_GAP_RATE = True

# --- team overtaking strength (v1.5) ---
# overtake_speed.py places each 2026 team on two axes built from attacking
# ability, defending ability and qualifying speed trap. PC1 carries 58% of
# that and reads as overall wheel-to-wheel strength: Mercedes and Red Bull at
# one end, Aston Martin at the other.
#
# The scale is not a preference. Regressing the fitted attack coefficient on
# PC1 gives 0.232 logit per PC1 unit, and a permutation test - team labels
# shuffled within each race, 200 draws - says only about half of the spread
# survives as signal (observed 0.500 against 0.351 from shuffled labels).
# Keeping that half:
#
#     0.232 * 0.51 = 0.118 logit per PC1 unit
#
# which puts the best and worst team 1.55x apart in per-lap pass chance at
# equal pace, against 3.7x for the raw coefficients. PC1 is mean-centred by
# construction, so this redistributes rather than making the field faster.
#
# Only the attacker's team is read. The defending half of the measurement
# failed its own permutation test outright (p = 0.395), so applying it would
# be dressing up noise.
TEAM_PASS_ENABLED = True
TEAM_PASS_COEF = 0.118       # logit per PC1 unit
TEAM_PASS_CAP = 0.40         # logit, either way

STRATEGY_TEMPERATURE = 1.0
MAX_STRATEGIES = 8

# --- reactive pit decision (v1.1) ---
# The planned pit lap is a suggestion now rather than an order. Inside a band
# around it the car asks, every lap, whether the next lap on this tyre costs
# more than the lap it would displace on a fresh one:
#
#     stay = penalty(current compound, age + 1) + rest of the race, a lap shorter
#     pit  = rest of the race, run on the planned compounds
#
# The pit loss sits on both sides - the stop happens either way, only the
# timing is in question - so it cancels, and the decision reduces to a
# comparison of tyre curves. The curve rises monotonically, so the sign flips
# once and checking the margin each lap finds the crossing without searching a
# window.
#
# The band stops the rule chattering. Lap-to-lap noise can push the margin
# either way, and unbounded a car would reconsider its strategy on lap three.
REACTIVE_PIT = True
PIT_DECISION_BAND = 5        # laps either side of the planned stop

# Strategies used to be priced per driver, scaling the tyre curve by that
# driver's deg_index so someone hard on tyres reached the cliff sooner and was
# offered less of a soft-heavy plan.
#
# The v1.2 tyre model has no driver term - every car on the same compound at
# the same age loses the same time - so there is nothing left to price
# differently and every column of the cost matrix would be identical. Turning
# it off is the honest form of that: a per-driver spread of exactly zero
# reported as a per-driver model is worse than no per-driver model.
#
# deg.py still measures deg_index and deg_by_driver.csv is still written. It
# feeds nothing here until a version puts the driver back in scope.
PER_DRIVER_STRATEGY = False

# --- track affinity (v0.7) ---
# Affinity is measured in finishing positions, so it needs a rate to become lap
# time. Kept deliberately small: the signal comes from a handful of races per
# driver per circuit and carries strategy and luck along with any real
# suitability, so it should nudge the order, never rewrite it.
AFFINITY_ENABLED = True
AFFINITY_TEAM_SHARE = 0.40          # rest comes from the driver's own record

# Affinity enters as a fraction of the lap, not a fixed number of seconds.
#
# The old form said one position of affinity is worth 0.020 s/lap at Monaco
# and 0.020 s/lap at Spa. It is not. A Spa lap is 2.15 times a Monaco lap by
# distance, and an advantage that comes from the car - downforce level, power
# unit deployment, how the thing rides kerbs - is spent over the whole lap, so
# it buys proportionally more time on a long one. Expressing it as a
# percentage is what makes the same affinity mean the same thing everywhere.
#
# The rate is pinned so this circuit reproduces the old number: 0.020 s on a
# 67 s reference lap is 0.030% of it. Zandvoort therefore behaves as before,
# and a longer circuit now scales up instead of being quietly understated.
AFFINITY_PCT_PER_POSITION = 0.00030   # fraction of lap time per position
AFFINITY_CAP_PCT = 0.00180            # 0.12 s on the same reference lap
AFFINITY_REFERENCE_LAP = 67.0         # only documents where the rate came from

# Where the team half of the affinity comes from.
#
# It used to come from track_affinity.csv, which measures 2022-2025. Those are
# different cars under different regulations, and asked directly whether that
# history predicts 2026, the answer was almost no:
#
#     2022-25 team affinity -> 2026 race affinity   corr +0.076, skill +0.6%
#     2026 qualifying       -> 2026 race affinity   corr +0.332, skill +11.0%
#
# team_affinity.py builds the second one: team-level, both cars averaged,
# retirements censored to the last position the car actually held rather than
# its classification, and shrunk by a factor fitted out of sample. The driver
# half still comes from the historical table, where it measures +1.5% - thin,
# but it is the only per-driver circuit record there is.
AFFINITY_TEAM_2026 = True           # team term from team_affinity_2026.csv

# The historical term leans on two or three races per driver per circuit from
# cars that no longer exist, which makes it swing harder than the evidence
# supports. Half the weight now goes to a 2026-only signal built the same way:
#
#     quali affinity = driver's 2026 average qualifying position
#                    - their qualifying position at this circuit
#
# Same shape, current car, and no leakage: qualifying runs before the race.
AFFINITY_QUALI_SHARE = 0.50

OPEN_REPORT = True           # open the HTML report in a browser when done

COMPOUND_NAMES = ['SOFT', 'MEDIUM', 'HARD']

# --- input loading ----------------------------------------------------------


def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def match_track(df, column='Race'):
    if df is None or df.empty:
        return None
    key = df[column].astype(str).str.lower()
    hit = df[key.apply(lambda s: any(a in s for a in TRACK_ALIASES))]
    return hit if not hit.empty else None


def resolve_pole():
    """
    This circuit's pole time and driver, read from f1_{SEASON}_poles.csv.

    fetch.py records a pole for every race it can reach qualifying for,
    target included - qualifying happens before the race, so reading it here
    is not leakage. A circuit whose qualifying has not run yet (predicting a
    future round ahead of its own weekend) has no row, and the fallback
    constants carry the pipeline until it does.
    """
    table = match_track(read_csv(f'f1_{SEASON}_poles.csv'))
    if table is not None and 'pole_time' in table.columns:
        row = table.iloc[0]
        if pd.notna(row['pole_time']):
            return float(row['pole_time']), str(row['pole_driver'])
    print(f'[pole] no qualifying on file for {TARGET_EVENT}; '
          f'using the fallback {POLE_TIME_FALLBACK:.3f} s / {POLE_DRIVER_FALLBACK}.')
    return POLE_TIME_FALLBACK, POLE_DRIVER_FALLBACK


POLE_TIME, POLE_DRIVER = resolve_pole()


def load_drivers():
    pace = read_csv(f'driver_pace_{SEASON}.csv')
    if pace is None:
        raise SystemExit('driver_pace CSV missing. Run clean.py first.')
    pace = pace.copy()
    pace['sigma'] = pace['sigma'].fillna(FALLBACK_SIGMA)

    # deg_by_driver.csv is deliberately not read. The v1.2 tyre curve has no
    # driver term, so scaling it by deg_index would be inventing a difference
    # the model does not carry. deg.py still writes the file for whenever the
    # driver comes back into scope.
    pace = apply_affinity(pace)

    pace = pace.sort_values('delta').reset_index(drop=True)

    # --- starting grid, 1 = pole -------------------------------------------
    order = list(GRID)
    source = 'manual override'

    if not order:
        grid_csv = read_csv(f'f1_{SEASON}_grid.csv')
        if grid_csv is not None and 'GridPosition' in grid_csv.columns:
            grid_csv = grid_csv.sort_values('GridPosition')
            order = grid_csv['Driver'].astype(str).tolist()
            source = f'f1_{SEASON}_grid.csv'

    if order:
        slot = {d: i + 1 for i, d in enumerate(order)}
        # anyone not on the grid sheet lines up at the back, in pace order
        missing = [d for d in pace['Driver'] if d not in slot]
        for i, d in enumerate(missing):
            slot[d] = len(order) + i + 1
        if missing:
            source += f' (+{len(missing)} appended: {", ".join(missing)})'
        pace['grid'] = pace['Driver'].map(slot)
    else:
        # pace order makes the grid a copy of delta, which double counts pace
        # and turns the model into the naive "finish where you started" baseline
        pace['grid'] = np.arange(1, len(pace) + 1)
        source = 'pace order (PLACEHOLDER)'

    pace = pace.sort_values('grid').reset_index(drop=True)
    pace.attrs['grid_source'] = source
    return pace


def quali_affinity(pace):
    """
    2026-only affinity: how much better a driver qualified here than they
    usually do this season, in positions. Positive means the circuit suits them.
    """
    quali = read_csv(f'f1_{SEASON}_quali.csv')
    if quali is None or 'quali_pos' not in quali.columns:
        return None

    quali = quali.copy()
    quali['quali_pos'] = pd.to_numeric(quali['quali_pos'], errors='coerce')
    quali = quali[quali['quali_pos'].notna()]

    key = quali['Race'].astype(str).str.lower()
    is_here = key.apply(lambda s: any(a in s for a in TRACK_ALIASES))

    here = quali[is_here]
    elsewhere = quali[~is_here]
    if here.empty or elsewhere.empty:
        return None

    season_avg = elsewhere.groupby('Driver')['quali_pos'].mean()
    here_pos = here.groupby('Driver')['quali_pos'].min()

    aff = (season_avg - here_pos).dropna()
    return pace['Driver'].map(aff.to_dict())


def team_affinity_2026(pace):
    """
    This circuit's team affinity, measured from 2026 alone.

    Returns None when the file is missing or says nothing about this circuit,
    which leaves the caller on the 2022-25 table rather than on zeros - a
    silent zero would look like "no circuit suits anyone in particular", which
    is a claim, not an absence.

    Build it with:  python -m Simülasyon.team_affinity
    """
    if not AFFINITY_TEAM_2026:
        return None, 'disabled'

    table = match_track(read_csv(f'team_affinity_{SEASON}.csv'))
    if table is None or 'affinity' not in table.columns:
        return None, 'missing'

    values = table.set_index('Team')['affinity'].to_dict()
    term = pace['Team'].map(values)
    hit = int(term.notna().sum())
    if hit == 0:
        return None, 'no team matched'

    shrink = float(table['shrinkage'].iloc[0]) if 'shrinkage' in table else 0.0
    note = f'2026 team ({hit}/{len(pace)} cars, shrinkage {shrink:.2f})'
    return term.fillna(0.0), note


def apply_affinity(pace):
    """
    Shifts delta by how well this circuit suits each driver, blending a
    historical record with a 2026-only qualifying signal.

    The historical half uses the driver's and the team's results here against
    their own season averages, so it measures suitability rather than who is
    quick in general. The 2026 half asks the same question of this season's car.
    The result is centred and capped: affinity redistributes the field a little,
    it never rewrites it.
    """
    pace = pace.copy()
    pace['affinity'] = 0.0
    pace['affinity_pct'] = 0.0
    pace['affinity_sec'] = 0.0
    pace.attrs['affinity_team_source'] = 'off'

    if not AFFINITY_ENABLED:
        return pace

    hist = None
    team_source = 'none'
    table = match_track(read_csv('track_affinity.csv'))
    if table is not None and 'entity' in table.columns:
        drv = table[table['kind'] == 'driver'].set_index('entity')['affinity'].to_dict()
        team_term, team_source = team_affinity_2026(pace)
        if team_term is None:
            tm = (table[table['kind'] == 'team']
                  .set_index('entity')['affinity'].to_dict())
            team_term = pace['Team'].map(tm).fillna(0.0)
            team_source = '2022-25 history'
        hist = ((1 - AFFINITY_TEAM_SHARE) * pace['Driver'].map(drv).fillna(0.0)
                + AFFINITY_TEAM_SHARE * team_term)

    quali = quali_affinity(pace)

    if hist is None and quali is None:
        print('[affinity] nothing available for this circuit, skipping.')
        return pace

    if hist is None:
        blended, source = quali.fillna(0.0), '2026 qualifying only'
    elif quali is None:
        blended, source = hist, 'historical only'
    else:
        q = quali.fillna(hist)          # no 2026 record: lean on history
        blended = (1 - AFFINITY_QUALI_SHARE) * hist + AFFINITY_QUALI_SHARE * q
        source = (f'{1 - AFFINITY_QUALI_SHARE:.0%} historical / '
                  f'{AFFINITY_QUALI_SHARE:.0%} 2026 qualifying')

    blended = blended - blended.mean()

    # positions -> fraction of the lap -> seconds at this circuit's lap time.
    # Going through the fraction rather than straight to seconds is the whole
    # point: the cap is a percentage too, so it scales with the circuit
    # instead of clipping long laps harder than short ones.
    fraction = np.clip(blended * AFFINITY_PCT_PER_POSITION,
                       -AFFINITY_CAP_PCT, AFFINITY_CAP_PCT)
    seconds = fraction * POLE_TIME

    pace['affinity'] = blended.round(2)
    pace['affinity_pct'] = (fraction * 100).round(4)
    pace['affinity_sec'] = seconds.round(3)
    pace['delta'] = pace['delta'] - seconds
    pace['delta'] = pace['delta'] - pace['delta'].min()
    pace.attrs['affinity_source'] = source
    pace.attrs['affinity_team_source'] = team_source
    return pace


# --- tyre curve -------------------------------------------------------------


def resolve_tyre_curves(store):
    """
    This circuit's three curves, looked up once.

    TARGET_EVENT is the season-entry name the rest of the project uses
    ('Netherlands'), while the parameter file is keyed on FastF1 event names
    ('Dutch Grand Prix'). tyre_store matches across the two, and raises when it
    cannot: a missing circuit has to stop the run rather than quietly hand
    Zandvoort the calendar average, which is a different circuit's answer
    wearing this one's name.
    """
    curves = {}
    for compound in COMPOUND_NAMES:
        for name in [TARGET_EVENT] + TRACK_ALIASES:
            try:
                curves[compound] = store.get(name, compound)
                break
            except UnknownProfile:
                continue
        else:
            raise SystemExit(
                f'No {compound} curve for {TARGET_EVENT}. Build one with:\n'
                f'    python Simülasyon/Tyre_model/fit_tyre_curve.py')
    return curves


def max_stint_laps(curve):
    """
    Longest stint this curve can answer for, as a stint length.

    The curve is indexed on its own age, which runs AGE_OFFSET behind
    TyreLife, and a stint of length L takes the tyre from TyreLife 1 to L. So
    the two differ by exactly the offset, and mixing them up is worth two laps
    of stint every time.
    """
    return curve.max_age + AGE_OFFSET


def curve_penalty(curve, ages):
    """
    D(a) for an array of curve ages, evaluated straight from the coefficients.

    TyreCurve.loss refuses an age past max_age, which is right for a query from
    outside but wrong inside the lap loop: the caller has already clamped, and
    raising per lap per car per simulation would cost more than the arithmetic.
    The clamping is the contract - ENFORCE_STINT_CAP keeps ages inside the
    measured range and this evaluates what the curve says there.
    """
    a = np.asarray(ages, dtype=float)
    d = curve.b1 * a + curve.b2 * a * a
    if curve.tau is not None and curve.gamma > 0:
        over = np.maximum(a - curve.tau, 0.0)
        d = d + curve.gamma * over * over
    return d


def stint_tyre_cost(tyre, compound, length):
    """
    Seconds a stint loses to tyre wear alone, summed lap by lap.

    Ages beyond the curve's measured range are clamped rather than
    extrapolated, so an overrunning plan is priced as flat-out-bad from the cap
    onwards instead of being handed an invented cliff. recost_strategies drops
    those plans anyway; this only has to not lie while it works that out.
    """
    curve = tyre[compound]
    n = max(int(round(length)), 1)
    ages = np.clip(np.arange(1, n + 1) - AGE_OFFSET, 0, curve.max_age)
    return float(curve_penalty(curve, ages).sum())


def build_cum_table(track, n_laps):
    """
    Accumulated tyre loss, indexed by compound and tyre age.

        CUM[c, a] = seconds a compound-c tyre has cost by the time it is
                    `a` laps old, where `a` is TyreLife

    Built once, before the race starts. Every question the pit rule asks -
    what does one more lap cost, what would this stint cost if it ran to lap
    forty - is then the difference of two lookups instead of a loop.

    There is no driver axis any more. The v1.2 curve has no driver term, so
    all twenty-two columns were identical and the table was twenty-two times
    larger than the thing it held.
    """
    max_age = n_laps + 2
    tyre_life = np.arange(0, max_age + 1)

    cum = np.zeros((3, max_age + 1))
    for ci, comp in enumerate(COMPOUND_NAMES):
        curve = track['tyre'][comp]
        ages = np.clip(tyre_life - AGE_OFFSET, 0, curve.max_age)
        pen = curve_penalty(curve, ages)
        pen[0] = 0.0
        cum[ci] = np.cumsum(pen)
    return cum


def recost_strategies(strategies, track):
    """
    Prices every plan against the measured tyre curve.

    est_cost in pit_strategies.csv came from the flat coefficient, where a long
    stint costs a little more than a short one and nothing worse. With a cliff
    the same plan can cost thirty seconds it did not before, and the softmax
    has to see that or it keeps handing out weight to strategies no team would
    run. Costs are rebased so the cheapest plan sits at zero, which keeps
    STRATEGY_TEMPERATURE meaning what it meant.
    """
    tyre, n_laps = track['tyre'], track['n_laps']
    kept, dropped = [], []

    for s in strategies:
        plan = plan_stints(s, n_laps)
        cost = track['pit_loss'] * (len(plan) - 1)
        overrun = False
        for comp, L in zip(s['sequence'], plan):
            cost += stint_tyre_cost(tyre, comp, L)
            cost += track['offsets'].get(comp, 0.0) * L
            if L > max_stint_laps(tyre[comp]) * STINT_OVERRUN_TOLERANCE:
                overrun = True
        s['est_cost_model'] = cost
        s['plan'] = plan
        if overrun:
            s['drop_reason'] = 'stint past compound life'
            dropped.append(s)
        else:
            kept.append(s)

    if not kept:                      # never leave the sim with nothing to run
        kept, dropped = strategies, []

    base = min(s['est_cost_model'] for s in kept)
    for s in kept:
        s['est_cost'] = s['est_cost_model'] - base

    # the shortlist is cut here, on the prices this model produced, rather
    # than upstream on the ones it replaced
    kept.sort(key=lambda s: s['est_cost'])
    if len(kept) > MAX_STRATEGIES:
        for s in kept[MAX_STRATEGIES:]:
            s['drop_reason'] = 'too expensive'
        dropped = dropped + kept[MAX_STRATEGIES:]
        kept = kept[:MAX_STRATEGIES]
    return kept, dropped


def shrink_neutralization(sc, source):
    """
    Pulls this circuit's SC and VSC rates toward the calendar average.

    A rate estimated from four races is mostly noise. Partial pooling keeps
    the circuit's own character where the evidence supports it and borrows the
    rest from every other track:

        lambda = (n * lambda_track + k0 * lambda_calendar) / (n + k0)

    Durations are left alone. How long a safety car lasts is a property of the
    procedure rather than the circuit, and it varies far less than how often
    one comes out.
    """
    if SC_SHRINK_RACES <= 0:
        return sc, source

    table = read_csv('neutralization.csv')
    if table is None or table.empty:
        return sc, source

    out = dict(sc)
    n = SC_DEFAULT_RACES
    for col in ('n_races', 'races', 'n_events', 'n'):
        if col in table.columns:
            here = match_track(table)
            if here is not None and pd.notna(here.iloc[0][col]):
                n = float(here.iloc[0][col])
            break

    notes = []
    for key in ('sc_lambda', 'vsc_lambda'):
        if key not in table.columns:
            continue
        calendar = float(pd.to_numeric(table[key], errors='coerce').median())
        raw = float(sc[key])
        pooled = (n * raw + SC_SHRINK_RACES * calendar) / (n + SC_SHRINK_RACES)
        out[key] = pooled
        notes.append(f'{key} {raw:.2f}->{pooled:.2f}')

    # The red flag rate needs the calendar total, not the calendar median.
    # Eighteen of twenty-five circuits have never had one in this window, so
    # the median across circuits is exactly zero - shrinking toward that would
    # drag every track to nothing and the feature would never fire. The pooled
    # rate, every red flag over every race, is the honest prior: a circuit
    # with none in four years is one that has not had one yet, not one where
    # they cannot happen.
    if 'rf_lambda' in table.columns:
        if {'rf_count', 'rf_races'} <= set(table.columns):
            total_rf = float(pd.to_numeric(table['rf_count'], errors='coerce').sum())
            total_races = float(pd.to_numeric(table['rf_races'], errors='coerce').sum())
            calendar = total_rf / max(total_races, 1.0)
        else:
            calendar = float(pd.to_numeric(table['rf_lambda'], errors='coerce').mean())

        raw = float(sc.get('rf_lambda', calendar))
        pooled = (n * raw + SC_SHRINK_RACES * calendar) / (n + SC_SHRINK_RACES)
        out['rf_lambda'] = float(np.clip(pooled, RF_PROB_FLOOR, RF_PROB_CAP))
        notes.append(f'rf_lambda {raw:.3f}->{out["rf_lambda"]:.3f}')

    if notes:
        source = (f'{source}, shrunk toward calendar '
                  f'({n:.0f} races vs {SC_SHRINK_RACES:.0f} prior): '
                  + ', '.join(notes))
    return out, source


def shrink_offsets(offsets, here, calendar, source):
    """
    Pulls this circuit's compound pace offsets toward the calendar median.

    Same partial pooling as shrink_neutralization, and for the same reason:

        off = (n * off_track + k0 * off_calendar) / (n + k0)

    Zandvoort has three dry races in the window. Three is not enough to trust
    a per-circuit compound offset, and the one it produced said the hard tyre
    was quicker than the medium, which is not a thing a hard tyre does.

    MEDIUM is the reference and stays at zero by construction, so only SOFT
    and HARD move.

    The honest caveat: the calendar median for SOFT is -0.019 s/lap, which is
    far smaller than the real soft-medium gap. Shrinking toward it damps the
    soft harder than the evidence warrants, and OFFSET_SHRINK_RACES is the
    dial if that turns out to have gone too far.
    """
    if OFFSET_SHRINK_RACES <= 0 or calendar is None:
        return offsets, source

    n = float(len(here))
    out, notes = dict(offsets), []
    for c in COMPOUND_NAMES:
        col = f'off_{c}'
        if (c == 'MEDIUM' or c not in OFFSET_SHRINK_COMPOUNDS
                or col not in calendar.columns):
            continue
        cal = pd.to_numeric(calendar[col], errors='coerce').median()
        if not np.isfinite(cal):
            continue
        raw = offsets[c]
        pooled = (n * raw + OFFSET_SHRINK_RACES * cal) / (n + OFFSET_SHRINK_RACES)
        out[c] = pooled
        notes.append(f'{c[0]} {raw:+.3f}->{pooled:+.3f}')

    if notes:
        source = (f'{source}, shrunk toward calendar '
                  f'({n:.0f} races vs {OFFSET_SHRINK_RACES:.0f} prior): '
                  + ', '.join(notes))
    return out, source


def load_track():
    params = get_track_params(TARGET_EVENT)
    deg_info = get_track_deg(TARGET_EVENT, base_lap_time=POLE_TIME, use_measured=True)
    racing = get_track_racing(TARGET_EVENT)

    summary = match_track(read_csv('pit_summary.csv'))
    if summary is not None:
        pit_loss = float(summary.iloc[0]['pit_loss'])
        compounds = str(summary.iloc[0]['compounds']).split('-')
    else:
        pit_loss, compounds = FALLBACK_PIT_LOSS, ['SOFT', 'MEDIUM', 'HARD']

    all_scaling = read_csv('compound_scaling.csv')
    scaling = match_track(all_scaling)
    offsets = dict(FALLBACK_OFFSETS)
    offset_source = 'fallback'
    if scaling is not None:
        for c in COMPOUND_NAMES:
            col = f'off_{c}'
            if col in scaling.columns and scaling[col].notna().any():
                offsets[c] = float(scaling[col].median())
        offset_source = f'measured over {len(scaling)} dry races'
        offsets, offset_source = shrink_offsets(
            offsets, scaling, all_scaling, offset_source)

    # --- tyre curve, per compound ------------------------------------------
    # Read once, here, and held for the rest of the run. The lap loop below
    # does 10,000 x 72 x 22 evaluations; anything that touched a file or a
    # socket inside it would dominate everything else the simulation does.
    store = load_tyre_store(TYRE_PARAMS_PATH)
    tyre = resolve_tyre_curves(store)
    tyre_model_version = store.model_version

    neutral = match_track(read_csv('neutralization.csv'))
    if neutral is not None:
        r = neutral.iloc[0]
        sc = {**FALLBACK_SC,
              **{k: float(r[k]) for k in FALLBACK_SC if k in neutral.columns}}
        sc_source = 'measured'
        sc, sc_source = shrink_neutralization(sc, sc_source)
    else:
        sc, sc_source = dict(FALLBACK_SC), 'fallback'

    if IS_WET:
        sc['sc_lambda'] *= WET_LAMBDA_MULTIPLIER
        sc['vsc_lambda'] *= WET_LAMBDA_MULTIPLIER

    over = match_track(read_csv('overtaking.csv'))
    pass_rate_wide = np.nan
    if over is not None and 'pass_rate' in over.columns:
        pass_rate_wide = float(over.iloc[0]['pass_rate'])
        pass_rate = pass_rate_wide
        pass_source = 'measured'

        # the rate measured on the window this model actually rolls in
        near = over.iloc[0].get('pass_rate_near') if USE_NEAR_GAP_RATE else None
        if near is not None and pd.notna(near):
            pass_rate = float(near)
            pass_source = f'measured < {ATTACK_GAP:.2f} s'
    else:
        pass_rate = racing['fallback_pass_rate']
        pass_source = 'fallback'

    evo = match_track(read_csv('track_evolution_agg.csv'))
    if evo is not None and 'evo_rate' in evo.columns and EVOLUTION_ENABLED:
        evo_rate = float(evo.iloc[0]['evo_rate'])
        evo_source = 'measured'
    else:
        evo_rate, evo_source = 0.0, 'off'

    grid_stats = match_track(read_csv('grid_stats.csv'))
    if grid_stats is not None:
        grid_gap = float(grid_stats.iloc[0]['grid_gap'])
        start_sigma = float(grid_stats.iloc[0]['start_sigma'])
        grid_source = 'measured'
    else:
        grid_gap, start_sigma = FALLBACK_GRID_GAP, FALLBACK_START_SIGMA
        grid_source = 'fallback'

    return {
        'n_laps': params['n_laps'],
        'fuel_effect': params['fuel_effect'],
        'deg': deg_info['deg_abs'],
        'deg_source': deg_info['source'],
        'tyre': tyre,
        'tyre_model_version': tyre_model_version,
        'pit_loss': pit_loss,
        'compounds': compounds,
        'offsets': offsets,
        'offset_source': offset_source,
        'sc': sc,
        'sc_source': sc_source,
        'pass_rate': pass_rate,
        'pass_rate_wide': pass_rate_wide,
        'pass_source': pass_source,
        'overtake_difficulty': racing['overtake_difficulty'],
        'dirty_air_penalty': racing['dirty_air_penalty'],
        'grid_gap': grid_gap,
        'start_sigma': start_sigma,
        'grid_source': grid_source,
        'evo_rate': evo_rate,
        'evo_source': evo_source,
    }


def load_strategies(track):
    """
    Every plan this circuit allows, unpruned.

    The shortlist used to be cut here, to the MAX_STRATEGIES cheapest by the
    est_cost column - and then recost_strategies immediately repriced the
    survivors against the tyre curve. So the pruning was done by the model the
    simulation had just replaced, and a plan the new model liked could be
    thrown away before it was ever priced. The one-stop medium-hard at
    Zandvoort sat at rank ten under the old costs and never reached the race,
    while the recosted numbers put it first.

    Selection now happens after recosting instead. This returns the whole
    feasible set and main() cuts it once the plans have been priced properly.
    """
    strat = match_track(read_csv('pit_strategies.csv'))
    if strat is not None and 'within_windows' in strat.columns:
        feasible = strat[strat['within_windows'] == True]
        if not feasible.empty:
            return [{'sequence': str(r['sequence']).split('-'),
                     'lengths': [float(x) for x in str(r['stint_laps']).split('-')],
                     'est_cost': float(r['est_cost'])}
                    for _, r in feasible.iterrows()]

    n = track['n_laps']
    return [
        {'sequence': ['MEDIUM', 'HARD'], 'lengths': [n * 0.45, n * 0.55], 'est_cost': 0.0},
        {'sequence': ['MEDIUM', 'HARD', 'MEDIUM'],
         'lengths': [n / 3, n / 3, n / 3], 'est_cost': 0.0},
    ]


# --- strategy expansion -----------------------------------------------------


def plan_stints(strategy, n_laps):
    lengths = strategy['lengths']
    scale = n_laps / sum(lengths)
    plan = [max(1, int(round(L * scale))) for L in lengths]
    while sum(plan) > n_laps:
        plan[int(np.argmax(plan))] -= 1
    while sum(plan) < n_laps:
        plan[int(np.argmax(plan))] += 1
    return plan


def build_strategy_tables(strategies, n_laps):
    plans = [plan_stints(s, n_laps) for s in strategies]
    max_stints = max(len(p) for p in plans)

    pit_laps = np.full((len(strategies), max_stints), n_laps + 1, dtype=int)
    compound_idx = np.zeros((len(strategies), max_stints), dtype=int)
    stint_len = np.zeros((len(strategies), max_stints), dtype=float)
    n_stints = np.array([len(p) for p in plans])

    for i, (s, plan) in enumerate(zip(strategies, plans)):
        cum = 0
        for j, L in enumerate(plan):
            c = s['sequence'][j]
            compound_idx[i, j] = COMPOUND_NAMES.index(c) if c in COMPOUND_NAMES else 1
            stint_len[i, j] = L
            cum += L
            if j < len(plan) - 1:
                pit_laps[i, j] = cum
    return pit_laps, compound_idx, n_stints, plans, stint_len


def assign_strategies(strategies, n_sims, n_drivers, rng, cost_matrix=None):
    """
    Draws a strategy for every car in every simulation.

    With a cost matrix each driver gets his own distribution, so the softmax
    is run per column. rng.choice takes a single probability vector, hence the
    loop over drivers - twenty-two draws of n_sims each, done once before the
    race starts.
    """
    if cost_matrix is None:
        costs = np.array([s['est_cost'] for s in strategies], dtype=float)
        cost_matrix = np.repeat(costs[:, None], n_drivers, axis=1)

    weights = np.exp(-(cost_matrix - cost_matrix.min(axis=0, keepdims=True))
                     / STRATEGY_TEMPERATURE)
    weights /= weights.sum(axis=0, keepdims=True)

    idx = np.empty((n_sims, n_drivers), dtype=int)
    for d in range(n_drivers):
        idx[:, d] = rng.choice(len(strategies), size=n_sims, p=weights[:, d])
    return idx, weights


# --- neutralization ---------------------------------------------------------


def sample_neutralizations(sc, n_sims, n_laps, rng):
    """
    (n_sims, n_laps): 0 green, 1 VSC, 2 SC, 3 red flag. One draw per race,
    shared by every car in it.

    Two things the first version got wrong. A Poisson duration can come back
    as one lap, and a real safety car cannot: it has to collect the leader and
    let the pack close before it comes in, so each code carries its own floor.
    And the start lap was uniform over the race, which put as many safety cars
    on lap 2 as on lap 40. They cluster nearer the middle instead, so the lap
    is drawn from a normal centred there and clipped to the race.
    """
    state = np.zeros((n_sims, n_laps), dtype=np.int8)
    mean_lap = n_laps * SC_TIMING_MEAN_FRACTION
    std_lap = n_laps * SC_TIMING_STD_FRACTION
    last_start = max(2, n_laps - 2)

    for code, lam_key, dur_key, min_dur in [
            (2, 'sc_lambda', 'sc_duration', MIN_SC_DURATION),
            (1, 'vsc_lambda', 'vsc_duration', MIN_VSC_DURATION)]:
        lam = max(sc[lam_key], 0.0)
        mean_dur = max(sc[dur_key], 1.0)
        counts = rng.poisson(lam, size=n_sims)
        for sim in np.flatnonzero(counts):
            for _ in range(counts[sim]):
                start = int(np.clip(rng.normal(mean_lap, std_lap), 1, last_start))
                dur = max(min_dur, int(rng.poisson(mean_dur)))
                end = min(n_laps, start + dur)
                state[sim, start:end] = np.maximum(state[sim, start:end], code)

    # A red flag is one lap and at most one per race, so it is a coin flip
    # rather than a count. It is written last and overwrites whatever was
    # there: once the race is stopped, whether a safety car happened to be
    # out on that lap stops mattering.
    if RED_FLAG_ENABLED:
        p_rf = float(np.clip(sc.get('rf_lambda', 0.0),
                             RF_PROB_FLOOR, RF_PROB_CAP))
        has_rf = rng.random(n_sims) < p_rf
        rf_lap = np.clip(rng.normal(mean_lap, std_lap, size=n_sims),
                         1, last_start).astype(int)
        for sim in np.flatnonzero(has_rf):
            state[sim, rf_lap[sim]] = 3

    return state


# --- overtaking -------------------------------------------------------------


def track_threshold(base_rate):
    """
    Shifts the pooled logistic so that a typical chase matches the measured rate.

    Scaling the curve by base_rate / sigmoid(reference) was wrong: the divisor is
    small, so a big pace advantage pushed the product past 1 and got clipped. A
    car 3 s/lap quicker ended up passing on 85% of laps when the circuit's
    measured rate was 14.5%. Solving for the threshold instead keeps the shape
    and anchors the level:

        sigmoid((REFERENCE_ADVANTAGE - threshold) / scale) = base_rate
    """
    b = float(np.clip(base_rate, 1e-4, 1 - 1e-4))
    logit = np.log(b / (1 - b))
    return REFERENCE_ADVANTAGE - POOLED_SCALE * logit


def build_pass_model(track):
    """
    The two anchors the lap loop needs, built once.

    Inside the gap fit's domain the level is set by the rate measured on that
    same domain - a car inside ATTACK_GAP with real pace in hand. Outside it
    the old advantage-only form applies, anchored where it always was, on the
    full-window rate. Carrying both is what lets the new term be used only
    where it was measured without disturbing anything else.
    """
    near = track['pass_rate']
    wide = track.get('pass_rate_wide')
    if wide is None or not np.isfinite(wide):
        wide = near

    b = float(np.clip(near, 1e-4, 1 - 1e-4))
    return {
        'intercept': (np.log(b / (1 - b))
                      - (GAP_ADV_COEF * REFERENCE_ADVANTAGE
                         - GAP_COEF * REFERENCE_GAP)),
        'threshold': track_threshold(wide),
    }


def pass_probability(advantage, model, gap=None, team_shift=0.0):
    """
    Per-lap chance a following car gets through.

    Three things move it: how much quicker the follower is, how close it
    actually is, and which team built the car.

    The gap term applies only where overtaking.py fitted it - a positive gap
    inside ATTACK_GAP with at least GAP_MODEL_MIN_ADVANTAGE of pace. Laps
    outside that get the advantage-only form this model used before, so the
    new term never speaks about situations it has not seen.

    `model` is either the dict from build_pass_model or a bare threshold, in
    which case only the old form is available.
    """
    if not isinstance(model, dict):
        model = {'threshold': model, 'intercept': None}

    z = (advantage - model['threshold']) / POOLED_SCALE

    if (GAP_IN_PASS_MODEL and gap is not None
            and model.get('intercept') is not None):
        g = np.clip(gap, 0.0, ATTACK_GAP)
        inside = (np.asarray(gap) >= 0.0) & (np.asarray(gap) < ATTACK_GAP) \
            & (np.asarray(advantage) >= GAP_MODEL_MIN_ADVANTAGE)
        z_gap = model['intercept'] + GAP_ADV_COEF * advantage - GAP_COEF * g
        z = np.where(inside, z_gap, z)

    if TEAM_PASS_ENABLED:
        z = z + np.clip(team_shift, -TEAM_PASS_CAP, TEAM_PASS_CAP)

    z = np.clip(z, -30, 30)
    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, MAX_PASS_PROB)


def team_pass_shift(pace):
    """
    Each car's logit shift from its team's PC1, as an array indexed the way
    the lap loop indexes drivers.

    Missing teams get zero rather than the field mean: a team with no
    measurement is not average by evidence, it is simply unmeasured, and
    zero is the only value that adds nothing.
    """
    if not TEAM_PASS_ENABLED:
        return np.zeros(len(pace)), 'off'

    table = read_csv(f'overtake_team_profile_{SEASON}.csv')
    if table is None or 'PC1' not in table.columns:
        return np.zeros(len(pace)), 'missing'

    pc1 = table.set_index('Team')['PC1'].to_dict()
    mapped = pace['Team'].map(pc1)
    hit = int(mapped.notna().sum())
    if hit == 0:
        return np.zeros(len(pace)), 'no team matched'

    shift = np.clip(mapped.fillna(0.0).to_numpy(float) * TEAM_PASS_COEF,
                    -TEAM_PASS_CAP, TEAM_PASS_CAP)
    return shift, f'PC1 x {TEAM_PASS_COEF:.3f} ({hit}/{len(pace)} cars)'


def build_start_gaps(order_positions, n_drivers, track, rng, shape,
                     lock_order=True):
    """
    Time gaps for a standing start, from the order the cars line up in.

    order_positions is (n_sims, n_drivers), one-based. At the race start that
    is the grid; at a red-flag restart it is the running order when the race
    was stopped. Nothing else differs between the two, which is the whole
    reason this is a function rather than two copies.

    The chaos term is why the front row is not as exposed as the midfield: the
    first cars launch into clean air on a single line, while the pack funnels
    into turn one three abreast. START_FRONT_STABILITY sets how much of the
    field-wide spread the front gets; the back gets all of it.
    """
    chaos = (START_FRONT_STABILITY
             + (1.0 - START_FRONT_STABILITY)
             * (order_positions - 1) / max(n_drivers - 1, 1))
    noise = rng.normal(0.0,
                       track['start_sigma'] * track['grid_gap'] * chaos,
                       size=shape)
    gaps = (order_positions - 1) * track['grid_gap'] + noise

    if lock_order:
        # keep the time gaps, drop the reshuffle: sorting the drawn times back
        # onto the starting order leaves a realistic spread with nobody promoted
        rank = np.argsort(np.argsort(order_positions, axis=1), axis=1)
        gaps = np.take_along_axis(np.sort(gaps, axis=1), rank, axis=1)
    return gaps - gaps.min(axis=1, keepdims=True)


# --- simulation -------------------------------------------------------------


def run_simulation(pace, track, strategies, n_sims, seed=None,
                   progress_callback=None):
    rng = np.random.default_rng(seed)
    n_laps = track['n_laps']
    n_drivers = len(pace)
    shape = (n_sims, n_drivers)

    delta = pace['delta'].to_numpy()
    sigma = pace['sigma'].to_numpy()
    grid = pace['grid'].to_numpy(dtype=float)
    offsets = np.array([track['offsets'].get(c, 0.0) for c in COMPOUND_NAMES])

    # The whole tyre model, as one lookup table: PENALTY[c, t] is what a
    # compound-c tyre costs on the lap it enters at TyreLife t. Built from the
    # fitted curves before the race starts, so the lap loop is an indexing
    # operation rather than three exponentials and a comparison.
    #
    # Rows are clamped at each curve's own max_age. That clamp is the only
    # thing standing between the simulation and a silently extrapolated
    # number, so ENFORCE_STINT_CAP has to stay on for it to never bind.
    max_life = n_laps + 2
    tyre_life_grid = np.arange(0, max_life + 1)
    penalty_table = np.zeros((3, max_life + 1))
    cap_life = np.zeros(3, dtype=np.int64)
    cliff_life_by_compound = np.full(3, np.inf)
    for ci, comp in enumerate(COMPOUND_NAMES):
        curve = track['tyre'][comp]
        ages = np.clip(tyre_life_grid - AGE_OFFSET, 0, curve.max_age)
        penalty_table[ci] = curve_penalty(curve, ages)
        cap_life[ci] = max_stint_laps(curve)
        if curve.tau is not None and curve.gamma > 0:
            cliff_life_by_compound[ci] = curve.tau + AGE_OFFSET

    pit_laps, compound_idx, n_stints, plans, stint_len = build_strategy_tables(
        strategies, n_laps)

    strat_idx, strat_weights = assign_strategies(
        strategies, n_sims, n_drivers, rng)

    # accumulated tyre loss, so the pit rule is two lookups rather than a loop
    cum_loss = build_cum_table(track, n_laps)
    max_age_idx = cum_loss.shape[1] - 1

    neutral = sample_neutralizations(track['sc'], n_sims, n_laps, rng)

    # Every simulation's running order is kept, not just one. Which race best
    # represents the set can only be judged once all of them have finished, and
    # int8 keeps the cost small: 10k sims over 72 laps is about 35 MB.
    trace_order = np.zeros((n_laps, n_sims, n_drivers), dtype=np.int8)
    trace_pits = np.zeros((n_laps, n_sims, n_drivers), dtype=bool)

    # Lap times are kept too, in float32. The representative race is only
    # chosen once every simulation has finished, so there is no way to record
    # just the one that matters; it is all of them or none. At 72 laps and 22
    # cars that is about 63 MB, which buys the winner's lap-by-lap trace.
    trace_laptime = np.full((n_laps, n_sims, n_drivers), np.nan, dtype=np.float32)

    # --- grid: everyone starts spread out, and the start itself is messy ---
    total = build_start_gaps(np.broadcast_to(grid, shape), n_drivers, track,
                             rng, shape, lock_order=LOCK_START_ORDER)

    tyre_age = np.ones(shape, dtype=np.int32)
    stint_no = np.zeros(shape, dtype=np.int32)
    n_stops = np.zeros(shape, dtype=np.int32)
    sc_stops = np.zeros(shape, dtype=np.int32)
    laps_stuck = np.zeros(shape, dtype=np.int32)
    overtakes = np.zeros(n_sims, dtype=np.int32)
    bunched = np.zeros(n_sims, dtype=np.int32)     # SC restarts, per race
    rf_stops = np.zeros(shape, dtype=np.int32)     # free changes under red flag
    rf_lap_seen = np.full(n_sims, -1, dtype=np.int32)
    past_cliff = np.zeros(shape, dtype=np.int32)
    tyre_loss_sum = np.zeros(shape)

    # AR(1) noise state, started from its own stationary distribution so the
    # opening laps are no calmer than the rest of the race
    noise = rng.normal(0.0, sigma[None, :], size=shape)

    # a constant per-lap hazard that integrates to DNF_RATE_PER_RACE
    dnf_hazard = (1.0 - (1.0 - DNF_RATE_PER_RACE) ** (1.0 / n_laps)) \
        if DNF_ENABLED else 0.0
    retired = np.zeros(shape, dtype=bool)
    retired_lap = np.full(shape, -1, dtype=np.int32)

    my_pit_laps = pit_laps[strat_idx]
    my_compounds = compound_idx[strat_idx]
    my_n_stints = n_stints[strat_idx]
    my_lengths = stint_len[strat_idx]
    max_stints = my_compounds.shape[2]
    reactive_pits = np.zeros(shape, dtype=np.int32)
    forced_pits = np.zeros(shape, dtype=np.int32)

    # every curve carries a cap, so there is no infinite case to guard any more
    cap_by_compound = np.minimum(cap_life, n_laps + 1).astype(float)

    # the longest stint the simulation actually ran, per compound. Without
    # this the only way to know whether a cap bit was to squint at one
    # representative race and guess.
    longest_stint = np.zeros(3, dtype=np.int32)
    longest_overall = np.zeros(shape, dtype=np.int32)

    rows = np.arange(n_sims)[:, None]
    cols = np.arange(n_drivers)[None, :]
    fuel_mid = (n_laps - 1) / 2.0
    dirty_max = track['dirty_air_penalty']
    pass_model = build_pass_model(track)
    team_shift, team_shift_source = team_pass_shift(pace)
    track['team_pass_source'] = team_shift_source

    dirty_penalty = np.zeros(shape)

    # Running order is carried from lap to lap, not recomputed by sorting on
    # total. Sorting each lap silently completed every pass before the model
    # could rule on it: a car that gained a second was already ahead by the time
    # the code looked. Order as state means a pass has to be earned.
    # pace is sorted by grid, so position p starts holding driver p.
    order = np.broadcast_to(np.arange(n_drivers), shape).copy()

    for lap in range(n_laps):
        if progress_callback is not None:
            # the caller decides what to do with it; None keeps this loop
            # exactly as it was for every existing entry point
            progress_callback(lap, n_laps)
        state = neutral[:, lap][:, None]
        is_sc = state == 2
        is_vsc = state == 1
        is_rf = state == 3
        rf_lap_now = neutral[:, lap] == 3
        neutral_lap = neutral[:, lap] > 0

        # The last lap under yellow, and the first lap back to green. Both are
        # their own kind of lap: the field bunches on one and cannot pass on
        # the other, so each needs naming before anything else uses it.
        next_green = (np.ones(n_sims, dtype=bool) if lap + 1 >= n_laps
                      else neutral[:, lap + 1] == 0)
        neutral_ends_this_lap = neutral_lap & next_green
        is_restart_lap = ((~neutral_lap) & (neutral[:, lap - 1] > 0) if lap > 0
                          else np.zeros(n_sims, dtype=bool))

        current_compound = my_compounds[rows, cols, stint_no]

        # --- tyre penalty ---------------------------------------------------
        # D(a) for the tyre each car is on, read straight out of the table.
        # This is the total loss at that age, not an increment: it replaces
        # the previous lap's penalty rather than adding to it, so nothing
        # accumulates twice. The old flat-coefficient path is gone - running
        # both at once was the failure mode this replaces.
        #
        # Relative to a fresh tyre of the same compound. The pace difference
        # between compounds is carried by `offsets`, so the two do not overlap
        # - provided compound_scaling.csv was measured with tyre age
        # controlled. If it was not, some of the wear difference lives in the
        # offsets too and softs will look better here than they should.
        life = np.minimum(tyre_age, max_life)
        tyre_penalty = penalty_table[current_compound, life]

        cliff_life = cliff_life_by_compound[current_compound]
        past_cliff += (tyre_age > cliff_life).astype(np.int32)

        tyre_loss_sum += tyre_penalty

        clean_pace = (POLE_TIME + delta[None, :]
                      + offsets[current_compound]
                      + tyre_penalty
                      - track['fuel_effect'] * (lap - fuel_mid)
                      - track['evo_rate'] * (lap - fuel_mid))

        noise = (NOISE_AUTOCORR * noise
                 + np.sqrt(1.0 - NOISE_AUTOCORR ** 2)
                 * rng.normal(0.0, sigma[None, :], size=shape))
        lap_time = clean_pace + noise
        lap_time = lap_time + dirty_penalty

        if NEUTRAL_FREEZES_GAPS:
            # one lap time for the whole field, so every gap survives the
            # neutralization untouched and nothing can be won behind it.
            # A red-flag lap is flattened the same way and for a stronger
            # reason: the lap that gets the race stopped is an accident, and
            # whatever pace anyone was carrying on it means nothing.
            frozen = POLE_TIME + np.where(is_sc, SC_LAP_PENALTY, VSC_LAP_PENALTY)
            frozen = np.where(is_rf, POLE_TIME + SC_LAP_PENALTY, frozen)
            lap_time = np.where(state > 0, frozen, lap_time)
        else:
            lap_time = np.where(is_sc, lap_time + SC_LAP_PENALTY, lap_time)
            lap_time = np.where(is_vsc, lap_time + VSC_LAP_PENALTY, lap_time)

        # --- pit decision -------------------------------------------------
        has_stops_left = stint_no < (my_n_stints - 1)
        planned_lap = my_pit_laps[rows, cols, stint_no]

        # A cheap stop stays available for the whole neutralization except its
        # final lap, which is where the field bunches. Letting a car take the
        # pit loss and the compression on the same lap has two mechanisms
        # rewriting the same gaps at once, and the order that comes out
        # depends on which ran first rather than on anything a driver did.
        opportunistic = (has_stops_left
                         & (is_sc | is_vsc)
                         & (tyre_age >= MIN_STINT_BEFORE_PIT)
                         & (lap + 1 >= planned_lap - PIT_WINDOW_TOLERANCE)
                         & ~neutral_ends_this_lap[:, None])

        if REACTIVE_PIT:
            # The plan no longer forces the stop on its own lap; it sets the
            # centre of a band, and inside that band the tyre decides. `due`
            # survives only as the far edge, so a car cannot run to the flag
            # on a plan it keeps declining to execute.
            due = has_stops_left & (lap + 1 >= planned_lap + PIT_DECISION_BAND)

            in_band = (has_stops_left
                       & (tyre_age >= MIN_STINT_BEFORE_PIT)
                       & (lap + 1 >= planned_lap - PIT_DECISION_BAND)
                       & (lap + 1 <= planned_lap + PIT_DECISION_BAND))

            # cost of one more lap on what is already fitted
            age_now = np.clip(tyre_age, 0, max_age_idx - 1)
            stay_cost = (cum_loss[current_compound, age_now + 1]
                         - cum_loss[current_compound, age_now])

            # What that lap would have cost on the tyres still to come. The
            # laps left over are shared between the remaining stints in the
            # proportions the plan asked for, so the comparison is against the
            # whole rest of the race rather than the next stint alone.
            #
            # This is the marginal cost of one extra lap, taken directly rather
            # than by differencing two totals. Differencing looked equivalent
            # and was not: stint lengths are integers, so shortening the race
            # by a lap moves some stints and not others, and the difference
            # came out as either zero or a whole lap's penalty. The decision
            # flipped every other lap on that artefact alone.
            remaining = n_laps - lap - 1
            displaced = np.zeros(shape)
            if remaining > 1:
                weight_sum = np.zeros(shape)
                for j in range(max_stints):
                    weight_sum += np.where(j > stint_no, my_lengths[..., j], 0.0)
                weight_sum = np.maximum(weight_sum, 1e-9)

                for j in range(max_stints):
                    active = j > stint_no
                    share = np.where(active, my_lengths[..., j], 0.0) / weight_sum
                    comp_j = my_compounds[..., j]
                    end_age = np.clip((share * remaining).astype(np.int32),
                                      0, max_age_idx - 1)
                    per_lap = (cum_loss[comp_j, end_age + 1]
                               - cum_loss[comp_j, end_age])
                    displaced += np.where(active, share * per_lap, 0.0)

            # pit when staying out costs more than the lap it displaces
            worth_it = stay_cost > displaced
            reactive = in_band & worth_it
        else:
            due = has_stops_left & (lap + 1 >= planned_lap)
            reactive = np.zeros(shape, dtype=bool)

        pitting = due | opportunistic | reactive

        if ENFORCE_STINT_CAP:
            # Nobody has ever run this compound this long here, so neither
            # does the simulation.
            #
            # This deliberately does not check has_stops_left. The final stint
            # is where the problem lived: the planned stops were all made,
            # nothing was left to spend, and a car could sit on the same set
            # for forty-three laps with the cap watching. Reaching the cap on
            # the last stint now buys a fresh set of the same compound -
            # stint_no is already clamped, so the strategy is unchanged and
            # only the tyre is new. It costs a full pit stop, which is the
            # point: a plan that ends up there was a bad plan.
            forced = ((tyre_age >= cap_by_compound[current_compound])
                      & (lap < n_laps - 1))
            pitting = pitting | forced
            forced_pits += (forced & ~due & ~opportunistic & ~reactive).astype(np.int32)

        # A red flag replaces every stop that lap. There is no window to take
        # and no pit loss to pay: the race is stopped, the cars are in the
        # pit lane, and everyone changes tyres for nothing. Suppressing the
        # normal stop here is what stops a car being charged for a stop it
        # was going to get free anyway.
        if RED_FLAG_ENABLED:
            pitting = pitting & ~rf_lap_now[:, None]

        reactive_pits += (reactive & ~due & ~opportunistic).astype(np.int32)

        cost = np.full(shape, track['pit_loss'])
        cost = np.where(is_sc, cost * SC_PIT_DISCOUNT, cost)
        cost = np.where(is_vsc, cost * VSC_PIT_DISCOUNT, cost)
        lap_time = lap_time + np.where(pitting, cost, 0.0)

        # --- retirements ---------------------------------------------------
        if dnf_hazard > 0:
            newly = (~retired) & (rng.random(shape) < dnf_hazard)
            retired = retired | newly
            retired_lap = np.where(newly, lap, retired_lap)
        else:
            newly = np.zeros(shape, dtype=bool)

        # a car that is out stops racing: its clock is replaced by a marker that
        # sorts it behind every finisher, and behind anyone who lasted longer
        total = total + np.where(retired, 0.0, lap_time)
        if dnf_hazard > 0:
            total = np.where(newly, 1e6 - retired_lap * 1e3, total)

        # --- the safety car closes the field up ----------------------------
        # On the last lap behind the safety car the queue has formed, and the
        # gaps that existed when it came out are gone. A leader who was
        # twenty-five seconds clear restarts half a second ahead, which is the
        # single largest thing a safety car does to a race and the part the
        # freeze alone never modelled.
        #
        # It runs on the finished `total` for the lap - pit loss, tyre penalty
        # and neutralization penalty all already in - and before the order is
        # resolved, so the passing rules below see the compressed field.
        #
        # The walk down the order is cumulative on purpose: each car is put
        # SC_QUEUE_GAP behind the one already moved ahead of it, so the whole
        # field ends up as one train rather than each pair closing
        # independently.
        #
        # A retired car must not be compressed. Its clock is a sorting marker
        # rather than a time, and pulling that marker up to half a second
        # behind a running car would put it back in the race.
        if SC_BUNCHING:
            sc_ends = (neutral[:, lap] == 2) & neutral_ends_this_lap
            if sc_ends.any():
                ord_bunch = np.take_along_axis(total, order, axis=1)
                ord_out_bunch = np.take_along_axis(retired, order, axis=1)
                for p in range(1, n_drivers):
                    gap = ord_bunch[:, p] - ord_bunch[:, p - 1]
                    compress = (sc_ends & (gap > SC_QUEUE_GAP)
                                & ~ord_out_bunch[:, p]
                                & ~ord_out_bunch[:, p - 1])
                    ord_bunch[:, p] = np.where(
                        compress, ord_bunch[:, p - 1] + SC_QUEUE_GAP,
                        ord_bunch[:, p])
                np.put_along_axis(total, order, ord_bunch, axis=1)
                bunched += sc_ends.astype(np.int32)

        # --- resolve the running order ------------------------------------
        # Passing splits in two, and collapsing them was wrong.
        #
        # A car that stops loses twenty seconds at once. The car behind is then
        # far up the road, not fighting for the position, and there is nothing
        # to defend with from the pit lane. Treating that as a duel let a
        # driver keep second place through his own stop while the man behind
        # stayed out: the block held him 0.35 s behind a car that was in the
        # pits, which quietly refunded the entire cost of the stop.
        #
        # So a gap wider than AUTO_PASS_MARGIN, or a car ahead that is pitting,
        # goes through unopposed. Only genuine wheel-to-wheel running gets a
        # dice roll. The sweep also repeats, because one stop should drop a car
        # several places in a single lap, not one.
        ord_total = np.take_along_axis(total, order, axis=1)
        ord_pace = np.take_along_axis(clean_pace, order, axis=1)
        ord_pitting = np.take_along_axis(pitting, order, axis=1)
        ord_out = np.take_along_axis(retired, order, axis=1)

        stuck_ord = np.zeros(shape, dtype=bool)

        # laps on which a car can change place by driving past another: not
        # under yellow, and not on the restart lap where the queue has just
        # been formed
        racing = ~neutral_lap & ~is_restart_lap

        for sweep in range(ORDER_SWEEPS):
            contested_sweep = sweep == 0      # dice are rolled once per lap
            moved = np.zeros(n_sims, dtype=bool)

            for p in range(1, n_drivers):
                ahead = ord_total[:, p - 1]
                behind = ord_total[:, p]
                gap = behind - ahead

                # Nothing can be defended from the pit lane, and a big enough
                # time advantage is not a fight either. Under a neutralization
                # the time-gap route is closed: the only ways past a car under
                # yellow are that it pitted or that it stopped, both of which
                # are position changes rather than overtakes.
                #
                # The restart lap is closed the same way. The field has just
                # been squeezed into a queue, so every car is inside a couple
                # of tenths of the one ahead and the raw time gaps would hand
                # out a pass everywhere at once. Pace still runs - the gaps
                # start opening again on this lap - but nobody changes place
                # by driving past. Pit stops and retirements still move cars,
                # for the same reason they do under yellow.
                uncontested = (((gap < -AUTO_PASS_MARGIN) & racing)
                               | ord_pitting[:, p - 1]
                               | ord_out[:, p - 1])

                do_pass = uncontested
                blocked = np.zeros(n_sims, dtype=bool)

                if contested_sweep:
                    attacking = ((gap < ATTACK_GAP)
                                 & ~uncontested
                                 & ~ord_pitting[:, p]
                                 & ~ord_out[:, p]
                                 & racing)
                    advantage = ord_pace[:, p - 1] - ord_pace[:, p]
                    # order[:, p] is the driver index in the slot behind, so
                    # the team term follows the cars through the sweep
                    # without a second array to permute
                    shift = team_shift[order[:, p]]
                    rolled = rng.random(n_sims) < pass_probability(
                        advantage, pass_model, gap=gap, team_shift=shift)
                    do_pass = do_pass | (attacking & rolled)
                    blocked = attacking & ~rolled

                    ord_total[:, p] = np.where(blocked, ahead + HELD_GAP, behind)
                    stuck_ord[:, p] = blocked
                    overtakes += (attacking & rolled).astype(np.int32)

                if not do_pass.any():
                    continue

                for arr in (order, ord_total, ord_pace, ord_pitting, ord_out):
                    front = arr[:, p - 1].copy()
                    arr[:, p - 1] = np.where(do_pass, arr[:, p], front)
                    arr[:, p] = np.where(do_pass, front, arr[:, p])
                stuck_front = stuck_ord[:, p - 1].copy()
                stuck_ord[:, p - 1] = np.where(do_pass, stuck_ord[:, p], stuck_front)
                stuck_ord[:, p] = np.where(do_pass, stuck_front, stuck_ord[:, p])

                moved |= do_pass

            if not moved.any():
                break

        # dirty air for next lap, fading linearly with the resolved gap
        dirty_ord = np.zeros(shape)
        gaps = ord_total[:, 1:] - ord_total[:, :-1]
        severity = np.clip(1.0 - gaps / DIRTY_AIR_RANGE, 0.0, 1.0)
        dirty_ord[:, 1:] = np.where(neutral_lap[:, None], 0.0, dirty_max * severity)

        np.put_along_axis(total, order, ord_total, axis=1)

        # --- the race is stopped and started again -------------------------
        # A red flag is the safety car's bunching taken to its limit. The
        # queue does not just close up, it disappears: every gap is deleted,
        # and the race restarts from a standing start in the order the cars
        # were running when it was thrown.
        #
        # So the field is levelled onto the leader and then the standing-start
        # gaps are laid back over it - the same function the race start uses,
        # given the running order instead of the grid. What was a
        # twenty-second lead becomes a grid slot.
        #
        # Retired cars are left alone. Their clock is a sorting marker rather
        # than a time, and levelling it onto the leader would put them back on
        # the lead lap.
        if RED_FLAG_ENABLED and rf_lap_now.any():
            running_pos = np.empty(shape, dtype=float)
            ranks_now = np.broadcast_to(np.arange(1, n_drivers + 1), shape)
            np.put_along_axis(running_pos, order, ranks_now, axis=1)

            restart_gaps = build_start_gaps(running_pos, n_drivers, track, rng,
                                            shape, lock_order=LOCK_START_ORDER)
            alive_now = ~retired
            leader = np.where(alive_now, total, np.inf).min(axis=1, keepdims=True)
            restarted = leader + restart_gaps

            apply_rf = rf_lap_now[:, None] & alive_now
            total = np.where(apply_rf, restarted, total)
            rf_lap_seen = np.where(rf_lap_now & (rf_lap_seen < 0), lap, rf_lap_seen)

        in_dirty = np.empty(shape, dtype=bool)
        np.put_along_axis(in_dirty, order, stuck_ord, axis=1)
        dirty_penalty = np.empty(shape)
        np.put_along_axis(dirty_penalty, order, dirty_ord, axis=1)
        laps_stuck += in_dirty.astype(np.int32)

        # A red flag hands everyone a new set for nothing. What set depends on
        # when it lands. In the opening laps the tyre on the car has barely
        # been used and in the closing laps there is not enough race left to
        # make a different compound pay, so both ends just refit what is
        # already fitted and the plan does not move on. In between, the free
        # stop is worth spending as the next step of the plan.
        #
        # The stint lengths behind it are not redistributed. That leaves the
        # last stint longer than planned, which ENFORCE_STINT_CAP then catches:
        # the car runs into its compound's cap and takes a real stop. Not
        # optimal, but not wrong, and redistributing is separate work.
        refit = np.zeros(shape, dtype=bool)
        advance_plan = np.zeros(shape, dtype=bool)
        if RED_FLAG_ENABLED:
            refit = rf_lap_now[:, None] & ~retired
            if (lap >= RF_EARLY_LAPS) and (lap < n_laps - RF_LATE_LAPS):
                advance_plan = refit

        trace_order[lap] = order.astype(np.int8)
        # the free change is recorded as a stop so the strategy panel splits
        # the stint there rather than drawing one bar straight through it
        trace_pits[lap] = pitting | refit
        # a retired car's clock was replaced by a sorting marker, so its lap
        # time is not a lap time any more and is left as nan
        trace_laptime[lap] = np.where(retired, np.nan, lap_time)

        alive = ~retired
        longest_overall = np.maximum(longest_overall,
                                     np.where(alive, tyre_age, 0))
        for ci in range(3):
            on_it = alive & (current_compound == ci)
            if on_it.any():
                longest_stint[ci] = max(int(longest_stint[ci]),
                                        int(tyre_age[on_it].max()))

        # --- advance state --------------------------------------------------
        pitting = pitting & ~retired

        resets = pitting | refit
        advances = pitting | advance_plan

        tyre_age = np.where(resets, 1, tyre_age + 1)
        stint_no = np.where(advances,
                            np.minimum(stint_no + 1, my_n_stints - 1), stint_no)
        n_stops += resets.astype(np.int32)          # a free change is still a stop
        sc_stops += (pitting & (is_sc | is_vsc)).astype(np.int32)
        rf_stops += refit.astype(np.int32)

    # the final running order is the result, not a re-sort of accumulated time
    positions = np.empty(shape, dtype=np.int64)
    ranks = np.broadcast_to(np.arange(1, n_drivers + 1), shape)
    np.put_along_axis(positions, order, ranks, axis=1)

    diag = {
        'neutral': neutral,
        'n_stops': n_stops,
        'sc_stops': sc_stops,
        'strat_weights': strat_weights,
        'plans': plans,
        'overtakes': overtakes,
        'bunched': bunched,
        'rf_stops': rf_stops,
        'rf_lap': rf_lap_seen,
        'laps_stuck': laps_stuck,
        'retired': retired,
        'retired_lap': retired_lap,
        'trace_order': trace_order,
        'trace_pits': trace_pits,
        'trace_laptime': trace_laptime,
        'strat_idx': strat_idx,
        'compound_table': compound_idx,
        'n_stints_table': n_stints,
        'past_cliff': past_cliff,
        'tyre_loss': tyre_loss_sum,
        'reactive_pits': reactive_pits,
        'forced_pits': forced_pits,
        'longest_stint': longest_stint,
        'longest_overall': longest_overall,
        'cap_by_compound': cap_by_compound,
    }
    return positions, total, strat_idx, diag


def format_time(seconds):
    """Race time as h:mm:ss.sss, the way a timing screen shows it."""
    if not np.isfinite(seconds):
        return '-'
    h, rem = divmod(float(seconds), 3600.0)
    m, s = divmod(rem, 60.0)
    return f'{int(h)}:{int(m):02d}:{s:06.3f}'


def summarize(pace, positions, strat_idx, strategies, diag, total):
    # gap is measured inside each simulation and then averaged: taking the
    # difference of two averages would mix races that were never run together
    winner_time = total.min(axis=1, keepdims=True)
    gaps = total - winner_time

    rows = []
    for i, row in pace.iterrows():
        pos = positions[:, i]
        modal = int(np.bincount(strat_idx[:, i], minlength=len(strategies)).argmax())
        rows.append({
            'Driver': row['Driver'],
            'Team': row['Team'],
            'grid': int(row['grid']),
            'delta': round(row['delta'], 3),
            'affinity': round(row.get('affinity', 0.0), 2),
            'strategy': '-'.join(c[0] for c in strategies[modal]['sequence']),
            'stops': round(diag['n_stops'][:, i].mean(), 2),
            'cliff_laps': round(float(diag['past_cliff'][:, i].mean()), 1),
            'tyre_s': round(float(diag['tyre_loss'][:, i].mean()), 1),
            'P_dnf': round(float(diag['retired'][:, i].mean()), 3),
            'stuck': round(diag['laps_stuck'][:, i].mean(), 1),
            'P_win': (pos == 1).mean(),
            'P_podium': (pos <= 3).mean(),
            'P_points': (pos <= 10).mean(),
            'mean_pos': pos.mean(),
            'race_time': format_time(total[:, i].mean()),
            'gap': round(float(gaps[:, i].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values('P_win', ascending=False).reset_index(drop=True)


def plot_position_histograms(pace, positions, out_path, top_n=6):
    codes = pace['Driver'].tolist()
    n_drivers = len(pace)
    mean_pos = positions.mean(axis=0)
    top_idx = np.argsort(mean_pos)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    bins = np.arange(0.5, n_drivers + 1.5, 1)
    for ax, i in zip(axes.ravel(), top_idx):
        ax.hist(positions[:, i], bins=bins, edgecolor='white')
        ax.set_title(f'{codes[i]}  (mean {mean_pos[i]:.1f})')
        ax.set_xlabel('Finishing position')
        ax.set_xticks(range(1, n_drivers + 1, 2))
    fig.suptitle(f'{TARGET_EVENT} {SEASON} - {positions.shape[0]:,} sims (v1.1-A)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f'Histogram saved: {out_path}')


def rebuild_compounds(diag, sim):
    """
    Per-lap compound for one race, rebuilt rather than stored.

    Which tyre a car is on follows from its strategy and the laps it actually
    stopped on, both of which are already recorded, so keeping a third array of
    the same size would be waste.
    """
    pits = diag['trace_pits'][:, sim, :]
    strat = diag['strat_idx'][sim]
    table = diag['compound_table']
    n_stints = diag['n_stints_table'][strat]

    names = np.array(COMPOUND_NAMES)
    n_laps, n_drivers = pits.shape
    out = np.empty((n_laps, n_drivers), dtype=object)

    stint_no = np.zeros(n_drivers, dtype=int)
    for lap in range(n_laps):
        out[lap] = names[table[strat, stint_no]]
        stepped = pits[lap]
        stint_no = np.where(stepped,
                            np.minimum(stint_no + 1, n_stints - 1),
                            stint_no)
    return out


def build_param_lines(pace, track, strategies, diag, positions, total):
    """Every setting the run used, for the diagnostics panel."""
    sc = track['sc']
    neutral = diag['neutral']
    grid = pace['grid'].to_numpy()
    shift = float(np.abs(positions - grid[None, :]).mean())

    lines = [
        f'target        {TARGET_EVENT} {SEASON}, {track["n_laps"]} laps',
        f'sims          {N_SIMS:,}   seed {RANDOM_SEED}',
        f'pole          {POLE_DRIVER} {POLE_TIME:.3f} s',
        f'drivers       {len(pace)}',
        f'grid source   {pace.attrs.get("grid_source", "?")}',
        f'grid gap      {track["grid_gap"]:.3f} s/slot   '
        f'start sigma {track["start_sigma"]:.2f} pos',
        f'start chaos   front {START_FRONT_STABILITY:.0%} of field-wide, back 100%',
        f'start order   {"locked to grid" if LOCK_START_ORDER else "raced"}',
        f'delta range   {pace["delta"].min():.3f} to {pace["delta"].max():.3f} s/lap',
        f'sigma range   {pace["sigma"].min():.3f} to {pace["sigma"].max():.3f} s/lap',
        f'fuel_effect   {track["fuel_effect"]:.3f} s/lap',
    ]

    lines.append(f'tyre model    D(a)=b1*a+b2*a^2+g*max(0,a-tau)^2  '
                 f'v{track["tyre_model_version"]}, age offset {AGE_OFFSET}')
    for c in COMPOUND_NAMES:
        p = track['tyre'][c]
        cliff = 'none' if p.tau is None else f'{p.tau + AGE_OFFSET:.0f}'
        lines.append(f'  {c:8s}    b1 {p.b1:.4f}  b2 {p.b2:.5f}  '
                     f'g {p.gamma:.5f}  cliff {cliff:>5s}  '
                     f'cap {max_stint_laps(p):>3d}  [{p.source}]')
    lines.append(f'cliff laps    {diag["past_cliff"].mean():.1f} per driver')

    lines += [
        f'evolution     {track["evo_rate"]:+.4f} s/lap ({track["evo_source"]})',
        f'noise rho     {NOISE_AUTOCORR:.2f}  (race spread x'
        f'{np.sqrt((1 + NOISE_AUTOCORR) / (1 - NOISE_AUTOCORR)):.2f} vs white)',
        f'DNF rate      {DNF_RATE_PER_RACE:.1%} per car'
        f'{"" if DNF_ENABLED else " (off)"}',
        f'tyre driver   not modelled - every car wears the same in v1.2',
        f'pit_loss      {track["pit_loss"]:.1f} s',
        f'compounds     {"-".join(track["compounds"])}   '
        + ' '.join(f'{c[0]}{v:+.2f}' for c, v in track['offsets'].items()),
        f'SC            lambda {sc["sc_lambda"]:.2f} x {sc["sc_duration"]:.1f} laps '
        f'({track["sc_source"]})',
        f'VSC           lambda {sc["vsc_lambda"]:.2f} x {sc["vsc_duration"]:.1f} laps',
        f'under yellow  {"gaps frozen, no passing" if NEUTRAL_FREEZES_GAPS else "racing continues (v1.0)"}',
        f'SC bunching   '
        + (f'queue at {SC_QUEUE_GAP:.2f} s on the last SC lap, no passing on '
           f'the restart, {diag["bunched"].mean():.2f} restarts per race'
           if SC_BUNCHING else 'off'),
        f'SC timing     start ~N({SC_TIMING_MEAN_FRACTION:.0%}, '
        f'{SC_TIMING_STD_FRACTION:.0%}) of distance, '
        f'min {MIN_SC_DURATION} laps SC / {MIN_VSC_DURATION} VSC',
        f'red flag      '
        + (f'p {sc.get("rf_lambda", 0.0):.3f} per race, restart from the '
           f'running order, free tyres ({(diag["rf_lap"] >= 0).mean():.1%} of races)'
           if RED_FLAG_ENABLED else 'off'),
        f'p_pass        {track["pass_rate"]:.3f}/lap ({track["pass_source"]}), '
        f'difficulty {track["overtake_difficulty"]}/5',
        f'gap term      {"on" if GAP_IN_PASS_MODEL else "off"}, '
        f'{GAP_COEF:.2f} logit/s, zero at {REFERENCE_GAP:.2f} s',
        f'team term     {track.get("team_pass_source", "off")}',
        f'dirty air     {track["dirty_air_penalty"]:.2f} s/lap over '
        f'{DIRTY_AIR_RANGE:.1f} s',
        f'attack gap    {ATTACK_GAP:.2f} s, held at {HELD_GAP:.2f} s when blocked '
        f'(measured median of a car that failed to pass)',
        f'auto pass     gap over {AUTO_PASS_MARGIN:.2f} s or car ahead in the pits',
        f'affinity      {pace.attrs.get("affinity_source", "off")}; team term '
        f'{pace.attrs.get("affinity_team_source", "off")}',
        f'affinity rate {AFFINITY_PCT_PER_POSITION * 100:.3f}% of the lap per '
        f'position = {AFFINITY_PCT_PER_POSITION * POLE_TIME:.3f} s here, cap '
        f'{AFFINITY_CAP_PCT * POLE_TIME:.2f} s',
        f'strategies    {len(strategies)} offered, temperature '
        f'{STRATEGY_TEMPERATURE:.1f}'
        f'{", recosted from the tyre curve" if RECOST_STRATEGIES else ""}',
        f'pit rule      '
        + (f'reactive, planned lap +/- {PIT_DECISION_BAND}'
           if REACTIVE_PIT else 'fixed at the planned lap'),
        f'reactive pits {diag["reactive_pits"].mean():.2f} per driver, '
        f'{diag["forced_pits"].mean():.2f} forced by the cap',
        '',
        f'races with SC {(neutral == 2).any(axis=1).mean():.1%}   '
        f'VSC {(neutral == 1).any(axis=1).mean():.1%}',
        f'overtakes     {diag["overtakes"].mean():.1f} per race',
        f'laps held up  {diag["laps_stuck"].mean():.1f} per driver',
        f'stops         {diag["n_stops"].mean():.2f} per driver, '
        f'{diag["sc_stops"].mean():.2f} under neutralization',
        f'tyre loss     {diag["tyre_loss"].mean():.1f} s per driver over the race',
        f'shift vs grid {shift:.2f} positions (naive benchmark 3.45)',
        f'winning time  {format_time(total.min(axis=1).mean())}',
    ]
    return lines


# --- headless entry point (v1.6) --------------------------------------------

# Every flag the interface is allowed to move. Anything not here stays out of
# reach on purpose: these are switches between two modelled behaviours, and
# the constants around them are calibrations that a slider would quietly
# invalidate.
# USE_TYRE_MODEL is deliberately absent. The v1.6 roadmap lists it, but the
# v1.2 tyre rewrite removed the k/W/m path it used to switch to, so there is
# no second behaviour left for it to select. A flag that cannot change
# anything is worse than no flag: it invites the reader to believe a
# comparison was made.
RUNTIME_FLAGS = {
    'ENFORCE_STINT_CAP': 'Refuse stints longer than the curve was measured '
                         'for, instead of extrapolating past the data.',
    'RECOST_STRATEGIES': 'Re-price the strategy menu on this circuit\'s own '
                         'curves before offering it.',
    'REACTIVE_PIT': 'Let a car pit early or late as its tyre actually goes '
                    'off. Off, it stops on the planned lap.',
    'PER_DRIVER_STRATEGY': 'Cost the strategy menu per driver. The v1.2 tyre '
                           'model has no driver term, so this does nothing yet.',
    'NEUTRAL_FREEZES_GAPS': 'Hold time gaps while the race is neutralised.',
    'SC_BUNCHING': 'Close the field up into a queue behind the safety car.',
    'RED_FLAG_ENABLED': 'Allow a red flag: a free tyre change and a standing '
                        'restart in running order.',
    'LOCK_START_ORDER': 'Keep the grid order through lap one instead of '
                        'racing the start.',
    'DNF_ENABLED': 'Let cars retire.',
    'EVOLUTION_ENABLED': 'Let the track surface rubber in across the race.',
    'AFFINITY_ENABLED': 'Shift pace by how well this circuit suits each car.',
    'AFFINITY_TEAM_2026': 'Take the team half of affinity from 2026 alone '
                          'rather than the 2022-25 record.',
    'GAP_IN_PASS_MODEL': 'Price a pass by how close the follower actually is, '
                         'not only by its pace advantage.',
    'TEAM_PASS_ENABLED': 'Let a team\'s measured overtaking strength (PC1) '
                         'move its pass odds.',
    'IS_WET': 'Run the race wet.',
}

# A flag named here but missing from the module would only surface when
# somebody toggled it in the interface, which is the worst place to find out.
_missing = [f for f in RUNTIME_FLAGS if f not in globals()]
if _missing:
    raise RuntimeError(f'RUNTIME_FLAGS names nothing: {", ".join(_missing)}')


@contextlib.contextmanager
def flag_overrides(overrides):
    """
    Temporarily moves module-level flags, then puts them back.

    The flags are globals read at call time, so setattr is enough and the
    functions below need no plumbing. It is not the design a config object
    would give, but it is the one that does not touch forty call sites.

    Restoring in a finally block matters more than it looks: the interface
    keeps the module alive between runs, so a flag left flipped by a failed
    run would silently poison every run after it.
    """
    if not overrides:
        yield {}
        return

    module = sys.modules[__name__]
    unknown = [k for k in overrides if k not in RUNTIME_FLAGS]
    if unknown:
        raise KeyError(f'not runtime flags: {", ".join(sorted(unknown))}')

    previous = {k: getattr(module, k) for k in overrides}
    for key, value in overrides.items():
        setattr(module, key, value)
    try:
        yield previous
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


def source_badges(pace, track):
    """
    Where every number in this run came from.

    This is the question the project keeps asking of itself, and until now
    the answer lived in console lines that scroll away. A measured value and
    a hand-set one carry very different weight, and a reader who cannot tell
    them apart will trust the wrong one.

        measured   read off this circuit's own data
        derived    computed from another cell rather than measured here
        borrowed   taken from a donor circuit
        shrunk     measured, then pulled toward the calendar for thin samples
        hand-set   chosen by a person, and owed a measurement
        fallback   nothing was available
    """
    def classify(text):
        t = str(text).lower()
        if 'fallback' in t or 'placeholder' in t:
            return 'fallback'
        if 'shrunk' in t or 'shrink' in t:
            return 'shrunk'
        if 'borrow' in t or 'donor' in t or 'copy' in t:
            return 'borrowed'
        if 'derived' in t or 'panel' in t or 'order-clip' in t or 'scaled' in t:
            return 'derived'
        if 'measured' in t or 'csv' in t or t.startswith('f1_'):
            return 'measured'
        return 'hand-set'

    # (name, source text, value, explicit kind or None to classify)
    #
    # Where the pipeline already hands back a source string, it is read.
    # Where it does not, the kind is stated rather than guessed: a blended
    # affinity and a coefficient derived from a measured PC1 both read as
    # hand-set to a string matcher, and mislabelling a derived value as
    # hand-set is exactly the error this panel exists to prevent.
    rows = [
        ('Grid', pace.attrs.get('grid_source', '?'),
         f'{track["grid_gap"]:.2f} s/slot', None),
        ('Pole time', f'f1_{SEASON}_poles.csv', f'{POLE_TIME:.3f} s',
         'measured'),
        ('Pass rate', track.get('pass_source', '?'),
         f'{track["pass_rate"]:.3f}/lap', None),
        ('Neutralisation', track.get('sc_source', '?'),
         f'SC {track["sc"]["sc_lambda"]:.2f}/race', None),
        ('Degradation', track.get('deg_source', '?'),
         f'{track["deg"]:.3f} s/lap', None),
        ('Compound offsets', track.get('offset_source', '?'),
         ', '.join(f'{c[0]}{track["offsets"][c]:+.2f}' for c in COMPOUND_NAMES),
         None),
        ('Track evolution', track.get('evo_source', '?'),
         f'{track.get("evo_rate", 0.0):.4f} s/lap', None),
        ('Affinity', pace.attrs.get('affinity_source', 'off'),
         f'{pace["affinity"].abs().max():.1f} pos max', 'derived'),
        ('Affinity, team', pace.attrs.get('affinity_team_source', 'off'),
         '', None),
        ('Affinity rate', f'{AFFINITY_PCT_PER_POSITION * 100:.3f}% of the lap '
         f'per position', f'{AFFINITY_PCT_PER_POSITION * POLE_TIME:.3f} s',
         'hand-set'),
        ('Team pass term', track.get('team_pass_source', 'off'),
         f'{TEAM_PASS_COEF:.3f} logit/PC1', 'derived'),
        ('Gap in pass model', 'overtaking.py joint fit, 4,490 laps'
         if GAP_IN_PASS_MODEL else 'off',
         f'{GAP_COEF:.2f} logit/s', 'measured' if GAP_IN_PASS_MODEL else None),
        ('Held gap', 'median of a car that failed to pass, 2022-25',
         f'{HELD_GAP:.2f} s', 'measured'),
        ('Dirty air', 'chosen, never measured',
         f'{track["dirty_air_penalty"]:.2f} s/lap over {DIRTY_AIR_RANGE:.1f} s',
         'hand-set'),
        ('DNF rate', 'chosen, never measured',
         f'{DNF_RATE_PER_RACE:.0%} per car', 'hand-set'),
        ('Pit loss', track.get('pit_source', 'measured'),
         f'{track["pit_loss"]:.1f} s', None),
    ]

    for compound in COMPOUND_NAMES:
        curve = track['tyre'][compound]
        rows.append((f'Tyre, {compound.lower()}', curve.source,
                     f'b1 {curve.b1:.4f}  cap {max_stint_laps(curve)} laps',
                     None))

    out = []
    for name, src, val, kind in rows:
        src = str(src)
        out.append({'name': name, 'source': src,
                    'kind': kind or classify(src), 'value': val})
    return out


def run(overrides=None, progress_callback=None):
    """
    One simulation, with nothing else attached.

    Prints nothing, draws nothing, writes nothing, opens no browser. main()
    below is the old behaviour rebuilt on top of this, so the console output
    is unchanged and the interface has something it can call without
    inheriting a matplotlib window.
    """
    with flag_overrides(overrides):
        pace = load_drivers()
        track = load_track()
        strategies = load_strategies(track)

        dropped = []
        if RECOST_STRATEGIES:
            strategies, dropped = recost_strategies(strategies, track)

        positions, total, strat_idx, diag = run_simulation(
            pace, track, strategies, N_SIMS, RANDOM_SEED,
            progress_callback=progress_callback)

        summary = summarize(pace, positions, strat_idx, strategies, diag, total)
        param_lines = build_param_lines(pace, track, strategies, diag,
                                        positions, total)

        rep, report = find_representative(positions, diag)
        retired_lap = diag['retired_lap'][rep] if 'retired_lap' in diag else None
        compounds = rebuild_compounds(diag, rep)
        trace = {
            'sim': rep,
            'order': diag['trace_order'][:, rep, :].astype(np.int32),
            'pits': diag['trace_pits'][:, rep, :],
            'neutral': diag['neutral'][rep],
            'retired_lap': retired_lap,
            'lap_times': diag['trace_laptime'][:, rep, :].astype(float),
            'compounds': compounds,
        }
        stints = build_stints(compounds, trace['pits'],
                              pace['Driver'].to_numpy(),
                              retired_lap=retired_lap)

        return {
            'pace': pace, 'track': track, 'strategies': strategies,
            'dropped': dropped, 'positions': positions, 'total': total,
            'strat_idx': strat_idx, 'diag': diag, 'summary': summary,
            'param_lines': param_lines, 'badges': source_badges(pace, track),
            'trace': trace, 'stints': stints, 'rep': rep, 'report': report,
            'event': TARGET_EVENT, 'season': SEASON, 'n_sims': N_SIMS,
            'seed': RANDOM_SEED, 'n_laps': track['n_laps'],
            'flags': {k: globals()[k] for k in RUNTIME_FLAGS},
        }


def position_distribution(positions, drivers):
    """
    P(driver finishes in position k), as a table.

    The console reports a mean finishing position, which hides the shape
    entirely: a car that alternates second and twentieth averages eleventh
    and so does one that finishes eleventh every time.
    """
    n_drivers = positions.shape[1]
    counts = np.zeros((n_drivers, n_drivers))
    for i in range(n_drivers):
        hist = np.bincount(positions[:, i].astype(int) - 1,
                           minlength=n_drivers)[:n_drivers]
        counts[i] = hist / positions.shape[0]
    return pd.DataFrame(counts, index=list(drivers),
                        columns=[f'P{i + 1}' for i in range(n_drivers)])


def main():
    """The console run: run() for the work, then everything it refuses to do."""
    os.makedirs(OUT_DIR, exist_ok=True)

    result = run()
    pace, track = result['pace'], result['track']
    strategies, dropped = result['strategies'], result['dropped']
    sc = track['sc']

    print(f'\n=== {TARGET_EVENT} {SEASON} | v1.2 | {N_SIMS:,} sims, '
          f'{track["n_laps"]} laps ===')
    print(f'Pole        : {POLE_DRIVER} {POLE_TIME:.3f} s   drivers: {len(pace)}')
    print(f'Grid        : {pace.attrs["grid_source"]}, '
          f'{track["grid_gap"]:.2f} s/slot, start sigma {track["start_sigma"]:.1f} '
          f'positions ({track["grid_source"]})')
    print(f'fuel_effect : {track["fuel_effect"]:.3f} s/lap')

    print(f'tyre model  : D(a) = b1*a + b2*a^2 + g*max(0,a-tau)^2   '
          f'v{track["tyre_model_version"]}')
    for c in COMPOUND_NAMES:
        p = track['tyre'][c]
        cliff = ('never' if p.tau is None
                 else f'{p.tau + AGE_OFFSET:.0f} laps')
        print(f'   {c:7s} b1 {p.b1:.4f}  b2 {p.b2:.5f}  g {p.gamma:.5f}  '
              f'cliff {cliff:>10s}  cap {max_stint_laps(p):>3d} laps  [{p.source}]')
        if p.notes:
            print(f'           {p.notes}')

    print(f'pit_loss    : {track["pit_loss"]:.1f} s')
    print(f'SC          : lambda {sc["sc_lambda"]:.2f} x {sc["sc_duration"]:.1f} laps | '
          f'VSC {sc["vsc_lambda"]:.2f} x {sc["vsc_duration"]:.1f} ({track["sc_source"]})')
    print(f'overtaking  : p_pass {track["pass_rate"]:.3f}/lap '
          f'({track["pass_source"]}), difficulty {track["overtake_difficulty"]}/5, '
          f'dirty air {track["dirty_air_penalty"]:.2f} s/lap over {DIRTY_AIR_RANGE:.1f} s')
    if pd.notna(track.get('pass_rate_wide')) and USE_NEAR_GAP_RATE:
        print(f'              anchored on the < {ATTACK_GAP:.2f} s window; the '
              f'full {1.5:.1f} s window reads {track["pass_rate_wide"]:.3f}')
    if GAP_IN_PASS_MODEL:
        m = build_pass_model(track)
        adv = np.array([REFERENCE_ADVANTAGE])
        near = float(pass_probability(adv, m, gap=np.array([0.25]))[0])
        far = float(pass_probability(adv, m, gap=np.array([0.95]))[0])
        print(f'              gap term on inside {ATTACK_GAP:.2f} s with '
              f'>= {GAP_MODEL_MIN_ADVANTAGE:.2f} s/lap in hand;')
        print(f'              at {REFERENCE_ADVANTAGE:.2f} s/lap, 0.25 s back '
              f'gives {near:.3f} and 0.95 s back {far:.3f}')
    if TEAM_PASS_ENABLED and 'team_pass_source' in track:
        print(f'              team term {track["team_pass_source"]}')

    unrunnable = [s for s in dropped
                  if s.get('drop_reason') == 'stint past compound life']
    if unrunnable:
        print(f'\nstrategies dropped as unrunnable ({len(unrunnable)} of '
              f'{len(dropped) + len(strategies)} feasible):')
        for s in unrunnable[:8]:
            seq = '-'.join(c[0] for c in s['sequence'])
            plan = '-'.join(str(x) for x in s['plan'])
            print(f'   {seq:8s} {plan:12s} stint past compound life')
        if len(unrunnable) > 8:
            print(f'   ... and {len(unrunnable) - 8} more')

    if AFFINITY_ENABLED and pace['affinity'].abs().sum() > 0:
        best = pace.nlargest(3, 'affinity')[['Driver', 'affinity', 'affinity_sec']]
        worst = pace.nsmallest(3, 'affinity')[['Driver', 'affinity', 'affinity_sec']]
        print(f'affinity    : {pace.attrs.get("affinity_source", "?")}')
        print(f'              team term {pace.attrs.get("affinity_team_source", "?")}')
        print(f'              {AFFINITY_PCT_PER_POSITION * 100:.3f}% of the lap '
              f'per position = {AFFINITY_PCT_PER_POSITION * POLE_TIME:.3f} s '
              f'at {POLE_TIME:.1f} s, cap {AFFINITY_CAP_PCT * POLE_TIME:.2f} s')
        print('              suits: ' + ', '.join(
            f"{r['Driver']} {r['affinity']:+.1f} ({r['affinity_sec']:+.3f}s)"
            for _, r in best.iterrows()))
        print('              hurts: ' + ', '.join(
            f"{r['Driver']} {r['affinity']:+.1f} ({r['affinity_sec']:+.3f}s)"
            for _, r in worst.iterrows()))

    if 'PLACEHOLDER' in pace.attrs['grid_source']:
        print('\n! No grid found. Run fetch.py to write data/f1_2026_grid.csv,')
        print('  otherwise the grid is just pace order and the result is circular.')

    positions, total = result['positions'], result['total']
    strat_idx, diag = result['strat_idx'], result['diag']

    neutral = diag['neutral']
    any_sc = (neutral == 2).any(axis=1)
    any_vsc = (neutral == 1).any(axis=1)

    print('\n--- neutralization ---')
    print(f'Races with SC : {any_sc.mean():.1%}   with VSC: {any_vsc.mean():.1%}')
    print(f'Stops under neutralization: {diag["sc_stops"].mean():.2f} per driver')

    first_sc = np.where(any_sc, (neutral == 2).argmax(axis=1) + 1, np.nan)
    if any_sc.any():
        print(f'First SC lap  : mean {np.nanmean(first_sc):.1f}, '
              f'p10-p90 {np.nanpercentile(first_sc, 10):.0f}-'
              f'{np.nanpercentile(first_sc, 90):.0f} of {track["n_laps"]}')
        # laps under SC in total, not the length of one event: a race can have
        # two, and a red flag overwrites whatever lap it lands on, which can
        # cut a period short
        sc_laps = (neutral == 2).sum(axis=1)[any_sc]
        print(f'SC laps total : mean {sc_laps.mean():.1f}, '
              f'min {sc_laps.min()}, max {sc_laps.max()}')
    if SC_BUNCHING:
        print(f'Field bunched : {diag["bunched"].mean():.2f} restarts per race '
              f'(queue gap {SC_QUEUE_GAP:.2f} s)')
    else:
        print('Field bunched : off - gaps only frozen, not closed')

    print('\n--- red flag ---')
    if RED_FLAG_ENABLED:
        had_rf = diag['rf_lap'] >= 0
        print(f'Races with one   : {had_rf.mean():.1%}   '
              f'(p {sc.get("rf_lambda", 0.0):.3f} per race, {track["sc_source"].split(",")[0]})')
        if had_rf.any():
            laps_rf = diag['rf_lap'][had_rf] + 1
            print(f'Restart at lap   : mean {laps_rf.mean():.1f}, '
                  f'p10-p90 {np.percentile(laps_rf, 10):.0f}-'
                  f'{np.percentile(laps_rf, 90):.0f} of {track["n_laps"]}')
            print(f'Free tyre changes: {diag["rf_stops"].mean():.3f} per driver '
                  f'({diag["rf_stops"][had_rf].mean():.2f} in the races that had one)')
        else:
            print('No red flag came out in any simulation.')
    else:
        print('Disabled - the race is never stopped.')

    if DNF_ENABLED:
        ret = diag['retired']
        print('\n--- retirements ---')
        print(f'Cars retired per race : {ret.sum(axis=1).mean():.2f}')
        print(f'Races with no retirement: {(~ret.any(axis=1)).mean():.1%}')

    print('\n--- racing ---')
    print(f'Overtakes per race : {diag["overtakes"].mean():.1f}')
    print(f'Laps held up, mean : {diag["laps_stuck"].mean():.1f} per driver')

    print('\n--- tyres ---')
    print(f'Laps run past the cliff : {diag["past_cliff"].mean():.1f} per driver')
    print(f'Time lost to wear       : {diag["tyre_loss"].mean():.1f} s per driver')
    print('If the first number is near zero the strategies never reach the')
    print('cliff, which means the curve is being priced but never paid.')

    print(f'\n--- strategy space ({len(strategies)}) ---')
    weights = np.asarray(diag['strat_weights'])
    field_share = weights.mean(axis=1) if weights.ndim == 2 else weights
    spread = (weights.max(axis=1) - weights.min(axis=1)
              if weights.ndim == 2 else np.zeros(len(strategies)))
    for i, (s, plan) in enumerate(zip(strategies, diag['plans'])):
        seq = '-'.join(c[0] for c in s['sequence'])
        line = (f"  {seq:8s} stints {'-'.join(str(x) for x in plan):12s} "
                f"cost {s['est_cost']:+8.1f}  share {field_share[i]:5.1%}")
        if PER_DRIVER_STRATEGY and weights.ndim == 2:
            line += f'  (driver spread {spread[i]:.1%})'
        print(line)

    if REACTIVE_PIT:
        print(f'\n--- reactive pit stops ---')
        print(f'Stops brought forward or held back by the tyre: '
              f'{diag["reactive_pits"].mean():.2f} per driver')
        print(f'Band: planned lap +/- {PIT_DECISION_BAND}')
    if ENFORCE_STINT_CAP:
        print(f'Stops forced by the measured stint cap: '
              f'{diag["forced_pits"].mean():.2f} per driver')

    print('\n--- longest stint actually run ---')
    print('If a number here exceeds its cap the cap is not being applied,')
    print('and no amount of reading the strategy table will show that.\n')
    caps = diag['cap_by_compound']
    for ci, c in enumerate(COMPOUND_NAMES):
        seen = int(diag['longest_stint'][ci])
        cap = caps[ci]
        cap_s = f'{cap:.0f}' if cap <= track['n_laps'] else 'none'
        flag = '   <-- OVER CAP' if (cap <= track['n_laps'] and seen > cap) else ''
        print(f'  {c:7s} longest {seen:3d} laps   cap {cap_s:>4s}{flag}')
    print(f'  longest stint of any compound: '
          f'{int(diag["longest_overall"].max())} laps')

    df = result['summary']
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print(f'\n{df.to_string(index=False)}')

    winner = total.min(axis=1).mean()
    print(f'\nMean winning race time: {format_time(winner)}  '
          f'({winner / 60:.2f} min)')
    spread = float((total.max(axis=1) - total.min(axis=1)).mean())
    print(f'Mean spread, winner to last: {spread:.1f} s')

    grid_arr = pace['grid'].to_numpy()
    mae = np.abs(positions - grid_arr[None, :]).mean()
    print(f'Mean shift from grid: {mae:.2f} positions')
    print('The naive benchmark from grid.py is 3.45 positions of error; a model')
    print('that never moves anyone cannot beat it, and neither can one that')
    print('scrambles the field at random.')

    csv_path = os.path.join(OUT_DIR, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f'Table saved: {csv_path}')

    # the mean finishing position hides the shape of the distribution, and the
    # shape is the thing worth looking at. A few KB of it.
    dist_path = os.path.join(OUT_DIR, 'position_distribution.csv')
    position_distribution(positions, pace['Driver']).to_csv(dist_path)
    print(f'Distribution saved: {dist_path}')

    plot_position_histograms(pace, positions, os.path.join(OUT_DIR, 'positions.png'))

    rep, report = result['rep'], result['report']
    trace, stints = result['trace'], result['stints']
    retired_lap = trace['retired_lap']

    if retired_lap is not None and (retired_lap >= 0).any():
        codes = pace['Driver'].to_numpy()
        outs = ', '.join(f'{codes[d]} (lap {int(retired_lap[d]) + 1})'
                         for d in np.flatnonzero(retired_lap >= 0))
        print(f'retired       : {outs}')

    print('\n--- representative race ---')
    print(f'simulation #{rep} of {N_SIMS:,}')
    print(f'event profile : {report["modal_retired"]} retirement(s), '
          f'SC {"yes" if report["modal_sc"] else "no"}  '
          f'-> {report["pool"]:,} races match')
    print(f'selection     : likelihood percentile '
          f'{report["loglik_percentile"]:.1%} within that pool, '
          f'{report["candidates"]} shortlisted')
    print(f'this race     : {report["overtakes"]} overtakes, '
          f'{report["stops"]:.2f} stops per driver, '
          f'{report["retired"]} retired, '
          f'{report["neutral_laps"]} neutralised laps')
    winner_code = pace['Driver'].to_numpy()[int(np.argmin(positions[rep]))]
    print(f'winner in that race: {winner_code}')

    param_lines = result['param_lines']
    subtitle = (f'sim #{rep}  |  {report["overtakes"]} overtakes  |  '
                f'{report["retired"]} retired  |  '
                f'{report["neutral_laps"]} neutralised laps')
    report_path = os.path.join(OUT_DIR, 'diagnostics.html')
    build_report(
        pace, positions, trace, stints, param_lines,
        f'{TARGET_EVENT} {SEASON}  -  {N_SIMS:,} simulations  -  v1.1-A',
        subtitle,
        report_path,
    )

    if OPEN_REPORT:
        # as_uri needs an absolute path, and a relative one silently opens nothing
        webbrowser.open(Path(report_path).resolve().as_uri())

    plt.show()


if __name__ == '__main__':
    main()