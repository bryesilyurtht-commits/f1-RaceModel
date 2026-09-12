"""
F1 Prediction Simulation - fit_tyre_curve.py
Fits D(a) = b1*a + b2*a^2 + gamma*max(0, a-tau)^2 per track and compound.

Reads  data/tyre_age_profile_track.csv   pooled age->loss profiles
       data/stint_index.csv              one row per stint, with how it ended
       data/stint_limits.csv             observed stint caps
       data/pit_summary.csv              pit loss per track
Writes data/tyre_curve_params.json       versioned parameter file

Why two stages
--------------
Lap times cannot see the cliff. Teams do not run a tyre off the edge, so the
degradation data is censored on the right: what the profiles contain is the
part of the tyre's life where it was still working. Fitting a cliff to that is
fitting a cliff to its own absence, and the previous attempt proved it - all
three compounds came back with the cliff parameters sitting on a grid boundary
and had to be hand-set.

Zandvoort is the clearest case. Measured from lap times alone, SOFT degrades at
0.003 s/lap and HARD at 0.032, so the soft tyre looks like the durable one.
Measured from when teams actually stop, SOFT ends at 19 laps, MEDIUM at 23 and
HARD at 30 - the physical order. The stopping decision carries the information
the lap times lost, because a team pitting at lap 19 knows something about that
tyre that its lap times were never allowed to show.

So:

  Stage 1  b1, b2   from the lap-time profile, inside the observed range only.
  Stage 2  tau, gamma from the stint-survival data, via the stopping rule.

Stage 2, in detail
------------------
A team ends a stint of length L when the next lap on the current tyre costs
more than the average lap of starting again, pit stop included:

    D(L+1)  >=  [ sum_{a=1..L} D(a) + P ] / L        P = pit loss, seconds

That gives a model-predicted stopping age L*. Run it with gamma = 0 and
compare against how long the tyre is actually used for:

  - the tyre is used LONGER than the no-cliff model predicts
        the curve is already steep enough; no cliff evidence, gamma = 0.
  - the tyre is retired EARLIER
        something is punishing it that the lap times do not show. That
        something is the cliff, and its severity is whatever reconciles the
        two numbers.

"How long it is actually used for" is the right edge of the stint-length
distribution, not its middle. Teams stop across a wide range and most of that
spread is strategy - undercuts, safety cars, traffic - none of which is the
tyre giving up. What the tyre decides is where the distribution ends. An
earlier version calibrated to the median instead and produced a cliff that
teams supposedly ignored for ten laps, pricing a 31-lap medium at 3.3 s/lap in
a cell where a quarter of real stints went that far.

tau and gamma are identified from different parts of the same data, which is
what keeps them apart: tau from the SHAPE of the stopping hazard (the age teams
start bailing out), gamma from its LEVEL (how hard they bail, priced against
the pit loss). Fitting both to a single summary number would leave them
trading off against each other, which is the boundary-solution failure again.

Censoring is handled properly. A stint that ran to the flag was never ended by
a decision, and neither was one that ended in a retirement - both say only
"at least this long" and enter the survival fit as censored, not as events.

Usage:
    python fit_tyre_curve.py
    python fit_tyre_curve.py --out data/tyre_curve_params.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tyre_curve import AGE_OFFSET, MODEL_NAME, TyreCurve

# --- config -----------------------------------------------------------------

# The repo root, two directories up from this file. See the note in
# simulate.py: the path used to be hardcoded to one machine.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

PARAMS_VERSION = '2.1.0'
COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

# 2022 moved to 18-inch wheels. Seasons either side of that are different
# rubber and are never averaged together; the older ones only fill cells the
# current era cannot measure at all.
ERA_BREAK = 2022
USE_LEGACY_ERA = True

# --- stage 1 ---
MIN_BINS = 4               # profile bins a cell needs before it is fitted
FALLBACK_PIT_LOSS = 22.0

# --- stage 2 ---
MIN_EVENTS = 10            # pit decisions needed before a cliff is considered
HAZARD_WINDOW = 3          # laps either side, smoothing the discrete hazard
HAZARD_RISE = 2.0          # hazard must reach this multiple of its early level
MIN_EVENTS_PAST_TAU = 5    # decisions past the knee, or the cliff is a guess
STOP_TOLERANCE = 1.0       # laps; disagreement smaller than this needs no cliff
GAMMA_MAX = 0.5            # s/lap^2; a harder cliff than this is a fit error

# --- SOFT rule ---
# Where SOFT cannot be measured its wear is taken from MEDIUM at the same
# track, and the shorter life is carried by the cap instead of the curve. A
# measured SOFT that comes out kinder than MEDIUM is treated as unmeasured:
# that ordering is not physically possible, so it is evidence about the data
# rather than about the tyre.
SOFT_MIN_STINTS = 8

# A soft set on the medium's wear curve has to be stopped earlier, or it
# becomes a medium in every respect and the compound choice stops meaning
# anything. The cap comes down to this share of the circuit's MEDIUM cap.
#
# The ratio is checked against the stopping data rather than picked: at
# Zandvoort the medium cap is 29, so this puts the soft at 18, and the age
# teams actually pit off softs there is 19. stint_limits.csv cannot supply
# this on its own - its p95 is dragged up by soft stints run behind a safety
# car, which is why SOFT and MEDIUM came out with the same 31-lap cap.
SOFT_CAP_RATIO = 0.62

# --- compound ordering ---
# A harder tyre that degrades faster than a softer one at the same circuit is
# not a finding, it is noise: the cells are fitted independently and nothing
# in the arithmetic knows that HARD is the durable one. Where the fitted
# curves cross, the harder compound is scaled down until it does not exceed
# the softer one. Scaling every coefficient by the same factor keeps the
# shape, the non-negativity and the continuity at tau intact.
ENFORCE_COMPOUND_ORDER = True
ORDER_REFERENCE_AGE = 15    # age used to decide whether SOFT measured sanely


# --- loading ----------------------------------------------------------------


def era_of(season):
    return '18inch' if int(season) >= ERA_BREAK else '13inch'


def load_inputs():
    # the season-level profile, not the pooled one: pooling across 2018-2025
    # would average 13-inch and 18-inch tyres into a single curve, and the two
    # are different rubber on different wheels
    prof = pd.read_csv(os.path.join(DATA_DIR, 'tyre_age_profile.csv'))
    if 'tyre_era' not in prof.columns:
        prof['tyre_era'] = prof['Season'].map(era_of)

    stints = pd.read_csv(os.path.join(DATA_DIR, 'stint_index.csv'))
    if 'tyre_era' not in stints.columns:
        stints['tyre_era'] = stints['Season'].map(era_of)

    limits = pd.read_csv(os.path.join(DATA_DIR, 'stint_limits.csv'))
    pits = pd.read_csv(os.path.join(DATA_DIR, 'pit_summary.csv'))

    # a second, independent estimate of how fast each compound wears, from
    # pit_analysis's within-driver panel. It is not as good as a stint profile
    # where the profile works, but it works where the profile does not: soft
    # stints are short and all run at the same phase of a race, which is
    # exactly the case the profile method cannot see and the panel can.
    panel = None
    path = os.path.join(DATA_DIR, 'compound_scaling.csv')
    if os.path.exists(path):
        panel = pd.read_csv(path)

    results = None
    for name in ('results_2018_2025.csv', 'results_2022_2025.csv'):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            frame = pd.read_csv(path)
            results = frame if results is None else pd.concat(
                [results, frame], ignore_index=True)
    if results is not None:
        results = results.drop_duplicates(subset=['Season', 'Race', 'Driver'])

    return prof, stints, limits, pits, results, panel


def panel_wear_table(panel):
    """
    Median panel degradation per track and compound, in s/lap of tyre age.

    Only the current era: compound_scaling.csv is built from SEASONS, which
    stops at 2022, so there is nothing older in here to keep out.
    """
    if panel is None:
        return {}
    out = {}
    for comp in COMPOUNDS:
        col = f'deg_{comp}'
        if col not in panel.columns:
            continue
        g = panel.dropna(subset=[col]).groupby('Race')[col].median()
        for race, v in g.items():
            if float(v) > 0:
                out[(str(race), comp)] = float(v)
    return out


def pool_profiles(prof, era):
    """
    Age -> loss per track and compound, pooled across the seasons of one era.

    Median across seasons rather than mean, for the same reason tyre_profile
    takes the median across stints: one wet-adjacent afternoon should not carry
    a circuit.
    """
    g = prof[prof['tyre_era'] == era]
    if g.empty:
        return g
    return (g.groupby(['Race', 'Compound', 'bin'])
             .agg(age=('age', 'first'),
                  loss_s=('loss_s', 'median'),
                  n_laps=('n_laps', 'sum'),
                  n_stints=('n_stints', 'sum'),
                  n_seasons=('Season', 'nunique'))
             .reset_index())


def mark_stint_ends(stints, results):
    """
    Flags every stint as an event (the team chose to pit) or as censored.

    Three ways a stint ends without a decision being made about the tyre: the
    race finished, the car retired, or the stint was never usable in the first
    place. Only the first of those is obvious in the stint table, so the
    classification joins the race results to find the retirements. Counting a
    blown engine as "the team judged this tyre done" would pull every cap down.
    """
    df = stints[stints['usable'] & stints['Compound'].isin(COMPOUNDS)].copy()

    df['ended_by_retirement'] = False
    if results is not None and 'Status' in results.columns:
        res = results.copy()
        status = res['Status'].astype(str)
        res['dnf'] = ~status.str.contains(r'Finished|\+\d+ Lap', regex=True, na=False)
        key = res.set_index(['Season', 'Race', 'Driver'])['dnf'].to_dict()
        dnf = np.array([bool(key.get((s, r, d), False))
                        for s, r, d in zip(df['Season'], df['Race'], df['Driver'])])
        last = (df.groupby(['Season', 'Race', 'Driver'])['Stint']
                  .transform('max') == df['Stint']).to_numpy()
        df['ended_by_retirement'] = dnf & last & ~df['runs_to_flag'].to_numpy()

    df['event'] = ~df['runs_to_flag'].to_numpy() & ~df['ended_by_retirement'].to_numpy()
    # curve ages, not FastF1 TyreLife
    df['end_age'] = df['tyre_life_end'].astype(float) - AGE_OFFSET
    df = df[df['end_age'] >= 1]
    return df


def cap_table(limits):
    """Observed stint caps in curve-age units, keyed by (track, compound)."""
    caps = {}
    for _, r in limits.iterrows():
        v = r.get('limit')
        if pd.notna(v):
            caps[(str(r['Race']), str(r['Compound']))] = max(int(round(float(v) - AGE_OFFSET)), 1)
    return caps


def pit_loss_table(pits):
    out = {}
    for _, r in pits.iterrows():
        if pd.notna(r.get('pit_loss')):
            out[str(r['Race'])] = float(r['pit_loss'])
    return out


# --- stage 1: b1, b2 from the lap-time profile ------------------------------


def fit_linear_quadratic(ages, losses, weights=None):
    """
    Least squares for D(a) = b1*a + b2*a^2 with both coefficients >= 0.

    No intercept: the profile is already anchored at its own zero, and letting
    the fit move that would reintroduce the offset the curve deliberately
    fixes.

    Non-negativity is imposed by trying the full fit, then each restriction in
    turn, and keeping the best feasible one. With two parameters that is four
    cases, which is cheaper and more predictable than a solver.
    """
    a = np.asarray(ages, dtype=float)
    y = np.asarray(losses, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    w = np.sqrt(np.maximum(w, 0.0))

    def sse(b1, b2):
        return float((w * w * (y - (b1 * a + b2 * a * a)) ** 2).sum())

    candidates = []

    # both free
    X = np.column_stack([a, a * a]) * w[:, None]
    try:
        beta, *_ = np.linalg.lstsq(X, y * w, rcond=None)
        if beta[0] >= 0 and beta[1] >= 0:
            candidates.append((sse(beta[0], beta[1]), float(beta[0]), float(beta[1])))
    except np.linalg.LinAlgError:
        pass

    # b2 = 0, linear only
    denom = float(((w * a) ** 2).sum())
    if denom > 1e-12:
        b1 = float((w * w * a * y).sum() / denom)
        b1 = max(b1, 0.0)
        candidates.append((sse(b1, 0.0), b1, 0.0))

    # b1 = 0, quadratic only
    denom2 = float(((w * a * a) ** 2).sum())
    if denom2 > 1e-12:
        b2 = float((w * w * a * a * y).sum() / denom2)
        b2 = max(b2, 0.0)
        candidates.append((sse(0.0, b2), 0.0, b2))

    # flat
    candidates.append((sse(0.0, 0.0), 0.0, 0.0))

    candidates.sort(key=lambda t: t[0])
    _, b1, b2 = candidates[0]
    return b1, b2


def fit_one_profile(g, cap):
    """b1, b2 for one pooled profile, or None with a reason."""
    g = g.sort_values('age').copy()
    g['a'] = g['age'].astype(float) - AGE_OFFSET
    if cap is not None:
        g = g[g['a'] <= cap]
    g = g[g['a'] >= 0]

    if len(g) < MIN_BINS:
        return None, None, len(g), f'only {len(g)} usable bins'

    w = g['n_laps'].to_numpy(float) if 'n_laps' in g.columns else None
    b1, b2 = fit_linear_quadratic(g['a'], g['loss_s'], w)
    return b1, b2, len(g), ''


def stage1(prof, caps):
    """
    b1 and b2 per cell, fitted inside the observed stint range.

    The current era is fitted first and wins wherever it has enough bins. The
    13-inch seasons are there to fill what 2022-2025 cannot answer - the cells
    with too few stints to measure, which is where the SOFT problem lives -
    and never to dilute a cell that the modern data already covers. A curve
    borrowed across the wheel-size change is marked as such, because a 2019
    tyre is evidence about a 2019 tyre.
    """
    modern = pool_profiles(prof, '18inch')
    legacy = pool_profiles(prof, '13inch')

    keys = set()
    for frame in (modern, legacy):
        if not frame.empty:
            keys |= set(map(tuple, frame[['Race', 'Compound']].drop_duplicates().values))

    out = {}
    for race, comp in sorted(keys):
        if comp not in COMPOUNDS:
            continue
        cap = caps.get((race, comp))

        b1 = b2 = None
        n_bins, reason, era = 0, '', None

        if not modern.empty:
            g = modern[(modern['Race'] == race) & (modern['Compound'] == comp)]
            if not g.empty:
                b1, b2, n_bins, reason = fit_one_profile(g, cap)
                era = '18inch'

        if b1 is None and not legacy.empty and USE_LEGACY_ERA:
            g = legacy[(legacy['Race'] == race) & (legacy['Compound'] == comp)]
            if not g.empty:
                lb1, lb2, ln, lreason = fit_one_profile(g, cap)
                if lb1 is not None:
                    b1, b2, n_bins, reason, era = lb1, lb2, ln, '', '13inch'

        out[(race, comp)] = {'b1': b1, 'b2': b2, 'n_bins': n_bins,
                             'era': era, 'reason': reason}
    return out


# --- stage 2: the stopping model --------------------------------------------


def kaplan_meier(end_ages, events):
    """
    Survival of "still running on this tyre" against age.

    Events are pit decisions. Censored rows - ran to the flag, or retired -
    contribute to the risk set up to their age and then leave without an event,
    which is exactly what "we know it lasted at least this long" means.

    Returns (ages, survival, hazard) on the observed event ages.
    """
    a = np.asarray(end_ages, dtype=float)
    e = np.asarray(events, dtype=bool)

    ages = np.unique(a[e])
    surv, haz, s = [], [], 1.0
    for t in ages:
        at_risk = float((a >= t).sum())
        d = float(((a == t) & e).sum())
        if at_risk <= 0:
            continue
        h = d / at_risk
        s *= (1.0 - h)
        surv.append(s)
        haz.append(h)
    return ages, np.array(surv), np.array(haz)


def km_quantile(ages, surv, q=0.5):
    """Age at which survival first drops to or below 1-q."""
    if len(ages) == 0:
        return None
    target = 1.0 - q
    below = np.flatnonzero(surv <= target)
    if len(below) == 0:
        return None
    return float(ages[below[0]])


def find_knee(df_cell):
    """
    The age at which teams start bailing out of the tyre.

    The discrete hazard is noisy at one-lap resolution - a handful of stints
    per age - so it is smoothed over a window before the knee is looked for.
    The knee is the first age whose smoothed hazard reaches HAZARD_RISE times
    the level seen early in the stint, with enough decisions still to come
    after it that the cliff is being measured rather than assumed.
    """
    a = df_cell['end_age'].to_numpy(float)
    e = df_cell['event'].to_numpy(bool)
    if e.sum() < MIN_EVENTS:
        return None, 'too few pit decisions'

    top = int(a.max())
    grid = np.arange(1, top + 1)
    smooth = np.zeros(len(grid))
    for i, t in enumerate(grid):
        lo, hi = t - HAZARD_WINDOW, t + HAZARD_WINDOW
        at_risk = float((a >= lo).sum())
        d = float(((a >= lo) & (a <= hi) & e).sum())
        smooth[i] = d / at_risk if at_risk > 0 else 0.0

    # the early level: the first third of the observed range, where the tyre
    # is still young and nobody is stopping because of it
    early_end = max(int(top * 0.33), 2)
    early = smooth[grid <= early_end]
    base = float(np.median(early)) if len(early) else 0.0
    if base <= 1e-6:
        base = float(np.mean(smooth[smooth > 0])) * 0.5 if (smooth > 0).any() else 0.0
    if base <= 1e-6:
        return None, 'hazard is flat at zero'

    for i, t in enumerate(grid):
        if t <= early_end:
            continue
        if smooth[i] >= HAZARD_RISE * base:
            past = int(((a > t) & e).sum())
            if past < MIN_EVENTS_PAST_TAU:
                return None, f'knee at {t} has only {past} decisions past it'
            return float(t), ''
    return None, 'hazard never rises'


def optimal_stop(b1, b2, tau, gamma, pit_loss, horizon):
    """
    The age at which staying out stops paying.

    A stint of length L costs the wear it accumulates plus the stop that ends
    it. Spread over its own laps that is (sum of D + P) / L, and the stint is
    worth extending exactly while the next lap costs less than that average.
    The first lap where it does not is where a team stops.
    """
    total = 0.0
    for L in range(1, horizon + 1):
        d_next = b1 * (L + 1) + b2 * (L + 1) ** 2
        if tau is not None and (L + 1) > tau:
            d_next += gamma * ((L + 1) - tau) ** 2
        d_L = b1 * L + b2 * L * L
        if tau is not None and L > tau:
            d_L += gamma * (L - tau) ** 2
        total += d_L
        if d_next >= (total + pit_loss) / L:
            return L
    return horizon


def solve_gamma(b1, b2, tau, pit_loss, target_stop, horizon):
    """
    The cliff severity that makes the model stop where the teams stop.

    L* falls as gamma rises, so a bisection finds it. If even gamma = 0 already
    stops early enough there is nothing to explain and the answer is zero.
    """
    if optimal_stop(b1, b2, tau, 0.0, pit_loss, horizon) <= target_stop + STOP_TOLERANCE:
        return 0.0, 'no cliff needed'

    lo, hi = 0.0, GAMMA_MAX
    if optimal_stop(b1, b2, tau, hi, pit_loss, horizon) > target_stop:
        return None, f'even gamma={GAMMA_MAX} does not stop by {target_stop:.0f}'

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if optimal_stop(b1, b2, tau, mid, pit_loss, horizon) > target_stop:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2.0, ''


def stage2(stints, stage1_params, caps, pit_losses):
    """tau and gamma per cell, from when teams actually stop."""
    out = {}
    for (race, comp), g in stints.groupby(['Race', 'Compound']):
        if comp not in COMPOUNDS:
            continue
        p1 = stage1_params.get((race, comp))
        info = {'tau': None, 'gamma': 0.0, 'n_events': int(g['event'].sum()),
                'n_stints': len(g), 'observed_stop': None, 'median_stop': None,
                'nocliff_stop': None, 'reason': ''}

        if p1 is None or p1['b1'] is None:
            info['reason'] = 'no stage-1 curve'
            out[(race, comp)] = info
            continue

        ages, surv, _ = kaplan_meier(g['end_age'], g['event'])
        info['median_stop'] = km_quantile(ages, surv, 0.5)

        # The target is the right edge of the stopping distribution, not its
        # middle.
        #
        # Teams stop across a wide range - at Zandvoort mediums come off
        # anywhere from lap 20 to lap 33 - and most of that spread is
        # strategy: undercuts, safety cars, traffic. None of it is the tyre
        # giving up. What the tyre decides is where the distribution ENDS,
        # because past that point the set is finished whatever the strategy
        # wanted.
        #
        # Calibrating gamma to the median instead made the model assert that
        # teams routinely run ten laps past the cliff, and priced a 31-lap
        # medium at 3.3 s/lap - a tyre nobody would ever take there, in a
        # cell where 28% of real stints did. Using the same quantile the cap
        # comes from keeps the cliff and the cap telling one story.
        cap = caps.get((race, comp), int(g['end_age'].max()))
        obs = float(cap)
        info['observed_stop'] = obs

        horizon = max(int(cap * 2), 20)
        P = pit_losses.get(race, FALLBACK_PIT_LOSS)

        nocliff = optimal_stop(p1['b1'], p1['b2'], None, 0.0, P, horizon)
        info['nocliff_stop'] = nocliff

        if info['n_events'] < MIN_EVENTS:
            info['reason'] = f"only {info['n_events']} pit decisions"
            out[(race, comp)] = info
            continue
        if nocliff <= obs + STOP_TOLERANCE:
            info['reason'] = (f'the tyre is used to {obs:.0f} and the cliff-free '
                              f'curve already stops at {nocliff} - the measured '
                              f'wear explains the stint length on its own')
            out[(race, comp)] = info
            continue

        tau, why = find_knee(g)
        if tau is None:
            info['reason'] = f'no identifiable knee ({why})'
            out[(race, comp)] = info
            continue
        if tau >= obs:
            info['reason'] = f'knee at {tau:.0f} is past the usable limit {obs:.0f}'
            out[(race, comp)] = info
            continue

        gamma, why = solve_gamma(p1['b1'], p1['b2'], tau, P, obs, horizon)
        if gamma is None:
            info['reason'] = why
            out[(race, comp)] = info
            continue

        info['tau'] = tau
        info['gamma'] = gamma
        info['reason'] = 'measured'
        out[(race, comp)] = info
    return out


# --- assembly ---------------------------------------------------------------


def soft_cap_for(race, caps, medium_cap):
    """
    The cap a soft set gets once it is running on the medium's wear curve.

    Its own measured cap is kept only if it is already below the ratio; where
    stint_limits.csv handed SOFT the same number as MEDIUM - which happens
    whenever long soft stints behind a safety car survive into the p95 - the
    ratio wins. Without that the soft becomes a medium with a different name.
    """
    own = caps.get((race, 'SOFT'))
    scaled = (max(int(round(medium_cap * SOFT_CAP_RATIO)), 1)
              if medium_cap is not None else None)
    if own is not None and scaled is not None:
        return min(own, scaled)
    return own if own is not None else scaled


def _loss_at(b1, b2, tau, gamma, a):
    d = b1 * a + b2 * a * a
    if tau is not None and a > tau:
        d += gamma * (a - tau) ** 2
    return d


def scale_curve(curve, factor, why):
    """Multiply a curve down. D scales linearly, so the shape survives."""
    curve.b1 *= factor
    curve.b2 *= factor
    curve.gamma *= factor
    if curve.gamma == 0.0:
        curve.tau = None
    curve.source = 'order-clipped' if curve.source == 'measured' else curve.source
    curve.notes = (curve.notes + '; ' if curve.notes else '') + why
    return curve


def enforce_compound_order(curves):
    """
    D_SOFT >= D_MEDIUM >= D_HARD at every age both curves are valid for.

    Where a harder compound comes out steeper, it is scaled down by the worst
    ratio on the shared range. MEDIUM is trusted over HARD because its cliff
    is identified far more often - 23 of 25 cells against 7 of 24 - so its
    curve is the better measured of the two.
    """
    by_track = {}
    for c in curves:
        by_track.setdefault(c.track, {})[c.compound] = c

    fixed = 0
    for track, cs in by_track.items():
        # softest pair first. Each clip has to see the neighbour's final
        # curve: clipping HARD against a MEDIUM that is itself about to be
        # scaled down leaves HARD above the MEDIUM it ends up next to, which
        # is how four circuits kept an inverted pair through the first pass.
        for softer, harder in (('SOFT', 'MEDIUM'), ('MEDIUM', 'HARD')):
            a_c, b_c = cs.get(softer), cs.get(harder)
            if a_c is None or b_c is None:
                continue
            top = min(a_c.max_age, b_c.max_age)
            if top < 1:
                continue

            worst = 1.0
            for a in range(1, top + 1):
                hard_loss = _loss_at(b_c.b1, b_c.b2, b_c.tau, b_c.gamma, a)
                soft_loss = _loss_at(a_c.b1, a_c.b2, a_c.tau, a_c.gamma, a)
                if hard_loss > 1e-9 and hard_loss > soft_loss:
                    worst = min(worst, soft_loss / hard_loss)

            if worst < 1.0 - 1e-9:
                scale_curve(b_c, worst,
                            f'scaled to {worst:.2f} so it does not exceed '
                            f'{softer} at this circuit')
                fixed += 1
    return fixed


def assemble(stage1_params, stage2_params, caps, stints, panel_wear=None):
    """
    One curve per track and compound, with where each number came from.

    The SOFT rule lives here. A soft tyre that cannot be measured, or that
    measures kinder than the medium at the same circuit, takes the medium's
    wear curve and keeps its own much shorter cap. The shortness of a soft
    stint is then carried by the cap, which is an observation, rather than by
    a degradation number the data never supported.
    """
    panel_wear = panel_wear or {}
    tracks = sorted({r for r, _ in stage1_params})
    stint_counts = stints.groupby(['Race', 'Compound']).size().to_dict()
    curves, report = [], []

    for race in tracks:
        med = stage1_params.get((race, 'MEDIUM'))
        med2 = stage2_params.get((race, 'MEDIUM'), {})
        med_cap = caps.get((race, 'MEDIUM'))

        for comp in COMPOUNDS:
            p1 = stage1_params.get((race, comp))
            p2 = stage2_params.get((race, comp), {})
            cap = caps.get((race, comp))
            n_stints = stint_counts.get((race, comp), 0)

            source, notes = 'measured', ''
            b1 = p1['b1'] if p1 else None
            b2 = p1['b2'] if p1 else None
            if p1 and p1.get('era') == '13inch':
                source = 'legacy-era'
                notes = ('fitted on 13-inch seasons; 2022-2025 has too little '
                         'of this compound here to measure')
            tau = p2.get('tau')
            gamma = p2.get('gamma', 0.0)

            if comp == 'SOFT':
                med_b1 = med['b1'] if med and med['b1'] is not None else None
                too_few = n_stints < SOFT_MIN_STINTS
                unmeasured = b1 is None

                # compared on total loss, not on b1: a cell that carries a
                # cliff puts part of its wear in gamma, so the linear terms of
                # two cells are not the same quantity
                inverted = False
                if b1 is not None and med_b1 is not None:
                    ref = ORDER_REFERENCE_AGE
                    if cap is not None:
                        ref = min(ref, cap)
                    mine = _loss_at(b1, b2, tau, gamma, ref)
                    theirs = _loss_at(med['b1'], med['b2'], med2.get('tau'),
                                      med2.get('gamma', 0.0), ref)
                    inverted = mine < theirs

                if (unmeasured or too_few or inverted) and med_b1 is not None:
                    reason = ('no soft profile' if unmeasured else
                              f'only {n_stints} soft stints' if too_few else
                              f'measures kinder than MEDIUM at age {ref}')
                    cap = soft_cap_for(race, caps, med_cap)

                    # The panel is asked first. Copying MEDIUM's curve keeps
                    # the soft's pace advantage - which is measured, and real -
                    # while deleting the wear that pays for it, so the soft
                    # ends up strictly better than the medium and the medium
                    # stops being used at all. The panel measures the wear
                    # directly and puts the bill back.
                    panel_b1 = panel_wear.get((race, 'SOFT'))
                    panel_med = panel_wear.get((race, 'MEDIUM'))
                    if (panel_b1 is not None and panel_med is not None
                            and panel_b1 > panel_med):
                        b1, b2 = panel_b1, 0.0
                        tau, gamma = None, 0.0
                        source = 'panel-wear'
                        notes = (f'wear from the compound_scaling panel '
                                 f'({reason}); the stint profile cannot see '
                                 f'this compound here')
                    else:
                        b1, b2 = med_b1, med['b2']
                        tau, gamma = med2.get('tau'), med2.get('gamma', 0.0)
                        source = 'medium-copy'
                        notes = (f'wear taken from MEDIUM at this circuit '
                                 f'({reason}); the panel has nothing usable '
                                 f'either, so the shorter soft life is carried '
                                 f'by max_age alone')

            if b1 is None or cap is None:
                report.append({'Race': race, 'Compound': comp, 'status': 'dropped',
                               'reason': (p1 or {}).get('reason', 'no cap')})
                continue

            if tau is not None and tau >= cap:
                notes = (notes + '; ' if notes else '') + \
                    f'cliff at {tau:.0f} sits past the cap, dropped'
                tau, gamma = None, 0.0

            curve = TyreCurve(track=race, compound=comp, b1=b1, b2=b2,
                              tau=tau, gamma=gamma, max_age=cap,
                              source=source, notes=notes)
            curves.append(curve)
            report.append({
                'Race': race, 'Compound': comp, 'status': 'ok', 'source': source,
                'b1': round(b1, 5), 'b2': round(b2, 6),
                'tau': tau, 'gamma': None if gamma == 0 else round(gamma, 5),
                'max_age': cap,
                'usable_limit': p2.get('observed_stop'),
                'median_stop': p2.get('median_stop'),
                'nocliff_stop': p2.get('nocliff_stop'),
                'cliff_reason': p2.get('reason', ''),
            })

    return curves, pd.DataFrame(report)


def enforce_cap_ordering(curves):
    """
    SOFT <= MEDIUM <= HARD on the cap, per circuit.

    A soft set that outlasts a hard one at the same track is a data artefact -
    usually a stint run mostly behind a safety car - and stint_limits.py
    already clips the worst of it. This repeats the check after the SOFT rule
    has had its say, because a borrowed cap can reintroduce the inversion.
    """
    by_track = {}
    for c in curves:
        by_track.setdefault(c.track, {})[c.compound] = c

    fixed = 0
    for track, cs in by_track.items():
        hard = cs.get('HARD')
        med = cs.get('MEDIUM')
        soft = cs.get('SOFT')
        if med and hard and med.max_age > hard.max_age:
            med.max_age = hard.max_age
            med.notes = (med.notes + '; ' if med.notes else '') + 'cap clipped to HARD'
            fixed += 1
        if soft and med and soft.max_age > med.max_age:
            soft.max_age = med.max_age
            soft.notes = (soft.notes + '; ' if soft.notes else '') + 'cap clipped to MEDIUM'
            fixed += 1
    return fixed


# --- output -----------------------------------------------------------------


def write_params(curves, path):
    payload = {
        'model_version': PARAMS_VERSION,
        'model': MODEL_NAME,
        'formula': 'D(a) = beta_1*a + beta_2*a^2 + gamma*max(0, a-tau)^2',
        'units': {'loss': 'seconds', 'age': 'laps'},
        'age_offset': AGE_OFFSET,
        'age_definition': (
            'a = FastF1 TyreLife - age_offset. The curve zero is the profile '
            'baseline (median of a stint\'s opening laps), not the tyre\'s '
            'first lap, so D(0) is a tyre that has already run age_offset laps.'),
        'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'deterministic': True,
        'curves': [c.as_dict() for c in sorted(curves, key=lambda c: (c.track, c.compound))],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DATA_DIR, 'tyre_curve_params.json'))
    ap.add_argument('--report', default=os.path.join(DATA_DIR, 'tyre_curve_report.csv'))
    args = ap.parse_args()

    prof, stints_raw, limits, pits, results, panel = load_inputs()
    stints = mark_stint_ends(stints_raw, results)
    caps = cap_table(limits)
    pit_losses = pit_loss_table(pits)
    panel_wear = panel_wear_table(panel)

    print(f'profiles : {prof.groupby(["Race", "Compound"]).ngroups} cells')
    print(f'stints   : {len(stints)} usable, '
          f'{int(stints["event"].sum())} ended by a pit decision, '
          f'{int((~stints["event"]).sum())} censored')
    print(f'caps     : {len(caps)}   pit losses: {len(pit_losses)}\n')

    s1 = stage1(prof, caps)
    fitted = sum(1 for v in s1.values() if v['b1'] is not None)
    print(f'stage 1  : {fitted} of {len(s1)} cells fitted')

    s2 = stage2(stints, s1, caps, pit_losses)
    with_cliff = sum(1 for v in s2.values() if v['tau'] is not None)
    print(f'stage 2  : {with_cliff} of {len(s2)} cells carry a measured cliff')

    print(f'panel     : {len(panel_wear)} track x compound wear estimates')
    curves, report = assemble(s1, s2, caps, stints, panel_wear)
    clipped = enforce_cap_ordering(curves)
    scaled = enforce_compound_order(curves) if ENFORCE_COMPOUND_ORDER else 0
    print(f'assembly : {len(curves)} curves, {clipped} caps clipped, '
          f'{scaled} curves scaled for compound order')

    # the report is written from the finished curves, so a number in it is the
    # number the simulation will actually use
    final = {(c.track, c.compound): c for c in curves}
    if not report.empty:
        for i, row in report.iterrows():
            c = final.get((row['Race'], row['Compound']))
            if c is None:
                continue
            report.loc[i, ['source', 'b1', 'b2', 'tau', 'gamma', 'max_age', 'notes']] = [
                c.source, round(c.b1, 5), round(c.b2, 6), c.tau,
                None if c.gamma == 0 else round(c.gamma, 5), c.max_age, c.notes]

    by_source = {}
    for c in curves:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print(f'sources  : {by_source}')

    payload = write_params(curves, args.out)
    report.to_csv(args.report, index=False)
    print(f'\nSaved: {args.out}  (version {payload["model_version"]})')
    print(f'Saved: {args.report}')

    print('\n--- target circuit ---')
    for c in curves:
        if 'dutch' in c.track.lower():
            print(f'  {c!r}')
            print(f'      source={c.source}  {c.notes}')


if __name__ == '__main__':
    main()
