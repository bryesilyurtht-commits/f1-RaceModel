"""
F1 Prediction Simulation - track_features.py
Measures what kind of circuit each track is, from qualifying telemetry.

Why this exists
---------------
track_affinity.py measures which circuits suit which teams, and it does that
from results alone: two or three races per circuit, and Monaco has three. That
is not enough to separate a real effect from a good weekend. If circuits can be
described by what they physically demand, then similar circuits can lend each
other data - Monaco's affinity can borrow from Singapore and Baku, and not from
Monza.

This file only measures. The pooling that uses these numbers is stage two, in
track_affinity.py, and it is not worth writing until the diagnostics below come
out sane.

Everything is a ratio, never an absolute
----------------------------------------
2026 changed the cars completely - active aero, a new power unit - so an
absolute speed measured in 2025 says nothing about 2026. A corner taken at
80 km/h is meaningless; the same corner taken at 24% of the lap's top speed is
a property of the corner. Every speed-derived feature here is divided by that
lap's own top speed, which is what makes the measurement survive the rule
change.

The curvature features are better still: they come from the X/Y trace and never
touch the speed channel at all, so no car influences them whatsoever.

The check this has to pass
--------------------------
Monaco's nearest neighbours should be Singapore and Baku. Monza's should be
somewhere near Spa and Baku. If Monza and Monaco come out next to each other
the features are wrong and stage two must not be written.

Outputs:
    data/track_features.csv          per circuit, raw features and z-scores
    data/track_features_scaling.csv  the mean and sd each feature was scaled by

Usage:
    python -m Simülasyon.data_prep.track_features
"""

import os
import warnings

import numpy as np
import pandas as pd

# fastf1 is imported inside collect(), not here. It is only needed to download
# telemetry, and this module also carries FEATURE_SET and standardise - two
# pure definitions that team_affinity.py reads. A top-level import made the
# whole no-network path depend on a package requirements.txt deliberately
# leaves out, so a fresh clone could not run the team affinity tests.

warnings.filterwarnings('ignore')

# scipy is installed here but unusable: the compiled extensions it needs
# (_distance_pybind, _slsqplib, cython_blas) are blocked by this machine's
# application control policy, which takes signal, spatial, cluster and stats
# down with them. The three things this file wanted from scipy - a prominence
# based peak finder, a distance matrix and Ward clustering - are written out
# below in numpy instead. They are short enough that the dependency was never
# worth much anyway.

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASON = 2025
SESSION_TYPE = 'Q'

WET_COMPOUND_SHARE_LIMIT = 0.2   # above this the session is wet, circuit skipped
SMOOTH_WINDOW_M = 25             # speed-trace smoothing window, metres
MIN_PROMINENCE_FRACTION = 0.08   # a local minimum must dip this share of top speed
RESAMPLE_STEP_M = 5.0            # uniform distance grid for the geometry work

FULL_THROTTLE_PCT = 98           # throttle at or above this counts as flat out
DRS_OPEN_FROM = 10               # FastF1 codes 10, 12, 14 mean the flap is open

# A straight is not the same thing as a throttle pedal being down.
#
# full_throttle_share counts every metre at full throttle, including the
# 150-metre snatches between corners on a flowing circuit. longest_flat_out
# counts only the single longest run, which misses Monza entirely: its lap is
# almost all straight, but chicanes chop it into several pieces so no single
# piece stands out.
#
# What separates the two is a length threshold. Runs above it are real
# straights - somewhere to use the engine, take DRS, plan a move. Runs below
# it are corner exits. Three thresholds are measured so the choice can be made
# from the correlation matrix rather than by assertion.
SUSTAINED_THRESHOLDS_M = (300.0, 500.0, 700.0)

CORRELATION_FLAG = 0.85

# The four axes the pooling actually uses. Everything else measured below is
# either a diagnostic or was dropped for being a restatement of one of these -
# see the correlation matrix this prints. Change this list only after looking
# at that matrix.
FEATURE_SET = [
    'corner_speed_median',    # how fast the slow parts are
    'straight_blend',         # straight-line character, see STRAIGHT_BLEND
    'braking_share',          # how much of the lap is spent stopping
    'curvature_p90',          # geometry of the tightest corners, speed-free
]

# The straight-line axis is two measurements mixed, not one.
#
# sustained_500 - how much of the lap sits in runs over 500 m - is the honest
# answer to "is this a straight-line circuit", and it is what finally put Monza
# where it belongs. But it correlates 0.75 with curvature_p90 and 0.64 with
# braking_share, so on its own it reinforces the axis those two already build:
# adding it pushed PC1 from 57% to 70% of the variance and squashed the map
# toward a single line.
#
# longest_flat_out has the opposite problem and the opposite virtue. It asks
# the narrower question - is there ONE long straight - and gets Monza wrong,
# but it correlates with nothing above 0.33, so it carries genuinely separate
# information.
#
# Mixing them keeps most of the first and borrows the independence of the
# second. The weights are not free parameters picked to taste: the ratio was
# scanned and 0.75/0.25 is the last point at which Monza still sits on the
# positive side of the straight axis. Past it the map balances slightly
# further (PC1 67% at 0.70) but Monza slides back to negative, which is the
# error this whole axis was rebuilt to fix.
#
# Lightening sustained_500 with a plain weight was tried first and does the
# reverse of what it looks like: it is the feature carrying PC2, so weakening
# it hands the map back to the correlated trio and PC1 grows to 73%.
STRAIGHT_BLEND = {'sustained_500': 0.75, 'longest_flat_out': 0.25}

# The straight-line axis has now been through three versions, and the reason
# it moved twice is worth keeping.
#
# full_throttle_share was first and correlates -0.86 with curvature_p90: it
# counts every metre at full throttle, including the 150-metre snatches
# between corners, so on this sample it was mostly restating "the corners are
# fast" rather than "there are straights".
#
# longest_flat_out replaced it and decorrelated cleanly (0.24), but it asks a
# narrower question than it appears to: is there ONE long straight. Monza
# proves the difference. Its lap is 82% full throttle and 74% of it sits in
# runs over 500 m - the highest in the field - but chicanes chop that into
# five pieces, so its longest single run is 19% and ranks thirteenth. The map
# read Monza as a short-burst circuit, which is the opposite of what it is.
#
# sustained_500 asks the question the others were standing in for: how much of
# the lap is spent in a run long enough to be a straight - somewhere to use the
# engine, take DRS, plan a move. Monza tops it at 74%, Monaco and Budapest sit
# at the bottom near 13%. It costs some orthogonality (0.75 against
# curvature_p90, against 0.24 for longest_flat_out, and PC1 grows from 57% to
# 70% of the variance) and that trade is deliberate: a feature that means the
# right thing beats one that decorrelates better and measures the wrong thing.
#
# The 300 m and 700 m variants are measured too and stay available; 500 m sits
# between a long corner exit and a DRS zone, which is the boundary the feature
# is trying to draw.

# full_throttle_share was the obvious choice for the power axis and it is not
# in the list. Measured over the 23 circuits it correlates with curvature_p90
# at -0.86, which is above the flag threshold below and is not a surprise: a
# lap with tight corners cannot also be spent at full throttle, so the two
# were largely restating each other. With both in, the first principal
# component carried 77% of the variance - four named axes behaving like one
# and a half.
#
# longest_flat_out asks a narrower question. Not how much of the lap is flat
# out, but whether there is one long uninterrupted run of it, which is what
# separates a circuit with a single big straight from one with the same total
# throttle spread over several short bursts. It correlates with nothing here
# above 0.33, and swapping it in drops the worst pair to 0.71 and PC1 to 57%.

# Altitude is deliberately absent. Mexico sits at 2200 m and the next highest
# circuit is around 700 m, so the variable is a Mexico dummy wearing a physical
# name. One observation does not teach a coefficient, it just gets memorised.

ALL_FEATURES = [
    'corner_speed_median', 'corner_speed_p10', 'n_corners',
    'full_throttle_share', 'longest_flat_out', 'mean_over_top_speed',
    'sustained_300', 'sustained_500', 'sustained_700', 'straight_blend',
    'n_braking_zones', 'braking_share',
    'curvature_median', 'curvature_p90', 'direction_changes',
    'lap_distance', 'n_drs_zones',
]


# --- numpy stand-ins for the blocked scipy pieces ---------------------------


def find_valleys(values, prominence):
    """
    Local minima that dip at least `prominence` below their surroundings.

    Prominence for a valley is the mirror of the usual definition for a peak:
    walk out in each direction until the trace drops below this valley - which
    means a deeper valley owns the territory - and take the highest point
    reached on the way. The shallower of the two climbs is how far this valley
    stands out. A plain "lower than both neighbours" test would return a local
    minimum every few metres on a noisy trace; this returns corners.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3:
        return np.array([], dtype=int)

    lower = (v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:])
    candidates = np.flatnonzero(lower) + 1

    keep = []
    for i in candidates:
        left_max = -np.inf
        j = i - 1
        while j >= 0 and v[j] >= v[i]:
            left_max = max(left_max, v[j])
            j -= 1
        right_max = -np.inf
        j = i + 1
        while j < n and v[j] >= v[i]:
            right_max = max(right_max, v[j])
            j += 1

        # a valley open to the end of the trace is judged on the side it has
        climbs = [c for c in (left_max, right_max) if np.isfinite(c)]
        if not climbs:
            continue
        if min(climbs) - v[i] >= prominence:
            keep.append(i)
    return np.array(keep, dtype=int)


def distance_matrix(x):
    """Euclidean distance between every pair of rows."""
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def ward_linkage(x):
    """
    Agglomerative clustering, Ward criterion, in the order the merges happen.

    Each step joins the two clusters whose merger adds least to the total
    within-cluster spread:

        cost(A, B) = |A||B| / (|A| + |B|) * ||centroid_A - centroid_B||^2

    Returns the merge list as (left, right, cost, size), with cluster ids
    numbered the way scipy numbers them - originals first, then one new id per
    merge - so the tree can be walked afterwards.
    """
    n = len(x)
    centroids = {i: x[i].astype(float) for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    members = {i: [i] for i in range(n)}
    active = set(range(n))
    merges = []

    next_id = n
    while len(active) > 1:
        best, best_cost = None, np.inf
        act = sorted(active)
        for ai in range(len(act)):
            for bi in range(ai + 1, len(act)):
                a, b = act[ai], act[bi]
                d = centroids[a] - centroids[b]
                cost = (sizes[a] * sizes[b] / (sizes[a] + sizes[b])) * float(d @ d)
                if cost < best_cost:
                    best, best_cost = (a, b), cost

        a, b = best
        na, nb = sizes[a], sizes[b]
        centroids[next_id] = (centroids[a] * na + centroids[b] * nb) / (na + nb)
        sizes[next_id] = na + nb
        members[next_id] = members[a] + members[b]
        merges.append((a, b, float(np.sqrt(best_cost)), na + nb))

        active.discard(a)
        active.discard(b)
        active.add(next_id)
        next_id += 1

    return merges, members


def leaf_order(merges, n):
    """Left-to-right leaf order of the tree the merges describe."""
    if not merges:
        return list(range(n))
    children = {}
    for k, (a, b, _, _) in enumerate(merges):
        children[n + k] = (a, b)

    root = n + len(merges) - 1
    order, stack = [], [root]
    while stack:
        node = stack.pop()
        if node in children:
            a, b = children[node]
            stack.extend([b, a])          # pushed reversed so a comes out first
        else:
            order.append(node)
    return order


# --- telemetry --------------------------------------------------------------


def wet_share(session):
    """Share of laps run on intermediate or wet tyres."""
    laps = session.laps
    if laps is None or laps.empty or 'Compound' not in laps.columns:
        return 0.0
    comp = laps['Compound'].astype(str).str.upper()
    known = comp.isin({'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'})
    if not known.any():
        return 0.0
    return float(comp[known].isin({'INTERMEDIATE', 'WET'}).mean())


def lap_telemetry(session, lap, hz=10.0):
    """
    Speed, throttle, brake, DRS and X/Y for one lap, merged onto a common
    clock and given a distance axis.

    FastF1's own lap.get_telemetry() would do this, but it merges channels
    through scipy.interpolate, which is blocked on this machine. The raw
    channels are not: session.car_data and session.pos_data are plain frames
    with their own SessionTime, so they can be sliced to the lap and put on a
    shared time base with np.interp.

    Distance is integrated from speed rather than taken from a channel, which
    is what FastF1 does too. The lap length that falls out is a useful check
    on the whole reconstruction - it should land within a few metres of the
    published circuit length.
    """
    num = str(lap['DriverNumber'])
    if num not in session.car_data or num not in session.pos_data:
        return None, 'no raw channels for this driver'

    start = lap.get('LapStartTime')
    if pd.isna(start):
        if pd.isna(lap.get('Time')) or pd.isna(lap.get('LapTime')):
            return None, 'lap has no usable time window'
        start = lap['Time'] - lap['LapTime']
    end = start + lap['LapTime']

    def window(frame):
        t = frame['SessionTime']
        return frame[(t >= start) & (t <= end)]

    car = window(session.car_data[num])
    pos = window(session.pos_data[num])
    if len(car) < 20 or len(pos) < 20:
        return None, f'too few samples in the lap ({len(car)} car, {len(pos)} pos)'

    def seconds(frame):
        return frame['SessionTime'].dt.total_seconds().to_numpy(dtype=float)

    t_car, t_pos = seconds(car), seconds(pos)
    t0 = max(t_car.min(), t_pos.min())
    t1 = min(t_car.max(), t_pos.max())
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 - t0 < 5.0:
        return None, 'car and position channels barely overlap'

    grid = np.arange(t0, t1, 1.0 / hz)
    out = {'Time': grid}

    def put(name, times, values):
        vals = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() >= 10:
            out[name] = np.interp(grid, times[ok], vals[ok])

    put('Speed', t_car, car['Speed'])
    put('Throttle', t_car, car['Throttle'])
    put('X', t_pos, pos['X'])
    put('Y', t_pos, pos['Y'])

    # Brake is a flag and DRS is a status code, so neither may be blended into
    # an average. Both are reduced to a boolean first and thresholded after,
    # which keeps the interpolation from inventing a DRS state of 7.
    if 'Brake' in car.columns:
        braking = car['Brake'].to_numpy()
        braking = np.nan_to_num(braking.astype(float), nan=0.0) > 0.5
        out['Brake'] = (np.interp(grid, t_car, braking.astype(float)) > 0.5).astype(float)
    if 'DRS' in car.columns:
        drs = pd.to_numeric(car['DRS'], errors='coerce').to_numpy(dtype=float)
        drs = np.nan_to_num(drs, nan=0.0) >= DRS_OPEN_FROM
        out['DRS'] = (np.interp(grid, t_car, drs.astype(float)) > 0.5).astype(float)
        out['DRS'] = out['DRS'] * DRS_OPEN_FROM     # keep the same units downstream

    if 'Speed' not in out:
        return None, 'no speed channel'

    # distance by integrating speed; km/h to m/s
    v = out['Speed'] / 3.6
    dt = np.diff(grid, prepend=grid[0])
    out['Distance'] = np.cumsum(v * dt)

    return pd.DataFrame(out), None


def resample_on_distance(tel, step=RESAMPLE_STEP_M):
    """
    Puts the telemetry on an even distance grid.

    The samples arrive at a fixed rate in time, so a car spends far more of
    them in a slow corner than down a straight. Differentiating position on
    that grid measures the sampling as much as the corner. Resampling by
    distance removes the speed from the geometry, which is the whole point of
    the curvature features.
    """
    dist = tel['Distance'].to_numpy(dtype=float)
    keep = np.isfinite(dist)
    dist = dist[keep]
    if len(dist) < 10:
        return None

    order = np.argsort(dist)
    dist = dist[order]
    grid = np.arange(dist.min(), dist.max(), step)

    out = {'Distance': grid}
    for col in ('Speed', 'Throttle', 'Brake', 'DRS', 'X', 'Y'):
        if col not in tel.columns:
            continue
        vals = pd.to_numeric(tel[col], errors='coerce').to_numpy(dtype=float)
        vals = vals[keep][order]
        if col == 'Brake':
            vals = np.nan_to_num(vals, nan=0.0)
        ok = np.isfinite(vals)
        if ok.sum() < 10:
            continue
        out[col] = np.interp(grid, dist[ok], vals[ok])
    return pd.DataFrame(out)


def smooth(values, window_m, step=RESAMPLE_STEP_M):
    n = max(int(round(window_m / step)), 1)
    if n <= 1:
        return values
    kernel = np.ones(n) / n
    padded = np.pad(values, (n // 2, n - 1 - n // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def runs_of(mask, step=RESAMPLE_STEP_M):
    """Contiguous True runs, as a list of lengths in metres."""
    lengths, run = [], 0
    for v in mask:
        if v:
            run += 1
        elif run:
            lengths.append(run * step)
            run = 0
    if run:
        lengths.append(run * step)
    return lengths


# --- features ---------------------------------------------------------------


def speed_features(tel):
    speed = smooth(tel['Speed'].to_numpy(dtype=float), SMOOTH_WINDOW_M)
    top = float(np.nanmax(speed))
    if not np.isfinite(top) or top <= 0:
        return None

    # local minima are corners: invert the trace and look for peaks. The
    # prominence threshold is a share of this lap's own top speed, so it
    # scales with the car rather than being a fixed km/h.
    minima = find_valleys(speed, prominence=MIN_PROMINENCE_FRACTION * top)
    corner_speeds = speed[minima] / top if len(minima) else np.array([])

    out = {
        'n_corners': int(len(minima)),
        'corner_speed_median': float(np.median(corner_speeds)) if len(corner_speeds) else np.nan,
        'corner_speed_p10': float(np.percentile(corner_speeds, 10)) if len(corner_speeds) else np.nan,
        'mean_over_top_speed': float(np.mean(speed) / top),
    }

    total = float(tel['Distance'].iloc[-1] - tel['Distance'].iloc[0])
    if 'Throttle' in tel.columns and total > 0:
        flat = tel['Throttle'].to_numpy(dtype=float) >= FULL_THROTTLE_PCT
        spans = runs_of(flat)
        out['full_throttle_share'] = float(flat.mean())
        out['longest_flat_out'] = float(max(spans) / total) if spans else 0.0

        # how much of the lap is spent in runs long enough to be a straight
        for limit in SUSTAINED_THRESHOLDS_M:
            big = [s for s in spans if s >= limit]
            key = f'sustained_{int(limit)}'
            out[key] = float(sum(big) / total) if big else 0.0
            out[f'n_{key}'] = int(len(big))
    else:
        out['full_throttle_share'] = np.nan
        out['longest_flat_out'] = np.nan
        for limit in SUSTAINED_THRESHOLDS_M:
            out[f'sustained_{int(limit)}'] = np.nan
            out[f'n_sustained_{int(limit)}'] = np.nan

    return out


def brake_features(tel):
    if 'Brake' not in tel.columns:
        return {'n_braking_zones': np.nan, 'braking_share': np.nan}

    braking = tel['Brake'].to_numpy(dtype=float) > 0.5
    spans = runs_of(braking)
    # a genuine braking zone is a stop, not a brush of the pedal
    real = [s for s in spans if s >= 20.0]
    return {
        'n_braking_zones': int(len(real)),
        'braking_share': float(braking.mean()),
    }


def geometry_features(tel):
    """
    Curvature from the X/Y trace, which never touches the speed channel.

    On a grid that is evenly spaced in distance the tangent is a unit vector,
    so the curvature reduces to the magnitude of the second derivative - but
    it is computed the full way here so the number stays right if the grid
    ever stops being even.
    """
    if 'X' not in tel.columns or 'Y' not in tel.columns:
        return {'curvature_median': np.nan, 'curvature_p90': np.nan,
                'direction_changes': np.nan}

    # FastF1 position data is in tenths of a metre
    x = smooth(tel['X'].to_numpy(dtype=float) / 10.0, SMOOTH_WINDOW_M)
    y = smooth(tel['Y'].to_numpy(dtype=float) / 10.0, SMOOTH_WINDOW_M)
    s = tel['Distance'].to_numpy(dtype=float)

    dx, dy = np.gradient(x, s), np.gradient(y, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)

    speed_sq = dx * dx + dy * dy
    denom = np.power(np.maximum(speed_sq, 1e-9), 1.5)
    signed = (dx * ddy - dy * ddx) / denom
    kappa = np.abs(signed)

    # trim the ends: the trace is not closed and the derivative is unreliable
    # where the smoothing window ran off the edge
    edge = max(int(SMOOTH_WINDOW_M / RESAMPLE_STEP_M), 1)
    kappa = kappa[edge:-edge] if len(kappa) > 2 * edge else kappa
    signed = signed[edge:-edge] if len(signed) > 2 * edge else signed

    p90 = float(np.percentile(kappa, 90))

    # A direction change is a corner that turns the other way. Only the parts
    # of the lap that are actually cornering count - on a straight the sign of
    # a near-zero curvature is noise and would count every few metres.
    turning = kappa > 0.25 * p90
    signs = np.sign(signed[turning])
    signs = signs[signs != 0]
    changes = int(np.count_nonzero(np.diff(signs) != 0)) if len(signs) > 1 else 0

    return {
        'curvature_median': float(np.median(kappa)),
        'curvature_p90': p90,
        'direction_changes': changes,
    }


def drs_features(tel):
    if 'DRS' not in tel.columns:
        return {'n_drs_zones': np.nan}
    open_flap = tel['DRS'].to_numpy(dtype=float) >= DRS_OPEN_FROM
    spans = [s for s in runs_of(open_flap) if s >= 50.0]
    return {'n_drs_zones': int(len(spans))}


def measure(session, event_name):
    lap = session.laps.pick_fastest()
    if lap is None or (hasattr(lap, 'empty') and lap.empty):
        return None, 'no fastest lap'

    tel, why = lap_telemetry(session, lap)
    if tel is None:
        return None, why

    grid = resample_on_distance(tel)
    if grid is None or len(grid) < 50:
        return None, 'telemetry too short to resample'

    row = {'Race': event_name,
           'lap_distance': float(grid['Distance'].iloc[-1] - grid['Distance'].iloc[0]),
           'driver': str(lap['Driver']) if 'Driver' in lap else '',
           'lap_time_s': (lap['LapTime'].total_seconds()
                          if pd.notna(lap.get('LapTime')) else np.nan)}

    sf = speed_features(grid)
    if sf is None:
        return None, 'speed trace unusable'
    row.update(sf)
    row.update(brake_features(grid))
    row.update(geometry_features(grid))
    row.update(drs_features(grid))
    return row, None


# --- collection -------------------------------------------------------------


def collect():
    import fastf1

    fastf1.Cache.enable_cache(CACHE_DIR)
    schedule = fastf1.get_event_schedule(SEASON, include_testing=False)

    rows, skipped = [], []
    for _, event in schedule.iterrows():
        name = event['EventName']
        try:
            session = fastf1.get_session(SEASON, name, SESSION_TYPE)
            session.load(telemetry=True, weather=False, messages=False)
        except Exception as exc:
            skipped.append((name, f'load failed ({type(exc).__name__})'))
            print(f'  {name:34s} skipped - load failed')
            continue

        share = wet_share(session)
        if share > WET_COMPOUND_SHARE_LIMIT:
            skipped.append((name, f'wet session ({share:.0%} wet tyres)'))
            print(f'  {name:34s} skipped - wet ({share:.0%})')
            continue

        row, why = measure(session, name)
        if row is None:
            skipped.append((name, why))
            print(f'  {name:34s} skipped - {why}')
            continue

        rows.append(row)
        print(f"  {name:34s} {row['n_corners']:2d} corners  "
              f"flat out {row['full_throttle_share']:.0%}  "
              f"brake {row['braking_share']:.0%}  "
              f"corner speed {row['corner_speed_median']:.2f} of top")

    return pd.DataFrame(rows), skipped


# --- diagnostics, none of which feed the model ------------------------------


def report_correlations(df):
    present = [f for f in ALL_FEATURES if f in df.columns and df[f].notna().any()]
    corr = df[present].corr()

    print('\n===== correlation between the measured features =====')
    print(corr.round(2).to_string())

    flagged = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= CORRELATION_FLAG:
                flagged.append((a, b, float(r)))

    print(f'\nPairs at or above |r| = {CORRELATION_FLAG}:')
    if not flagged:
        print('  none - every measured feature is carrying its own information')
    for a, b, r in sorted(flagged, key=lambda t: -abs(t[2])):
        keep = 'both in FEATURE_SET' if a in FEATURE_SET and b in FEATURE_SET else ''
        print(f'  {a:22s} {b:22s} r = {r:+.2f}  {keep}')
    return corr, flagged


def add_straight_blend(df):
    """
    Builds the mixed straight-line feature and returns it with its recipe.

    The components are z-scored before mixing, because they are shares of
    different things on different scales - one is a sum of runs, the other a
    single run - and weighting raw values would let whichever happens to have
    the wider spread decide the mix. The result is z-scored again so it enters
    the feature set on the same footing as everything else.

    The component means and standard deviations come back with it: a circuit
    measured later has to be put through this same scale, and re-deriving it
    from a different set of circuits would silently move every existing value.
    """
    df = df.copy()
    recipe = []
    blend = np.zeros(len(df), dtype=float)

    for name, weight in STRAIGHT_BLEND.items():
        col = df[name].astype(float)
        mean, sd = float(col.mean()), float(col.std(ddof=0))
        if sd <= 0 or not np.isfinite(sd):
            sd = 1.0
        blend += weight * ((col - mean) / sd).to_numpy()
        recipe.append({'feature': name, 'role': 'straight_blend component',
                       'weight': weight, 'mean': mean, 'std': sd})

    mean, sd = float(blend.mean()), float(blend.std(ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        sd = 1.0
    df['straight_blend'] = (blend - mean) / sd
    recipe.append({'feature': 'straight_blend', 'role': 'blend restandardised',
                   'weight': 1.0, 'mean': mean, 'std': sd})
    return df, pd.DataFrame(recipe)


def standardise(df):
    z = pd.DataFrame({'Race': df['Race']})
    scaling = []
    for f in FEATURE_SET:
        col = df[f].astype(float)
        mean, sd = float(col.mean()), float(col.std(ddof=0))
        if sd <= 0 or not np.isfinite(sd):
            sd = 1.0
        z[f] = (col - mean) / sd
        scaling.append({'feature': f, 'mean': mean, 'std': sd})
    return z, pd.DataFrame(scaling)


def report_neighbours(z, k=3):
    feats = z[FEATURE_SET].to_numpy(dtype=float)
    names = z['Race'].tolist()
    d = distance_matrix(feats)
    np.fill_diagonal(d, np.inf)

    print(f'\n===== nearest {k} circuits in feature space =====')
    print('The check: Monaco should sit with Singapore and Baku, Monza with')
    print('Spa and Baku. Monza next to Monaco means the features are wrong.\n')
    for i, name in enumerate(names):
        near = np.argsort(d[i])[:k]
        txt = ', '.join(f'{names[j]} ({d[i, j]:.2f})' for j in near)
        print(f'  {name:34s} {txt}')
    return d


def report_pca(z):
    x = z[FEATURE_SET].to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2 / max(len(x) - 1, 1)
    share = var / var.sum()

    print('\n===== PCA (diagnostic only) =====')
    print(f'  PC1 explains {share[0]:.0%}, PC2 {share[1]:.0%}, '
          f'together {share[0] + share[1]:.0%}')
    for i in (0, 1):
        loading = ', '.join(f'{f} {vt[i, j]:+.2f}'
                            for j, f in enumerate(FEATURE_SET))
        print(f'  PC{i + 1} loadings: {loading}')

    coords = x @ vt[:2].T
    print('\n  circuit coordinates:')
    for name, (a, b) in zip(z['Race'], coords):
        print(f'    {name:34s} PC1 {a:+6.2f}   PC2 {b:+6.2f}')


def report_clusters(z):
    x = z[FEATURE_SET].to_numpy(dtype=float)
    names = z['Race'].tolist()
    merges, members = ward_linkage(x)

    print('\n===== hierarchical clustering, Ward (diagnostic only) =====')
    print('  leaves left to right, circuits beside each other are alike:')
    for leaf in leaf_order(merges, len(names)):
        print(f'    {names[leaf]}')

    print('\n  the first merges - the pairs the features call most similar:')
    for a, b, cost, size in merges[:6]:
        left = ', '.join(names[i] for i in members[a])
        right = ', '.join(names[i] for i in members[b])
        print(f'    {cost:5.2f}  [{left}] + [{right}]')


# --- main -------------------------------------------------------------------


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f'Season {SEASON}, {SESSION_TYPE} sessions, fastest lap of each.')
    print('Telemetry is far larger than lap data - the first run downloads a')
    print('few hundred MB and takes a long while. Later runs read the cache.')
    print('\nThe check this has to pass: Monaco near Singapore and Baku, Monza')
    print('near Spa and Baku. Monza beside Monaco means the features are wrong')
    print('and the pooling in track_affinity.py must not be built on them.\n')

    df, skipped = collect()
    if df.empty:
        print('\nNothing measured.')
        return

    print(f'\nMeasured {len(df)} circuits, skipped {len(skipped)}.')
    for name, why in skipped:
        print(f'  {name:34s} {why}')

    df, blend_recipe = add_straight_blend(df)
    mix = ' + '.join(f'{w:g}*{n}' for n, w in STRAIGHT_BLEND.items())
    print(f'\nstraight_blend = {mix}  (z-scored components, restandardised)')

    report_correlations(df)

    usable = df.dropna(subset=FEATURE_SET).reset_index(drop=True)
    if len(usable) < len(df):
        lost = set(df['Race']) - set(usable['Race'])
        print(f'\nDropped {len(lost)} circuit(s) missing a FEATURE_SET value: '
              f'{", ".join(sorted(lost))}')

    z, scaling = standardise(usable)
    scaling = pd.concat([scaling.assign(role='feature', weight=1.0),
                         blend_recipe], ignore_index=True)
    report_neighbours(z)
    report_pca(z)
    report_clusters(z)

    out = usable.merge(z, on='Race', suffixes=('', '_z'))
    # 2025 is a single season, so a layout change cannot be detected: Barcelona
    # lost its final chicane in 2023 and Melbourne rebuilt corners in 2022, and
    # the affinity data those feed comes from the older layout.
    out['measured_season'] = SEASON
    out['layout_note'] = 'single season, layout change not detectable'

    feats_path = os.path.join(DATA_DIR, 'track_features.csv')
    scale_path = os.path.join(DATA_DIR, 'track_features_scaling.csv')
    out.to_csv(feats_path, index=False)
    scaling.to_csv(scale_path, index=False)
    print(f'\nSaved: {feats_path}')
    print(f'Saved: {scale_path}')


if __name__ == '__main__':
    main()
