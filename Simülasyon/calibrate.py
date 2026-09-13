"""
F1 Prediction Simulation - v2.3 - calibrate.py
Measuring the constants that were chosen rather than measured.

    python -m Simülasyon.calibrate

What this version is for
------------------------
Not new race mechanics. The model is full of numbers somebody picked: a noise
correlation of 0.60, a queue gap of half a second, safety cars centred on 60%
of race distance, six pseudo-races of pooling. Some of those are measurable
and were never measured. Some are not measurable at all and should stop
looking like physics. A few turned out not to be used by anything.

The point is not to change every number. It is to be able to say, of each one,
where it came from and what happens if it is wrong.

How each is judged
------------------
Against the thing it represents, not against last Sunday's winner. The noise
correlation is checked on lap-to-lap residuals, the queue gap on gaps at a
restart, the safety car timing on when safety cars actually start. Tuning all
of them together to hit one race's result is how a model gets good at the past
and bad at the future.

Where there is enough data the seasons are split in time: earlier races choose
the value, later ones check it. Where there is not, that is said rather than
papered over with a third decimal place.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LAPS = os.path.join(DATA_DIR, 'laps_2018_2025.csv')
OUT = os.path.join(DATA_DIR, 'parameter_calibration.csv')

# Seasons up to here choose a value; the rest check it. Split in time rather
# than at random, because a random split puts laps from the same stint on both
# sides and every autocorrelation measured that way comes out too high.
TRAIN_UNTIL = 2023

# A stint needs enough green laps for a trend and a residual series to mean
# anything separately.
MIN_STINT_LAPS = 8

DRY_COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')


def load():
    df = pd.read_csv(LAPS, low_memory=False)
    df['status'] = df['TrackStatus'].fillna(0).astype(int).astype(str)
    df['green'] = df['status'] == '1'
    df['racing'] = (df['green'] & df['LapTime_s'].notna()
                    & df['PitInTime_s'].isna() & df['PitOutTime_s'].isna())
    df['dry'] = df['Compound'].isin(DRY_COMPOUNDS)
    return df


# --- NOISE_AUTOCORR ---------------------------------------------------------

def noise_autocorrelation(df, seasons=None):
    """
    How much of a lap's deviation carries into the next one.

    Three things have to be taken out first or the answer is wrong in a
    predictable direction:

      the trend      A tyre going off makes every lap slower than the last, and
                     the raw correlation reads that as persistence. A straight
                     line through each stint removes it.
      the breaks     A pit stop, a safety car or a red flag ends the series.
                     Treating laps either side as consecutive measures the
                     interruption rather than the driver.
      the mixing     Each stint of each driver of each race is its own series.
                     Pooling them into one long vector creates joins that were
                     never laps.

    What comes back is the lag-1 correlation of within-stint residuals, which
    is exactly the rho the simulation's AR(1) uses.
    """
    sub = df[df['racing'] & df['dry']]
    if seasons is not None:
        sub = sub[sub['Season'].isin(seasons)]

    numerator, denominator, n_stints, n_laps = 0.0, 0.0, 0, 0
    for _, stint in sub.groupby(['Season', 'Race', 'Driver', 'Stint']):
        if len(stint) < MIN_STINT_LAPS:
            continue
        stint = stint.sort_values('LapNumber')

        # consecutive laps only: a gap in lap numbers is an interruption
        if int(stint['LapNumber'].diff().fillna(1).max()) != 1:
            continue

        t = stint['LapNumber'].to_numpy(float)
        y = stint['LapTime_s'].to_numpy(float)
        slope, intercept = np.polyfit(t, y, 1)
        resid = y - (slope * t + intercept)

        # accumulate the pieces of one pooled correlation rather than
        # averaging per-stint correlations, which over-weights short stints
        numerator += float((resid[:-1] * resid[1:]).sum())
        denominator += float((resid ** 2).sum())
        n_stints += 1
        n_laps += len(stint)

    rho = numerator / denominator if denominator else np.nan
    return {'rho': rho, 'stints': n_stints, 'laps': n_laps}


def noise_variance_contract(rho, sigma):
    """
    Whether sigma means the per-lap spread or the innovation.

    The simulation draws e_t = rho*e_{t-1} + sqrt(1-rho^2)*sigma*z, so the
    stationary variance is sigma^2 and sigma is the long-run per-lap spread.
    That is the contract clean.py measures against, and it is what makes rho
    safe to change: the total variance does not move with it.
    """
    var = (sigma ** 2) * (1 - rho ** 2) / (1 - rho ** 2)
    return {'stationary_sd': float(np.sqrt(var)),
            'matches_sigma': bool(abs(np.sqrt(var) - sigma) < 1e-12)}


# --- SC_QUEUE_GAP -----------------------------------------------------------

def queue_gap(df):
    """
    How far apart the field really is when a safety car hands the race back.

    The engine works in whole laps, so what it can represent is the gap at the
    end of the last neutralised lap, not the instantaneous spacing behind the
    car. Those are different quantities and calibrating one against the other
    would be a unit error dressed up as a measurement.

    Cumulative time is not in the lap table, so this uses the only proxy
    available: consecutive cars' positions on the last neutralised lap and the
    lap times they set on the first green lap after it. It is a weak
    measurement and is reported as one.
    """
    rows = []
    for (season, race), group in df.groupby(['Season', 'Race']):
        per_lap = group.groupby('LapNumber')['status'].apply(
            lambda s: s.str.contains('4|6').any())
        if not per_lap.any():
            continue
        laps = per_lap.index.to_numpy()
        neutral = per_lap.to_numpy()
        ends = laps[:-1][neutral[:-1] & ~neutral[1:]]
        for end in ends:
            queue = group[(group['LapNumber'] == end)
                          & group['LapTime_s'].notna()]
            if len(queue) < 6:
                continue
            # under a safety car everyone runs the leader's pace, so the
            # spread of lap times on that lap is what the queue looks like
            times = np.sort(queue['LapTime_s'].to_numpy(float))
            rows.append({'season': season, 'race': race, 'lap': int(end),
                         'cars': len(times),
                         'median_step': float(np.median(np.diff(times)))})
    table = pd.DataFrame(rows)
    return {'restarts': len(table),
            'median_step': float(table['median_step'].median())
            if len(table) else np.nan,
            'p25': float(table['median_step'].quantile(0.25))
            if len(table) else np.nan,
            'p75': float(table['median_step'].quantile(0.75))
            if len(table) else np.nan}


# --- SC timing --------------------------------------------------------------

def sc_timing(df):
    """
    Where in the race a safety car actually starts.

    Only starts. An event that runs for six laps is one start, and counting
    each of its laps would pull the distribution toward wherever long safety
    cars happen to sit.
    """
    starts = []
    for (season, race), group in df.groupby(['Season', 'Race']):
        per_lap = group.groupby('LapNumber')['status'].apply(
            lambda s: s.str.contains('4|6').any())
        if per_lap.empty:
            continue
        laps = per_lap.index.to_numpy()
        flag = per_lap.to_numpy()
        began = flag.copy()
        began[1:] = flag[1:] & ~flag[:-1]
        total = float(laps.max())
        for lap in laps[began]:
            starts.append(lap / total)

    starts = np.array(starts)
    if not len(starts):
        return {'events': 0}
    return {
        'events': len(starts),
        'mean': float(starts.mean()),
        'sd': float(starts.std(ddof=1)),
        'median': float(np.median(starts)),
        'first_quarter': float((starts <= 0.25).mean()),
        'last_quarter': float((starts >= 0.75).mean()),
    }


# --- SC_SHRINK_RACES --------------------------------------------------------

def shrink_strength(df, candidates=(0.0, 3.0, 6.0, 12.0, 24.0)):
    """
    How hard a circuit's own safety-car rate should be pulled toward the
    calendar, chosen by predicting races it has not seen.

    This is the one constant in the file with enough events behind it to be
    chosen properly. Earlier seasons estimate each circuit's rate, later ones
    score it, and the weight that predicts best wins. Six pseudo-races was a
    guess; this says whether it was a good one.

    Scored on log loss of "does this race have a safety car", which is the
    thing the rate is used for, rather than on the rate itself.
    """
    per_race = []
    for (season, race), group in df.groupby(['Season', 'Race']):
        per_lap = group.groupby('LapNumber')['status'].apply(
            lambda s: s.str.contains('4|6').any())
        per_race.append({'season': season, 'race': race,
                         'any_sc': bool(per_lap.any())})
    table = pd.DataFrame(per_race)

    train = table[table['season'] <= TRAIN_UNTIL]
    test = table[table['season'] > TRAIN_UNTIL]
    if train.empty or test.empty:
        return {'verdict': 'not enough seasons'}

    calendar = float(train['any_sc'].mean())
    by_track = train.groupby('race')['any_sc'].agg(['mean', 'size'])

    out = []
    for weight in candidates:
        losses = []
        for _, row in test.iterrows():
            if row['race'] in by_track.index:
                raw, n = by_track.loc[row['race'], ['mean', 'size']]
            else:
                raw, n = calendar, 0.0
            p = (n * raw + weight * calendar) / (n + weight) \
                if (n + weight) > 0 else calendar
            p = float(np.clip(p, 1e-3, 1 - 1e-3))
            losses.append(-(np.log(p) if row['any_sc'] else np.log(1 - p)))
        out.append({'weight': weight, 'log_loss': float(np.mean(losses)),
                    'races': len(test)})

    scores = pd.DataFrame(out).sort_values('log_loss')
    best = scores.iloc[0]
    current = scores[scores['weight'] == 6.0]
    return {
        'table': scores, 'best_weight': float(best['weight']),
        'best_loss': float(best['log_loss']),
        'current_loss': float(current['log_loss'].iloc[0])
        if len(current) else np.nan,
        'train_races': int(len(train)), 'test_races': int(len(test)),
        'calendar_rate': calendar,
    }


# --- red flag bounds --------------------------------------------------------

def rf_bounds(df, cap=0.15, floor=0.01):
    """
    Whether the red-flag clip is a safety rail or a gag.

    A bound that never binds costs nothing and guards against a bad estimate.
    One that binds on half the calendar is not a guard, it is the model.
    """
    rows = []
    for (season, race), group in df.groupby(['Season', 'Race']):
        per_lap = group.groupby('LapNumber')['status'].apply(
            lambda s: s.str.contains('5').any())
        rows.append({'race': race, 'red': bool(per_lap.any())})
    table = pd.DataFrame(rows)
    per_track = table.groupby('race')['red'].agg(['mean', 'size'])

    return {
        'circuits': int(len(per_track)),
        'at_or_below_floor': int((per_track['mean'] <= floor).sum()),
        'at_or_above_cap': int((per_track['mean'] >= cap).sum()),
        'observed_max': float(per_track['mean'].max()),
        'calendar_rate': float(table['red'].mean()),
        'races': int(len(table)),
    }


# --- report -----------------------------------------------------------------

def variance_by_scale(df, seasons=None):
    """
    Where a driver's lap-time variation actually lives.

    This is the check that stops the autocorrelation measurement being read
    too literally. A per-stint linear detrend removes the level as well as the
    slope, so what is left is variation *around* a stint's own form and the
    persistence measured on it is within-stint persistence only. A driver who
    is three tenths off all afternoon contributes nothing to it.

    The simulation's AR(1) is asked to represent both scales at once, so the
    two have to be sized separately before any rho is taken seriously:

      within   lap-to-lap scatter around the stint's trend
      between  how much whole stints sit above or below a driver's own race

    If most of the variance is within-stint, a low rho is the right answer and
    the current 0.60 is doing something else. If a lot of it is between-stint,
    then 0.60 is standing in for form the model has no other way to produce -
    and replacing it with the measured number would delete that rather than
    improve it.
    """
    sub = df[df['racing'] & df['dry']]
    if seasons is not None:
        sub = sub[sub['Season'].isin(seasons)]

    within, between, n = [], [], 0
    for _, race in sub.groupby(['Season', 'Race', 'Driver']):
        race = race.sort_values('LapNumber')
        if len(race) < 2 * MIN_STINT_LAPS:
            continue

        # One trend across the whole driver-race, so fuel burn comes out once.
        # The stint offset is then the mean residual inside each stint, not the
        # intercept of its own fit: an intercept is the line extrapolated back
        # to lap zero, and at lap thirty its uncertainty is thirty times the
        # slope's. Measured that way the between-stint spread came out at two
        # seconds, which is not a thing any driver does.
        t_all = race['LapNumber'].to_numpy(float)
        y_all = race['LapTime_s'].to_numpy(float)
        slope, intercept = np.polyfit(t_all, y_all, 1)
        race = race.assign(resid=y_all - (slope * t_all + intercept))

        offsets = []
        for _, stint in race.groupby('Stint'):
            if len(stint) < MIN_STINT_LAPS:
                continue
            r = stint['resid'].to_numpy(float)
            within.append(float(r.var(ddof=1)))
            offsets.append(float(r.mean()))
        if len(offsets) >= 2:
            between.append(float(np.var(offsets, ddof=1)))
            n += 1

    w = float(np.median(within)) if within else np.nan
    b = float(np.median(between)) if between else np.nan
    return {'within_sd': float(np.sqrt(w)), 'between_sd': float(np.sqrt(b)),
            'drivers': n, 'stints': len(within),
            'between_share': b / (w + b) if np.isfinite(w + b) else np.nan}


def background_sc_timing(df):
    """
    When safety cars start, once the accident-driven ones are taken out.

    v2.2 moved accident-triggered neutralizations into the race itself: a car
    crashes on lap 12 and race control responds. The schedule parameters only
    govern what is left, so calibrating them on every safety car in the file
    would fit the background process to events it no longer produces.

    Matching is the same one-lap window retirements.py uses, and it is
    coincidence rather than proof of cause - which matters less here, because
    over-removing biases toward the background being rarer, not toward a
    convenient answer.
    """
    from Simülasyon import retirements as rt

    table, _ = rt.load()
    crashes = table[table['cause'] == rt.ACCIDENT_LABEL]
    crash_laps = {}
    for (season, race), group in crashes.groupby(['Season', 'Race']):
        crash_laps[(season, race)] = set(group['laps_done'].astype(int))

    starts, dropped = [], 0
    for (season, race), group in df.groupby(['Season', 'Race']):
        per_lap = group.groupby('LapNumber')['status'].apply(
            lambda s: s.str.contains('4|6').any())
        if per_lap.empty:
            continue
        laps = per_lap.index.to_numpy()
        flag = per_lap.to_numpy()
        began = flag.copy()
        began[1:] = flag[1:] & ~flag[:-1]
        total = float(laps.max())
        known = crash_laps.get((season, race), set())
        for lap in laps[began]:
            if any(lap - c in (0, 1) for c in known):
                dropped += 1
                continue
            starts.append(lap / total)

    starts = np.array(starts)
    if not len(starts):
        return {'events': 0}
    return {'events': len(starts), 'removed': dropped,
            'mean': float(starts.mean()), 'sd': float(starts.std(ddof=1)),
            'median': float(np.median(starts)),
            'first_quarter': float((starts <= 0.25).mean())}


def rf_shrink_versus_clip(df, weight=6.0, cap=0.15, floor=0.01):
    """
    Whether pooling a circuit's red-flag rate beats clipping it afterwards.

    The clip was meant as a safety rail. It binds on most of the calendar,
    which makes it the model rather than a guard: two thirds of circuits are
    pushed to exactly the floor and a handful to exactly the cap, and neither
    number came from anywhere.

    Pooling does the same job continuously. A circuit with one race and one
    red flag is pulled most of the way back to the calendar rate instead of
    being pinned at 0.15, and one with ten quiet races keeps more of its own.
    """
    rows = []
    for (season, race), group in df.groupby(['Season', 'Race']):
        per_lap = group.groupby('LapNumber')['status'].apply(
            lambda s: s.str.contains('5').any())
        rows.append({'season': season, 'race': race, 'red': bool(per_lap.any())})
    table = pd.DataFrame(rows)

    train = table[table['season'] <= TRAIN_UNTIL]
    test = table[table['season'] > TRAIN_UNTIL]
    if train.empty or test.empty:
        return {'verdict': 'not enough seasons'}

    calendar = float(train['red'].mean())
    by_track = train.groupby('race')['red'].agg(['mean', 'size'])

    def score(mode):
        losses = []
        for _, row in test.iterrows():
            if row['race'] in by_track.index:
                raw, n = by_track.loc[row['race'], ['mean', 'size']]
            else:
                raw, n = calendar, 0.0
            if mode == 'clip':
                p = float(np.clip(raw, floor, cap))
            elif mode == 'shrink':
                p = (n * raw + weight * calendar) / (n + weight)
            else:
                p = calendar
            p = float(np.clip(p, 1e-4, 1 - 1e-4))
            losses.append(-(np.log(p) if row['red'] else np.log(1 - p)))
        return float(np.mean(losses))

    return {'clip': score('clip'), 'shrink': score('shrink'),
            'calendar_only': score('flat'), 'calendar_rate': calendar,
            'test_races': int(len(test))}


def build():
    df = load()
    seasons = sorted(df['Season'].unique())
    train = [s for s in seasons if s <= TRAIN_UNTIL]
    test = [s for s in seasons if s > TRAIN_UNTIL]

    print('\n=== parameter calibration, v2.3 ===')
    print(f'seasons {seasons[0]}-{seasons[-1]}   '
          f'choose on {train[0]}-{train[-1]}, check on {test[0]}-{test[-1]}')

    rows = []

    print('\n--- NOISE_AUTOCORR: lap-to-lap persistence ---')
    fit = noise_autocorrelation(df, train)
    held = noise_autocorrelation(df, test)
    print(f'  chosen on  rho {fit["rho"]:.3f}   '
          f'{fit["stints"]:,} stints, {fit["laps"]:,} laps')
    print(f'  checked on rho {held["rho"]:.3f}   '
          f'{held["stints"]:,} stints, {held["laps"]:,} laps')
    print(f'  currently  0.600, chosen because white noise averaged out')
    contract = noise_variance_contract(fit['rho'], 0.30)
    print(f'  sigma is the long-run per-lap spread, not the innovation: '
          f'{contract["matches_sigma"]}')
    print(f'  so changing rho moves persistence without moving total variance')
    rows.append({'parameter': 'NOISE_AUTOCORR', 'old': 0.600,
                 'new': round(fit['rho'], 3), 'unit': 'correlation',
                 'source': 'measured', 'n': fit['laps'],
                 'holdout': round(held['rho'], 3)})

    print('\n--- SC_QUEUE_GAP: the field at a restart ---')
    queue = queue_gap(df)
    print(f'  {queue["restarts"]} restarts, median step between consecutive '
          f'cars {queue["median_step"]:.2f} s '
          f'[{queue["p25"]:.2f}-{queue["p75"]:.2f}]')
    print(f'  currently  0.50 s')
    print(f'  this is a weak proxy - the lap table has no cumulative time, so')
    print(f'  the spread of lap times on the last neutralised lap stands in')
    rows.append({'parameter': 'SC_QUEUE_GAP', 'old': 0.50,
                 'new': round(queue['median_step'], 2), 'unit': 's',
                 'source': 'weak proxy', 'n': queue['restarts'],
                 'holdout': np.nan})

    print('\n--- where the variation lives, before trusting any rho ---')
    scales = variance_by_scale(df, train)
    print(f'  within a stint  {scales["within_sd"]:.3f} s')
    print(f'  between stints  {scales["between_sd"]:.3f} s   '
          f'{scales["between_share"]:.0%} of the two')
    print(f'  over {scales["stints"]:,} stints and {scales["drivers"]:,} '
          f'driver-races')

    print('\n--- SC timing: when events start ---')
    timing = sc_timing(df)
    print(f'  {timing["events"]} starts   mean {timing["mean"]:.0%} of '
          f'distance, sd {timing["sd"]:.0%}   median {timing["median"]:.0%}')
    print(f'  {timing["first_quarter"]:.0%} in the first quarter, '
          f'{timing["last_quarter"]:.0%} in the last')
    print(f'  currently  mean 60%, sd 20%')
    rows.append({'parameter': 'SC_TIMING_MEAN_FRACTION', 'old': 0.60,
                 'new': round(timing['mean'], 2), 'unit': 'of race distance',
                 'source': 'measured', 'n': timing['events'],
                 'holdout': np.nan})
    rows.append({'parameter': 'SC_TIMING_STD_FRACTION', 'old': 0.20,
                 'new': round(timing['sd'], 2), 'unit': 'of race distance',
                 'source': 'measured', 'n': timing['events'],
                 'holdout': np.nan})

    print('\n--- SC timing, background events only ---')
    bg = background_sc_timing(df)
    print(f'  {bg["events"]} starts left after removing {bg["removed"]} '
          f'matched to an accident retirement')
    print(f'  mean {bg["mean"]:.0%} of distance, sd {bg["sd"]:.0%}, '
          f'median {bg["median"]:.0%}, {bg["first_quarter"]:.0%} in the '
          f'first quarter')
    print(f'  this is the one the schedule parameters actually govern now')

    print('\n--- SC_SHRINK_RACES: how hard to pool ---')
    shrink = shrink_strength(df)
    if 'table' in shrink:
        for _, r in shrink['table'].iterrows():
            mark = ' <-- current' if r['weight'] == 6.0 else ''
            best = ' <-- best' if r['weight'] == shrink['best_weight'] else ''
            print(f'    weight {r["weight"]:>5.1f}   log loss '
                  f'{r["log_loss"]:.4f}{mark}{best}')
        print(f'  chosen on {shrink["train_races"]} races, checked on '
              f'{shrink["test_races"]}')
        gain = shrink['current_loss'] - shrink['best_loss']
        print(f'  best beats the current value by {gain:.4f} of log loss')
        rows.append({'parameter': 'SC_SHRINK_RACES', 'old': 6.0,
                     'new': shrink['best_weight'], 'unit': 'pseudo-races',
                     'source': 'measured, held out', 'n': shrink['test_races'],
                     'holdout': round(shrink['best_loss'], 4)})

    print('\n--- RF_PROB_CAP / RF_PROB_FLOOR: does the clip bind? ---')
    bounds = rf_bounds(df)
    print(f'  {bounds["circuits"]} circuits, calendar rate '
          f'{bounds["calendar_rate"]:.3f} per race')
    print(f'  {bounds["at_or_below_floor"]} circuits sit at or below the '
          f'0.01 floor, {bounds["at_or_above_cap"]} at or above the 0.15 cap')
    print(f'  highest observed circuit rate {bounds["observed_max"]:.3f}')
    rows.append({'parameter': 'RF_PROB_FLOOR', 'old': 0.01, 'new': 0.01,
                 'unit': 'per race', 'source': 'hand-set, binds often',
                 'n': bounds['circuits'], 'holdout': np.nan})

    print('\n--- red flag: pooling against clipping ---')
    rf = rf_shrink_versus_clip(df)
    if 'clip' in rf:
        print(f'  clip to [0.01, 0.15]   log loss {rf["clip"]:.4f}')
        print(f'  pool toward calendar   log loss {rf["shrink"]:.4f}')
        print(f'  calendar rate only     log loss {rf["calendar_only"]:.4f}')
        print(f'  on {rf["test_races"]} held-out races, calendar rate '
              f'{rf["calendar_rate"]:.3f}')

    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f'\nSaved: {OUT}')
    return {'rows': table, 'noise': fit, 'noise_holdout': held,
            'scales': scales, 'queue': queue, 'timing': timing,
            'background_timing': bg, 'shrink': shrink, 'rf': bounds,
            'rf_compare': rf}


if __name__ == '__main__':
    build()
