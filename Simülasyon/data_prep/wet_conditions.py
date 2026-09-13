"""
F1 Prediction Simulation - v2.1 - wet_conditions.py
What the lap data can and cannot say about running in the rain.

    python -m Simülasyon.data_prep.wet_conditions

This runs before any wet model is built, and its job is to find out which
parts of that model are allowed to claim they were measured. The answer turns
out to be: some of them.

The measurement problem
-----------------------
There is no water-depth channel in the data. The obvious substitute is to read
the condition off the tyre - inters are out, so the track was inter-wet - and
that is circular the moment it is used to show inters were the right choice.
The whole argument reduces to "teams fitted the tyre they fitted".

So nothing here labels the track from a compound and then scores compounds.
Three separate questions are asked instead, each with its own denominator:

  1. How much slower is running in the wet at all? Measured against the same
     race's own dry median - same cars, same circuit, same afternoon. That is
     a condition penalty, not a tyre ranking.

  2. Where do two categories cross over? Measured only on laps where cars were
     running both at once, so the comparison is between teams that disagreed
     with each other. Some of them were wrong, and that is exactly what
     identifies the crossing. This is the honest question and it has almost no
     data behind it.

  3. Do neutralizations really happen more often in the wet? Counted as events
     per eligible lap rather than per race, so one long safety car is one
     event and a wet race is not penalised for being long.

What comes out
--------------
Question 1 is answerable: 3,167 green inter laps over 13 races. Question 3 is
answerable and contradicts the constant the project has been carrying. Question
2 is not answerable - 32 lap-instants over 11 races - and the crossover in the
simulation is therefore a scenario, labelled as one, anchored where these
measurements reach and interpolated where they do not.

Degradation is the other gap. A drying track makes an inter quicker as it ages,
which is the opposite sign to wear, and nothing in the lap data separates the
two. The age curve for the wet categories is an assumption.
"""

import os

import numpy as np
import pandas as pd

# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LAPS = os.path.join(DATA_DIR, 'laps_2018_2025.csv')
OUT = os.path.join(DATA_DIR, 'wet_profiles.csv')

DRY_COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')
WET_CATEGORIES = ('INTERMEDIATE', 'WET')

# A lap counts as run in wet conditions when most of the field is on a wet
# tyre. This is a compound-derived label and it is used only for the
# neutralization rate - never to score a compound against another, which is
# the circle this module exists to stay out of.
WET_FIELD_SHARE = 0.50

# Both sides of a crossover comparison need enough cars for a median to mean
# anything. Two is already generous; the table reports what each threshold
# leaves, because the honest headline here is how little there is.
MIN_CARS_PER_SIDE = 2

# Enough laps for a race to contribute a pace profile at all
MIN_LAPS_FOR_PROFILE = 30
MIN_DRY_LAPS_FOR_BASELINE = 50


def load():
    """Green, non-pit, timed laps, with a category on each."""
    df = pd.read_csv(LAPS, low_memory=False)
    df['category'] = np.select(
        [df['Compound'].isin(DRY_COMPOUNDS),
         df['Compound'] == 'INTERMEDIATE',
         df['Compound'] == 'WET'],
        ['DRY', 'INTERMEDIATE', 'WET'], default=None)
    df = df[df['category'].notna()].copy()
    df['status'] = df['TrackStatus'].fillna(0).astype(int).astype(str)

    # A safety car lap is not a slow lap, it is a different event, and letting
    # it into any of these distributions would make the wet look slower than
    # it is for a reason that has nothing to do with water.
    df['green'] = df['status'] == '1'
    df['racing'] = (df['green'] & df['LapTime_s'].notna()
                    & df['PitInTime_s'].isna() & df['PitOutTime_s'].isna())
    return df


def condition_penalty(df):
    """
    How much slower a race's wet running was than its own dry running.

    The baseline is the same race's dry median, which controls for the circuit,
    the cars and the era all at once - the things a cross-race comparison would
    have to model and would get wrong. What is left is the condition.

    This is a penalty for the afternoon, not for the tyre: it is measured on
    cars that were on the right tyre for what the track was doing, so it says
    nothing about what the wrong tyre would have cost.
    """
    rows = []
    for (season, race), sub in df[df['racing']].groupby(['Season', 'Race']):
        dry = sub.loc[sub['category'] == 'DRY', 'LapTime_s']
        if len(dry) < MIN_DRY_LAPS_FOR_BASELINE:
            continue
        base = float(dry.median())
        for category in WET_CATEGORIES:
            wet = sub[sub['category'] == category]
            if len(wet) < MIN_LAPS_FOR_PROFILE:
                continue
            rows.append({
                'season': season, 'race': race, 'category': category,
                'laps': len(wet), 'dry_median': base,
                'penalty_pct': float(wet['LapTime_s'].median()) / base - 1.0,
                'sigma': _lap_to_lap(wet),
            })
    return pd.DataFrame(rows)


def _lap_to_lap(sub):
    """
    Lap-to-lap spread, from consecutive differences rather than a standard
    deviation about a mean.

    A wet race has a trend in it - the track is drying or filling - and the
    spread about the race's own mean would measure that trend rather than the
    driver. Successive laps are close enough together that the trend cancels,
    and the median absolute difference over 1.128 recovers a normal sigma
    without a handful of aquaplaning moments setting the number.
    """
    per_driver = sub.groupby('Driver')['LapTime_s'].apply(
        lambda v: v.diff().abs().median() / 1.128)
    return float(per_driver.median()) if len(per_driver) else np.nan


def dry_sigma(df):
    """The same statistic on dry laps, so the wet one has something to be."""
    per = df[df['racing'] & (df['category'] == 'DRY')].groupby(
        ['Season', 'Race', 'Driver'])['LapTime_s'].apply(
        lambda v: v.diff().abs().median() / 1.128)
    return float(per.median())


def crossover(df):
    """
    The only non-circular view of where one category beats another: laps where
    cars were out on both at the same time.

    Teams disagreed with each other on these laps. Some were early, some were
    late, and the ones who were wrong are what makes the comparison informative
    - a lap where everybody agrees contains no information about the choice.

    It is also almost all of the data there is, and there is not much of it.
    """
    racing = df[df['racing']]
    rows = []
    for (season, race, lap), sub in racing.groupby(['Season', 'Race',
                                                    'LapNumber']):
        counts = sub['category'].value_counts()
        if len(counts) < 2:
            continue
        median = sub.groupby('category')['LapTime_s'].median()
        share = sub['category'].isin(WET_CATEGORIES).mean()
        rows.append({
            'season': season, 'race': race, 'lap': lap, 'wet_share': share,
            **{f'n_{c.lower()}': int(counts.get(c, 0))
               for c in ('DRY',) + WET_CATEGORIES},
            **{f't_{c.lower()}': median.get(c, np.nan)
               for c in ('DRY',) + WET_CATEGORIES},
        })
    return pd.DataFrame(rows)


def crossover_summary(mixed, a, b, floor=MIN_CARS_PER_SIDE):
    """The gap between two categories, and how little stands behind it."""
    if mixed.empty:
        return None
    ca, cb = f'n_{a.lower()}', f'n_{b.lower()}'
    both = mixed[(mixed[ca] >= floor) & (mixed[cb] >= floor)].copy()
    if both.empty:
        return {'pair': f'{a}-{b}', 'instants': 0, 'races': 0,
                'median_gap': np.nan}
    both['gap'] = both[f't_{b.lower()}'] - both[f't_{a.lower()}']
    return {
        'pair': f'{a}-{b}',
        'instants': len(both),
        'races': both.groupby(['season', 'race']).ngroups,
        'median_gap': float(both['gap'].median()),
        'dry_end_gap': float(both.loc[both['wet_share'] <= 0.25, 'gap'].median())
        if (both['wet_share'] <= 0.25).any() else np.nan,
        'wet_end_gap': float(both.loc[both['wet_share'] >= 0.75, 'gap'].median())
        if (both['wet_share'] >= 0.75).any() else np.nan,
    }


def neutralization_rate(df):
    """
    Whether the wet really does neutralise more often, per eligible lap.

    Per race is the wrong denominator twice over. A wet race tends to be longer
    in laps under yellow, so counting events per race rewards the wet for the
    very thing being measured; and a single safety car that runs for six laps
    is one event, not six. Counting the laps a new event could start on, and
    the laps one actually did, fixes both.

    The project has been carrying 1.8x as the wet multiplier with a comment
    calling it a roadmap number. It is not a measurement, and this is.
    """
    per_lap = df.groupby(['Season', 'Race', 'LapNumber'], as_index=False).agg(
        wet_share=('category', lambda s: s.isin(WET_CATEGORIES).mean()),
        neutral=('status', lambda s: s.str.contains('4|6').any()))
    per_lap['wet'] = per_lap['wet_share'] > WET_FIELD_SHARE

    # only the lap an event begins on is an event; the rest is the same one
    started = per_lap['neutral'] & ~per_lap.groupby(
        ['Season', 'Race'])['neutral'].shift(1, fill_value=False)
    per_lap['started'] = started

    out = {}
    for label, sub in (('dry', per_lap[~per_lap['wet']]),
                       ('wet', per_lap[per_lap['wet']])):
        out[label] = {'laps': len(sub), 'events': int(sub['started'].sum()),
                      'rate': float(sub['started'].mean()) if len(sub) else
                      np.nan}
    out['ratio'] = (out['wet']['rate'] / out['dry']['rate']
                    if out['dry']['rate'] else np.nan)
    out['wet_lap_share'] = float(per_lap['wet'].mean())
    return out


def stint_lengths(df):
    """
    The longest stint each wet category was actually run for.

    This is not a life. It is the longest anyone happened to need, and on a
    drying track that is set by the weather rather than by the tyre. It is
    recorded so the cap in the simulation has something to be anchored near,
    and labelled so nobody mistakes it for a measured limit.
    """
    out = {}
    for category in WET_CATEGORIES:
        sub = df[(df['category'] == category) & df['TyreLife'].notna()]
        if sub.empty:
            out[category] = {'max_observed': np.nan, 'p90': np.nan, 'stints': 0}
            continue
        per_stint = sub.groupby(['Season', 'Race', 'Driver', 'Stint'])[
            'TyreLife'].max()
        out[category] = {'max_observed': float(per_stint.max()),
                         'p90': float(per_stint.quantile(0.90)),
                         'stints': int(len(per_stint))}
    return out


def age_slope(df):
    """
    Whether the wet categories show a wear trend at all, and why the number
    that comes back cannot be used as one.

    On a drying track an inter gets quicker as it ages, because the track is
    improving faster than the tyre is going off. The regression below sees the
    sum of the two and there is nothing in the lap data to separate them. A
    negative slope here is the track drying, not a tyre that improves with use.
    """
    out = {}
    for category in WET_CATEGORIES:
        sub = df[df['racing'] & (df['category'] == category)]
        rows = []
        for _, s in sub.groupby(['Season', 'Race', 'Driver', 'Stint']):
            if len(s) < 5 or s['TyreLife'].nunique() < 3:
                continue
            slope = np.polyfit(s['TyreLife'].to_numpy(float),
                               s['LapTime_s'].to_numpy(float), 1)[0]
            rows.append(slope)
        out[category] = {'stints': len(rows),
                         'median_slope': float(np.median(rows)) if rows
                         else np.nan}
    return out


def build():
    df = load()
    penalties = condition_penalty(df)
    base_sigma = dry_sigma(df)
    mixed = crossover(df)
    neutral = neutralization_rate(df)
    caps = stint_lengths(df)
    slopes = age_slope(df)

    print(f'\n=== wet conditions, 2018-2025 ===')
    print(f'{len(df):,} laps with a category, '
          f'{int(df["racing"].sum()):,} of them green and out of the pits')
    counts = df[df['racing']]['category'].value_counts()
    print('  ' + '   '.join(f'{k} {v:,}' for k, v in counts.items()))

    print(f'\n--- 1. condition penalty, against each race\'s own dry median ---')
    rows = []
    for category in WET_CATEGORIES:
        sub = penalties[penalties['category'] == category]
        if sub.empty:
            print(f'  {category:<14} nothing measurable')
            continue
        med = float(sub['penalty_pct'].median())
        sig = float(sub['sigma'].median())
        print(f'  {category:<14} {med:+.1%} per lap   '
              f'[{sub["penalty_pct"].quantile(.25):+.1%} to '
              f'{sub["penalty_pct"].quantile(.75):+.1%}]   '
              f'sigma {sig:.3f} s ({sig / base_sigma:.1f}x dry)   '
              f'{int(sub["laps"].sum()):,} laps, {len(sub)} races')
        rows.append({'key': f'penalty_pct_{category}', 'value': med,
                     'n': int(sub['laps'].sum()), 'races': len(sub),
                     'kind': 'measured' if len(sub) >= 8 else 'thin'})
        rows.append({'key': f'sigma_{category}', 'value': sig,
                     'n': int(sub['laps'].sum()), 'races': len(sub),
                     'kind': 'measured' if len(sub) >= 8 else 'thin'})
    print(f'  {"DRY":<14} reference, sigma {base_sigma:.3f} s')
    rows.append({'key': 'sigma_DRY', 'value': base_sigma, 'n': int(
        (df['racing'] & (df['category'] == 'DRY')).sum()),
        'races': df[df['racing']].groupby(['Season', 'Race']).ngroups,
        'kind': 'measured'})

    print(f'\n--- 2. crossover, from laps run on both at once ---')
    for a, b in (('DRY', 'INTERMEDIATE'), ('INTERMEDIATE', 'WET')):
        s = crossover_summary(mixed, a, b)
        if s is None or s['instants'] == 0:
            print(f'  {a}/{b}: nothing')
            continue
        print(f'  {a}/{b}: {s["instants"]} lap-instants over {s["races"]} '
              f'races (>= {MIN_CARS_PER_SIDE} cars a side)')
        print(f'      median {b} minus {a}: {s["median_gap"]:+.2f} s   '
              f'mostly-dry laps {s["dry_end_gap"]:+.2f}   '
              f'mostly-wet laps {s["wet_end_gap"]:+.2f}')
        rows.append({'key': f'crossover_{a}_{b}_instants',
                     'value': s['instants'], 'n': s['instants'],
                     'races': s['races'], 'kind': 'too thin to fit'})
        if np.isfinite(s['dry_end_gap']):
            rows.append({'key': f'mismatch_{b}_on_dry', 'value': s['dry_end_gap'],
                         'n': s['instants'], 'races': s['races'],
                         'kind': 'thin'})
    print('  This is not enough to fit a crossover curve. The simulation\'s')
    print('  crossover is a scenario anchored on 1 and 3, not a measurement.')

    print(f'\n--- 3. neutralization, events per eligible lap ---')
    for label in ('dry', 'wet'):
        n = neutral[label]
        print(f'  {label:<4} {n["events"]:>4} events over {n["laps"]:>6,} laps'
              f'   {n["rate"] * 1000:6.2f} per 1,000')
    print(f'  wet / dry: {neutral["ratio"]:.2f}x   '
          f'({neutral["wet_lap_share"]:.1%} of laps were wet)')
    print(f'  simulate.py has been assuming 1.80x, sourced to a roadmap '
          f'rather than to data.')
    rows.append({'key': 'neutral_wet_ratio', 'value': neutral['ratio'],
                 'n': neutral['wet']['laps'],
                 'races': neutral['wet']['events'], 'kind': 'measured'})
    rows.append({'key': 'neutral_wet_lap_share', 'value':
                 neutral['wet_lap_share'], 'n': neutral['wet']['laps'],
                 'races': 0, 'kind': 'measured'})

    print(f'\n--- 4. how long a wet set was ever run ---')
    for category in WET_CATEGORIES:
        c = caps[category]
        print(f'  {category:<14} longest {c["max_observed"]:.0f} laps, '
              f'p90 {c["p90"]:.0f}, {c["stints"]} stints')
        rows.append({'key': f'longest_stint_{category}',
                     'value': c['max_observed'], 'n': c['stints'], 'races': 0,
                     'kind': 'observed, not a life'})
    print('  The longest anyone needed, set by the weather. Not a tyre life.')

    print(f'\n--- 5. age slope, and why it is not degradation ---')
    for category in WET_CATEGORIES:
        s = slopes[category]
        print(f'  {category:<14} {s["median_slope"]:+.4f} s per lap of age '
              f'over {s["stints"]} stints')
    print('  A drying track and a wearing tyre have opposite signs and the')
    print('  lap data cannot separate them. The simulation assumes a rate.')

    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f'\nSaved: {OUT}')
    return table


if __name__ == '__main__':
    build()
