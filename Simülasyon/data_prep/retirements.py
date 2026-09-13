"""
F1 Prediction Simulation - v2.2 - retirements.py
Why cars stop, how often, when in the race, and whether a safety car follows.

    python -m Simülasyon.data_prep.retirements

What was there before
---------------------
One flat number. Every car carried a 6% chance of not seeing the flag, spread
evenly over every lap of every race, and the reason was never asked. A power
unit letting go on lap 3 and a first-corner collision were the same event, and
neither of them brought out a safety car - the safety car schedule was drawn
separately, before the race started, from a rate that had no idea any of this
had happened.

What this module measures
-------------------------
Four things, each with its own denominator:

  1. Which retirements were accidents and which were failures, from the status
     text the results carry. A third category is kept for the records that do
     not say, because "Retired" on its own is not a gearbox.

  2. Exposure. A driver who crashed twice in five races did not crash on 40% of
     laps; they crashed twice in however many laps they were actually out
     there. A car that stopped on lap 6 was not exposed on lap 7, and counting
     it as though it were understates every rate in the file.

  3. Where in the race failures happen. The roadmap proposes a bell: risk
     rising to the middle of the race and falling away. That is a candidate,
     not a fact, and it is tested here against a flat rate - on hazard, which
     divides by the cars still running, rather than on a histogram of events,
     which does not.

  4. What follows an accident. The project has been carrying "about 90% of
     accident retirements bring out a safety car" as an assumption. This
     matches accident retirements against the neutralizations that actually
     started on the same lap or the next one, deduplicated so a single
     three-car pile-up counts once.

What it deliberately does not do
--------------------------------
It does not spread the unexplained retirements over the two known causes. A
fifth of the retirements in this data say only "Retired", and inventing a
split for them would turn a gap in the evidence into a number that looks
measured. The share is reported instead, and the simulation's two causes are
scaled to the retirements whose cause is known.

It also does not call the accident term a measure of driver error. A driver
collected by somebody else's mistake takes the same classification, so this is
observed accident-retirement risk and nothing stronger.
"""

import os
import re

import numpy as np
import pandas as pd

# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RESULTS = os.path.join(DATA_DIR, 'results_2018_2025.csv')
LAPS = os.path.join(DATA_DIR, 'laps_2018_2025.csv')
OUT_DRIVER = os.path.join(DATA_DIR, 'dnf_driver_risk.csv')
OUT_TEAM = os.path.join(DATA_DIR, 'dnf_team_risk.csv')
OUT_PROFILE = os.path.join(DATA_DIR, 'dnf_profile.csv')

# --- classification ---------------------------------------------------------
# Status text into causes. The lists are explicit rather than keyword-matched
# on a couple of stems, because "Damage" and "Collision damage" want different
# treatment from "Water damage" and a regex that catches one catches all three.
ACCIDENT = {
    'Collision', 'Accident', 'Collision damage', 'Spun off', 'Damage',
    'Debris', 'Front wing', 'Rear wing', 'Undertray',
}
MECHANICAL = {
    'Engine', 'Brakes', 'Power Unit', 'Gearbox', 'Suspension', 'Hydraulics',
    'Power loss', 'Wheel', 'Puncture', 'Fuel pressure', 'Overheating',
    'Exhaust', 'Water pressure', 'Oil leak', 'Electronics', 'Turbo',
    'Mechanical', 'Electrical', 'Transmission', 'Water leak', 'Tyre',
    'Steering', 'Radiator', 'Out of fuel', 'Battery', 'Wheel nut',
    'Driveshaft', 'Fuel leak', 'Water pump', 'Cooling system', 'Fuel pump',
    'Vibrations', 'Differential', 'Engine fire', 'Clutch', 'Throttle',
    'Oil pressure', 'Collision damage ',
}
# Reached the flag, however far back
FINISHED = re.compile(r'^(Finished|Lapped|\+\d+ Laps?)$')

# Neither finished nor retired from a running car. These are not DNFs and
# padding them into one would put a stewards' decision into a reliability
# model.
NOT_A_RETIREMENT = {'Disqualified', 'Did not start', 'Withdrew',
                    'Did not qualify', 'Did not prequalify', 'Illness',
                    'Injury', 'Injured'}

FINISHED_LABEL, ACCIDENT_LABEL = 'finished', 'accident'
MECHANICAL_LABEL, UNKNOWN_LABEL = 'mechanical', 'unknown_other'
EXCLUDED_LABEL = 'excluded'

# Where the wing damage entries go is a judgement call. A front wing failure
# is usually the consequence of contact rather than a component giving up on
# its own, so they sit with the accidents; there are four of them in eight
# seasons and nothing turns on it.

# How much a thin sample is pulled toward the field. Not fitted - there is no
# out-of-sample exercise behind it - so it carries a hand-set label wherever
# it surfaces. It is expressed in car-races of pseudo-data: a driver with this
# many races of their own keeps about half of their measured rate.
SHRINK_RACES = 30.0

# Bands the race is split into when asking where failures fall. Wide, because
# there are 250-odd classified mechanical retirements in the whole file and a
# twenty-bin histogram of that is a picture of nothing.
PROFILE_BANDS = 5

# An accident and the neutralization it caused are not simultaneous: the car
# has to stop, the marshals have to see it, and race control has to decide.
# One lap of slack either way, and not more, or every accident in a race with
# a safety car anywhere in it gets matched.
MATCH_WINDOW = 1


def classify(status):
    """One status string into a cause, or into the bin marked 'not counted'."""
    text = str(status).strip()
    if FINISHED.match(text):
        return FINISHED_LABEL
    if text in NOT_A_RETIREMENT:
        return EXCLUDED_LABEL
    if text in ACCIDENT:
        return ACCIDENT_LABEL
    if text in MECHANICAL:
        return MECHANICAL_LABEL
    return UNKNOWN_LABEL


def load():
    results = pd.read_csv(RESULTS, low_memory=False)
    results['cause'] = results['Status'].map(classify)

    laps = pd.read_csv(LAPS, low_memory=False)
    laps['status'] = laps['TrackStatus'].fillna(0).astype(int).astype(str)

    # How far each car actually got, and how long the race was. A retirement
    # on lap 6 was exposed for six laps and not for the other fifty.
    ran = laps.groupby(['Season', 'Race', 'Driver'], as_index=False).agg(
        laps_done=('LapNumber', 'max'))
    distance = laps.groupby(['Season', 'Race'], as_index=False).agg(
        race_laps=('LapNumber', 'max'))

    merged = results.merge(ran, on=['Season', 'Race', 'Driver'], how='left')
    merged = merged.merge(distance, on=['Season', 'Race'], how='left')
    merged = merged[merged['race_laps'].notna()].copy()

    # A car with no laps at all never started, whatever the status says
    merged['laps_done'] = merged['laps_done'].fillna(0)
    merged.loc[merged['laps_done'] <= 0, 'cause'] = EXCLUDED_LABEL

    # A finisher was exposed for the whole race; a retirement up to the lap it
    # stopped on. Both are the laps on which the event could have been seen.
    merged['exposure'] = np.where(merged['cause'] == FINISHED_LABEL,
                                  merged['race_laps'], merged['laps_done'])
    merged['progress'] = np.clip(merged['laps_done'] / merged['race_laps'],
                                 0.0, 1.0)
    return merged, laps


def coverage(table):
    """How much of the data the two causes actually account for."""
    retired = table[~table['cause'].isin([FINISHED_LABEL, EXCLUDED_LABEL])]
    counts = retired['cause'].value_counts()
    total = int(len(retired))
    known = int(counts.get(ACCIDENT_LABEL, 0) + counts.get(MECHANICAL_LABEL, 0))
    return {
        'retirements': total,
        'accident': int(counts.get(ACCIDENT_LABEL, 0)),
        'mechanical': int(counts.get(MECHANICAL_LABEL, 0)),
        'unknown': int(counts.get(UNKNOWN_LABEL, 0)),
        'known_share': known / total if total else np.nan,
        'excluded': int((table['cause'] == EXCLUDED_LABEL).sum()),
        'starts': int((table['cause'] != EXCLUDED_LABEL).sum()),
    }


def _rate(table, key, cause):
    """
    Events per car-lap at risk, per group, pulled toward the pooled figure.

    The pooling is what keeps a driver who has never crashed from being given
    a risk of exactly zero. Nobody's real risk is zero, and a model that says
    so will hand that driver a championship.
    """
    live = table[table['cause'] != EXCLUDED_LABEL]
    grouped = live.groupby(key).agg(
        events=('cause', lambda s: int((s == cause).sum())),
        exposure=('exposure', 'sum'),
        races=('cause', 'size')).reset_index()

    pooled = live['exposure'].sum()
    pooled_rate = (live['cause'] == cause).sum() / pooled if pooled else 0.0

    # weight in car-races, converted to laps through the group's own average
    laps_per_race = live['exposure'].sum() / max(len(live), 1)
    prior = SHRINK_RACES * laps_per_race
    grouped['raw_rate'] = grouped['events'] / grouped['exposure'].clip(lower=1)
    grouped['rate'] = ((grouped['events'] + pooled_rate * prior)
                       / (grouped['exposure'] + prior))
    grouped['pooled_rate'] = pooled_rate
    grouped['shrinkage'] = prior / (grouped['exposure'] + prior)
    return grouped.sort_values('rate', ascending=False), pooled_rate


def timing_profile(table, bands=PROFILE_BANDS):
    """
    Where in the race mechanical failures happen, as a hazard.

    The distinction that matters: a histogram of events peaks early simply
    because every car is still running early, and it falls away at the end
    because there are fewer cars left to fail. Dividing by the cars still out
    there is what turns counts into risk, and it is the difference between
    "most failures happen in the first half" - true and uninformative - and
    "a car is more likely to fail per lap in the first half", which is a
    claim about the machinery.
    """
    live = table[table['cause'] != EXCLUDED_LABEL]
    edges = np.linspace(0.0, 1.0, bands + 1)

    rows = []
    for i in range(bands):
        lo, hi = edges[i], edges[i + 1]
        # laps each car ran inside this band, which is the exposure
        band_laps = (np.clip(live['progress'], lo, hi) - lo) * live['race_laps']
        failures = ((live['cause'] == MECHANICAL_LABEL)
                    & (live['progress'] > lo) & (live['progress'] <= hi)).sum()
        crashes = ((live['cause'] == ACCIDENT_LABEL)
                   & (live['progress'] > lo) & (live['progress'] <= hi)).sum()
        exposure = float(band_laps.sum())
        rows.append({
            'band': i, 'from': lo, 'to': hi, 'exposure_laps': exposure,
            'mechanical': int(failures), 'accident': int(crashes),
            'mech_hazard': failures / exposure if exposure else np.nan,
            'acc_hazard': crashes / exposure if exposure else np.nan,
        })
    return pd.DataFrame(rows)


def bell_versus_flat(profile):
    """
    Whether the hazard really is bell-shaped, or whether flat fits as well.

    A five-parameter shape will always describe five bands better than one
    parameter does, so the comparison is on a criterion that charges for the
    parameters. If the flat model wins, the honest thing is to keep it and
    report that the bell was not supported - which is what the roadmap asks
    for and the opposite of what a bell-shaped prior wants to hear.
    """
    hazard = profile['mech_hazard'].to_numpy(float)
    exposure = profile['exposure_laps'].to_numpy(float)
    events = profile['mechanical'].to_numpy(float)
    ok = np.isfinite(hazard) & (exposure > 0)
    if ok.sum() < 3:
        return {'verdict': 'not enough data', 'flat_aic': np.nan,
                'bell_aic': np.nan}

    centre = (profile['from'] + profile['to']).to_numpy(float)[ok] / 2
    events, exposure = events[ok], exposure[ok]

    def poisson_ll(mu):
        mu = np.clip(mu, 1e-12, None)
        return float((events * np.log(mu) - mu).sum())

    flat = events.sum() / exposure.sum()
    flat_ll = poisson_ll(flat * exposure)

    # the best bell over a coarse grid: no optimiser is available here and a
    # grid is honest about how little is being fitted
    best_ll, best = -np.inf, None
    for peak in np.linspace(0.15, 0.85, 15):
        for width in np.linspace(0.12, 0.60, 13):
            shape = np.exp(-0.5 * ((centre - peak) / width) ** 2)
            scale = events.sum() / max((shape * exposure).sum(), 1e-12)
            ll = poisson_ll(scale * shape * exposure)
            if ll > best_ll:
                best_ll, best = ll, (peak, width, scale)

    flat_aic = 2 * 1 - 2 * flat_ll
    bell_aic = 2 * 3 - 2 * best_ll
    return {
        'flat_rate': flat, 'flat_aic': flat_aic, 'bell_aic': bell_aic,
        'peak': best[0], 'width': best[1],
        'verdict': 'bell' if bell_aic < flat_aic - 2 else 'flat',
        'margin': flat_aic - bell_aic,
    }


def accident_neutralizations(table, laps):
    """
    What race control did after an accident retirement.

    Deduplicated at the event level. Three cars out of one pile-up is one
    safety car, and counting it three times would put the "does an accident
    bring out a safety car" rate close to one by construction.

    Time proximity is not proof of cause. A safety car on the lap a car
    crashed is very probably for that crash, and this cannot show it; what it
    can show is how often the two coincide, which is the number the simulation
    needs and all it is claimed to be.
    """
    laps = laps.copy()
    laps['neutral'] = laps['status'].str.contains('4|6')
    laps['red'] = laps['status'].str.contains('5')

    per_lap = laps.groupby(['Season', 'Race', 'LapNumber'], as_index=False).agg(
        neutral=('neutral', 'any'), red=('red', 'any'))
    per_lap = per_lap.sort_values(['Season', 'Race', 'LapNumber'])
    per_lap['started'] = per_lap['neutral'] & ~per_lap.groupby(
        ['Season', 'Race'])['neutral'].shift(1, fill_value=False)

    # what kind of neutralization it was, so the simulation can draw the same
    # mix rather than treating every accident as a full safety car
    laps['vsc'] = laps['status'].str.contains('6')
    laps['sc'] = laps['status'].str.contains('4')
    kinds = laps.groupby(['Season', 'Race', 'LapNumber'], as_index=False).agg(
        vsc=('vsc', 'any'), sc=('sc', 'any'), red=('red', 'any'))
    per_lap = per_lap.merge(kinds, on=['Season', 'Race', 'LapNumber'],
                            how='left', suffixes=('', '_k'))

    crashes = table[table['cause'] == ACCIDENT_LABEL]
    events = 0
    outcome = {'none': 0, 'VSC': 0, 'SC': 0, 'RF': 0}
    for (season, race), group in crashes.groupby(['Season', 'Race']):
        race_laps = per_lap[(per_lap['Season'] == season)
                            & (per_lap['Race'] == race)]
        if race_laps.empty:
            continue
        by_lap = race_laps.set_index('LapNumber')
        starts = set(race_laps.loc[race_laps['started'], 'LapNumber'])
        reds = set(race_laps.loc[race_laps['red_k'].fillna(False), 'LapNumber'])

        # one pile-up, one event: crashes on the same lap are one incident
        for lap in sorted(set(group['laps_done'].astype(int))):
            events += 1
            window = [w for w in range(lap, lap + MATCH_WINDOW + 1)]
            if any(w in reds for w in window):
                outcome['RF'] += 1
            elif any(w in starts for w in window):
                hit = next(w for w in window if w in starts)
                row = by_lap.loc[hit]
                outcome['SC' if bool(np.atleast_1d(row['sc'])[0]) else 'VSC'] += 1
            else:
                outcome['none'] += 1

    matched = events - outcome['none']
    return {'incidents': events, 'with_neutralization': matched,
            'share': matched / events if events else np.nan,
            'outcome': {k: v / events if events else np.nan
                        for k, v in outcome.items()}}


def background_share(table, laps):
    """
    How much of the safety-car rate is left once accident-linked ones are
    taken out.

    This is the number that stops v2.2 inflating the total. The circuit rates
    the simulation already uses were measured over races that contained these
    accidents, so switching on accident-triggered safety cars while leaving
    the old rate alone counts them twice. The background share is what the old
    process should be scaled to.
    """
    laps = laps.copy()
    laps['neutral'] = laps['status'].str.contains('4|6')
    per_lap = laps.groupby(['Season', 'Race', 'LapNumber'], as_index=False).agg(
        neutral=('neutral', 'any'))
    per_lap = per_lap.sort_values(['Season', 'Race', 'LapNumber'])
    per_lap['started'] = per_lap['neutral'] & ~per_lap.groupby(
        ['Season', 'Race'])['neutral'].shift(1, fill_value=False)

    total = int(per_lap['started'].sum())
    crashes = table[table['cause'] == ACCIDENT_LABEL]

    explained = 0
    for (season, race), group in crashes.groupby(['Season', 'Race']):
        race_laps = per_lap[(per_lap['Season'] == season)
                            & (per_lap['Race'] == race)]
        starts = sorted(race_laps.loc[race_laps['started'], 'LapNumber'])
        crash_laps = sorted(set(group['laps_done'].astype(int)))
        used = set()
        for start in starts:
            for lap in crash_laps:
                if start - lap in range(0, MATCH_WINDOW + 1) and start not in used:
                    used.add(start)
                    break
        explained += len(used)

    return {'starts': total, 'accident_linked': explained,
            'background': total - explained,
            'background_share': (total - explained) / total if total else np.nan}


def build():
    table, laps = load()
    cov = coverage(table)

    print('\n=== retirements, 2018-2025 ===')
    print(f'{cov["starts"]:,} race starts, {cov["retirements"]} retirements, '
          f'{cov["excluded"]} excluded (disqualified, did not start, withdrew)')
    print(f'  accident      {cov["accident"]:>4}'
          f'   {cov["accident"] / cov["retirements"]:.1%} of retirements')
    print(f'  mechanical    {cov["mechanical"]:>4}'
          f'   {cov["mechanical"] / cov["retirements"]:.1%}')
    print(f'  unexplained   {cov["unknown"]:>4}'
          f'   {cov["unknown"] / cov["retirements"]:.1%}   '
          f'status says only "Retired" or similar')
    print(f'  cause known for {cov["known_share"]:.1%} of retirements. The '
          f'rest are a gap in the data,')
    print(f'  not a third cause, and are not redistributed over the two.')

    drivers, driver_pooled = _rate(table, 'Driver', ACCIDENT_LABEL)
    teams, team_pooled = _rate(table, 'Team', MECHANICAL_LABEL)

    print(f'\n--- accident risk per driver, per lap at risk ---')
    print(f'  pooled {driver_pooled * 1000:.3f} per 1,000 laps; '
          f'{SHRINK_RACES:.0f} car-races of pooling (hand-set)')
    for _, r in drivers.head(5).iterrows():
        print(f'    {r["Driver"]:<5} {r["rate"] * 1000:5.3f}  raw '
              f'{r["raw_rate"] * 1000:5.3f}  {int(r["races"]):>3} races  '
              f'{int(r["events"])} crashes  pulled {r["shrinkage"]:.0%}')
    print(f'    ... {len(drivers)} drivers, '
          f'{int((drivers["events"] == 0).sum())} with none observed - which '
          f'is not a risk of zero')

    print(f'\n--- mechanical risk per team, per lap at risk ---')
    print(f'  pooled {team_pooled * 1000:.3f} per 1,000 laps')
    for _, r in teams.head(5).iterrows():
        print(f'    {r["Team"]:<22} {r["rate"] * 1000:5.3f}  raw '
              f'{r["raw_rate"] * 1000:5.3f}  {int(r["races"]):>4} races  '
              f'{int(r["events"])} failures')

    profile = timing_profile(table)
    verdict = bell_versus_flat(profile)
    print(f'\n--- when failures happen, as hazard not as a histogram ---')
    for _, r in profile.iterrows():
        print(f'  {r["from"]:.0%}-{r["to"]:.0%}  {int(r["mechanical"]):>3} '
              f'failures over {r["exposure_laps"]:>8,.0f} car-laps'
              f'   {r["mech_hazard"] * 1000:5.3f} per 1,000')
    print(f'  flat AIC {verdict["flat_aic"]:.1f}   '
          f'bell AIC {verdict["bell_aic"]:.1f}   '
          f'-> {verdict["verdict"]}')
    if verdict['verdict'] == 'flat':
        print('  The bell the roadmap proposed is not supported. A flat '
              'per-lap hazard')
        print('  describes these bands as well and costs two fewer '
              'parameters, so the')
        print('  simulation keeps the flat one and this is reported rather '
              'than hidden.')
    else:
        print(f'  peak at {verdict["peak"]:.0%} of race distance, '
              f'width {verdict["width"]:.2f}')

    linked = accident_neutralizations(table, laps)
    print(f'\n--- what follows an accident ---')
    print(f'  {linked["incidents"]} accident incidents (same lap, same race '
          f'= one incident)')
    print(f'  {linked["with_neutralization"]} were followed by a '
          f'neutralization within {MATCH_WINDOW} lap'
          f'   {linked["share"]:.1%}')
    print(f'  simulate.py has been assuming about 90%.')
    print('  what followed: ' + '  '.join(
        f'{k} {v:.1%}' for k, v in linked['outcome'].items()))

    back = background_share(table, laps)
    print(f'\n--- and how much of the safety-car rate that explains ---')
    print(f'  {back["starts"]} neutralization starts, '
          f'{back["accident_linked"]} matched to an accident retirement, '
          f'{back["background"]} not')
    print(f'  background share {back["background_share"]:.1%} - what the '
          f'existing circuit rate has to')
    print(f'  be scaled to once accidents generate their own, or the total '
          f'is counted twice.')

    drivers.to_csv(OUT_DRIVER, index=False)
    teams.to_csv(OUT_TEAM, index=False)
    profile.to_csv(OUT_PROFILE, index=False)
    print(f'\nSaved: {OUT_DRIVER}')
    print(f'       {OUT_TEAM}')
    print(f'       {OUT_PROFILE}')

    return {'coverage': cov, 'drivers': drivers, 'teams': teams,
            'profile': profile, 'verdict': verdict, 'linked': linked,
            'background': back}


if __name__ == '__main__':
    build()
