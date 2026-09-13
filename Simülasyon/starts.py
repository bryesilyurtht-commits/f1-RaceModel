"""
F1 Prediction Simulation - v2.7 - starts.py
What actually happens between the grid and the end of lap one.

The model currently holds the grid order through lap one and only spreads the
time gaps - `LOCK_START_ORDER`. Replacing that with something data-driven needs
the data first, and the data needs cleaning before it means anything.

The trap this module exists to avoid
------------------------------------
A driver who starts eighth and is sixth at the end of lap one has not
necessarily passed two cars. Two cars ahead may have retired at turn one, or
pitted for a new front wing, or started from the pit lane and never been ahead
at all. Counting that as two successful starts teaches the model that the
midfield gains places for free, and the model will then hand those places out
in every simulation.

So the change measured here is computed among the cars that were actually
racing: everyone who took the start, completed lap one, and neither pitted nor
retired during it. Within that set the grid order and the lap-one order are
both re-ranked, and the difference is how many cars a driver really got past.

What the resolution allows
--------------------------
The archive gives a position at the end of lap one and nothing inside it. The
launch, the run to turn one, the braking and the first sequence of corners are
one observation, not four. They are modelled and reported as one combined
start performance, and this module does not pretend to separate them - a
coefficient named "reaction" would be a name attached to a number nothing
measured.

Outputs:
    data/start_profile.csv       per grid slot: retention, gain, loss, spread
    data/start_events.csv        per car-race: the cleaned observation
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LAPS = os.path.join(DATA_DIR, 'laps_2018_2025.csv')
RESULTS = os.path.join(DATA_DIR, 'results_2018_2025.csv')

PROFILE_OUT = os.path.join(DATA_DIR, 'start_profile.csv')
EVENTS_OUT = os.path.join(DATA_DIR, 'start_events.csv')

# TrackStatus is a concatenation of every flag seen during the lap: 1 green,
# 2 yellow, 4 safety car, 5 red, 6 and 7 virtual safety car. A lap that saw a
# car-stopping flag is not a lap anyone raced, and the start it followed is a
# different event from a green-flag start.
NEUTRAL_CODES = ('4', '5', '6', '7')

# The regulation regime the model is built for. 2018-2021 raced a different
# car with different aerodynamics in traffic, and start behaviour is exactly
# the sort of thing that would change with it.
ERA_FIRST_SEASON = 2022

MIN_SLOT_SAMPLE = 25       # observations a grid slot needs before it is fitted


def load():
    """Lap one, the grid it came from, and how the race ended for each car."""
    laps = pd.read_csv(LAPS, usecols=['Driver', 'LapNumber', 'Position',
                                      'Season', 'Race', 'RoundNumber',
                                      'TrackStatus', 'PitInTime_s',
                                      'PitOutTime_s', 'race_laps'])
    results = pd.read_csv(RESULTS)
    results['grid_pos'] = pd.to_numeric(results['GridPosition'], errors='coerce')

    last_lap = (laps.groupby(['Season', 'RoundNumber', 'Driver'])['LapNumber']
                .max().rename('last_lap').reset_index())

    lap1 = laps[laps['LapNumber'] == 1].copy()
    lap1['TrackStatus'] = lap1['TrackStatus'].astype(str)

    frame = lap1.merge(results[['Driver', 'Season', 'RoundNumber', 'grid_pos',
                                'Status']],
                       on=['Driver', 'Season', 'RoundNumber'], how='inner')
    return frame.merge(last_lap, on=['Season', 'RoundNumber', 'Driver'],
                       how='left')


def classify(frame):
    """
    Label every car-race, so that only the ones that raced are measured.

    The categories are exclusive and every row gets one, because a row quietly
    belonging to none of them is a row that leaves the sample without being
    counted as excluded.
    """
    frame = frame.copy()
    status = frame['TrackStatus'].astype(str)

    frame['neutralised'] = status.apply(
        lambda s: any(code in s for code in NEUTRAL_CODES))
    frame['pitted'] = (frame['PitInTime_s'].notna()
                       | frame['PitOutTime_s'].notna())
    frame['retired_lap1'] = frame['last_lap'] <= 1
    frame['no_grid'] = frame['grid_pos'].isna()
    frame['no_position'] = frame['Position'].isna()

    def label(row):
        if row['no_grid'] or row['no_position']:
            return 'incomplete record'
        if row['retired_lap1']:
            return 'lap-one retirement'
        if row['pitted']:
            return 'lap-one pit or pit-lane start'
        if row['neutralised']:
            return 'lap one neutralised'
        return 'raced'

    frame['category'] = frame.apply(label, axis=1)
    return frame


def racing_change(frame):
    """
    Places actually gained, counted among the cars that were racing.

    Both the grid order and the lap-one order are re-ranked inside the racing
    set, so a car ahead that retired or pitted does not hand everyone behind it
    a free place. This is the difference between measuring starts and measuring
    other people's misfortune.
    """
    raced = frame[frame['category'] == 'raced'].copy()
    parts = []
    for _, group in raced.groupby(['Season', 'RoundNumber']):
        group = group.copy()
        group['grid_rank'] = group['grid_pos'].rank(method='first')
        group['lap1_rank'] = group['Position'].rank(method='first')
        group['change'] = group['grid_rank'] - group['lap1_rank']
        group['field'] = len(group)
        parts.append(group)
    if not parts:
        return raced.assign(grid_rank=np.nan, lap1_rank=np.nan, change=np.nan,
                            field=np.nan)
    return pd.concat(parts, ignore_index=True)


def slot_profile(raced, min_sample=MIN_SLOT_SAMPLE):
    """
    Per grid slot: how often a car holds it, gains, loses, and by how much.

    The spread is what the start model has to reproduce. The mean is close to
    zero everywhere by construction - places gained by one car are lost by
    another - so a model matched on the mean alone would be a model that never
    moves anybody.
    """
    rows = []
    for slot, group in raced.groupby(raced['grid_rank'].round().astype(int)):
        n = len(group)
        change = group['change']
        rows.append(dict(
            slot=slot, n=n,
            held=float((change == 0).mean()),
            gained=float((change > 0).mean()),
            lost=float((change < 0).mean()),
            mean_change=float(change.mean()),
            sd_change=float(change.std()),
            p10=float(change.quantile(0.10)),
            p90=float(change.quantile(0.90)),
            thin=n < min_sample))
    return pd.DataFrame(rows).sort_values('slot').reset_index(drop=True)


def front_row_stability(raced, front=2):
    """
    What START_FRONT_STABILITY is supposed to stand for, measured.

    The constant is 0.35 and it scales the noise the front row is exposed to,
    not a probability of holding position - so it cannot be read off this table
    directly. What this gives is the thing the constant exists to produce: how
    much less the front of the grid is reshuffled than the back.
    """
    rows = []
    for name, mask in (('front row', raced['grid_rank'] <= front),
                       ('rows 2-5', (raced['grid_rank'] > front)
                        & (raced['grid_rank'] <= 10)),
                       ('back half', raced['grid_rank'] > 10)):
        group = raced[mask]
        if group.empty:
            continue
        rows.append(dict(band=name, n=len(group),
                         held=float((group['change'] == 0).mean()),
                         lost_one_or_more=float((group['change'] < 0).mean()),
                         sd_change=float(group['change'].std())))
    return pd.DataFrame(rows)


def leader_retention(raced):
    """How often the car on pole still leads at the end of lap one."""
    pole = raced[raced['grid_rank'] == 1]
    if pole.empty:
        return None
    return {'n': len(pole), 'still_leading': float((pole['lap1_rank'] == 1).mean())}


def grid_side_test(raced, degree=3, draws=2_000, seed=42):
    """
    Whether the side of the grid explains anything the slot does not.

    The first version of this test scored each car against the mean change of
    its own slot and compared odd slots against even ones. It returned exactly
    zero, and it had to: parity is a deterministic function of the slot, so
    subtracting slot means removes the entire effect being tested. The test
    could not have found anything, which is worse than finding nothing.

    What makes a side effect identifiable at all is that the clean line is not
    on the odd side everywhere - it depends on which way the circuit turns out
    of the grid. So the quantity with any hope of being real is not a single
    odd-side advantage across the calendar; it is per-circuit parity effects
    that disagree with each other. That is what is tested here.

    Position is removed with a smooth curve in grid rank rather than with slot
    means, so parity survives the subtraction. Each circuit then gets its own
    odd-minus-even residual, and the spread of those is compared against
    permuting the side label within each race - which keeps field size, circuit
    and conditions fixed and destroys only the association in question.

    Nothing clearing the null returns no coefficient. The brief is explicit
    that an unseparable effect gets zero, and that zero is a simplification
    rather than a finding that the sides are equal.
    """
    frame = raced.copy()
    frame['slot'] = frame['grid_rank'].round().astype(int)
    frame['odd'] = (frame['slot'] % 2) == 1

    # A smooth trend in grid position, not a per-slot mean: the mean would take
    # parity with it and guarantee the answer.
    coefficients = np.polyfit(frame['slot'], frame['change'], degree)
    frame['residual'] = frame['change'] - np.polyval(coefficients, frame['slot'])

    def spread(side_labels):
        """Spread of per-circuit odd-minus-even residuals."""
        effects = []
        for _, index in circuits.items():
            odd = side_labels[index]
            residual = residuals[index]
            if odd.all() or not odd.any():
                continue
            effects.append(residual[odd].mean() - residual[~odd].mean())
        return float(np.std(effects)) if len(effects) > 1 else 0.0, effects

    residuals = frame['residual'].to_numpy()
    circuits = {name: group.index.to_numpy()
                for name, group in frame.reset_index(drop=True)
                .assign(residual=residuals).groupby('Race')}
    frame = frame.reset_index(drop=True)
    residuals = frame['residual'].to_numpy()
    labels = frame['odd'].to_numpy()

    observed_spread, effects = spread(labels)

    rng = np.random.default_rng(seed)
    race_index = [group.index.to_numpy() for _, group
                  in frame.groupby(['Season', 'RoundNumber'])]
    null = np.empty(draws)
    for d in range(draws):
        shuffled = labels.copy()
        for index in race_index:
            shuffled[index] = rng.permutation(labels[index])
        null[d] = spread(shuffled)[0]

    p = float((null >= observed_spread).mean())
    pooled = float(residuals[labels].mean() - residuals[~labels].mean())
    return {'circuit_spread': observed_spread, 'p_value': p, 'draws': draws,
            'n': len(frame), 'circuits': len(effects),
            'pooled_odd_advantage': pooled,
            'supported': bool(p < 0.05),
            'note': ('per-circuit side effects disagree more than chance'
                     if p < 0.05 else
                     'not separable from grid position in this sample; '
                     'the model carries no side term')}


def build(era_only=True):
    """The whole measurement, from the archive to the two output tables."""
    frame = classify(load())
    if era_only:
        frame = frame[frame['Season'] >= ERA_FIRST_SEASON]
    raced = racing_change(frame)
    return frame, raced, slot_profile(raced)


def main():
    frame, raced, profile = build()

    print(f'\n=== start observations, {ERA_FIRST_SEASON}+ ===')
    counts = frame['category'].value_counts()
    for name, n in counts.items():
        print(f'  {name:32s} {n:5d}')
    print(f'  {"":32s} {len(frame):5d} total')

    print(f'\n  {len(raced)} racing observations across '
          f'{raced.groupby(["Season", "RoundNumber"]).ngroups} races')

    print('\n=== by grid slot ===')
    print('  slot     n   held  gained   lost   mean     sd')
    for _, row in profile.iterrows():
        mark = '  thin' if row['thin'] else ''
        print(f'  {row["slot"]:4d} {row["n"]:5d}  {row["held"]:.2f}   '
              f'{row["gained"]:.2f}   {row["lost"]:.2f}  {row["mean_change"]:+.2f}  '
              f'{row["sd_change"]:5.2f}{mark}')

    print('\n=== front-row stability, measured ===')
    print(front_row_stability(raced).to_string(index=False))
    leader = leader_retention(raced)
    print(f'\n  pole still leading at the end of lap one: '
          f'{leader["still_leading"]:.1%} of {leader["n"]}')

    print('\n=== grid side ===')
    side = grid_side_test(raced)
    print(f'  pooled odd-slot residual {side["pooled_odd_advantage"]:+.3f} places')
    print(f'  per-circuit spread {side["circuit_spread"]:.3f} over '
          f'{side["circuits"]} circuits, p = {side["p_value"]:.3f} '
          f'({side["draws"]} permutations)')
    print(f'  {side["note"]}')

    profile.to_csv(PROFILE_OUT, index=False)
    keep = ['Season', 'RoundNumber', 'Race', 'Driver', 'grid_pos', 'Position',
            'grid_rank', 'lap1_rank', 'change', 'field']
    raced[keep].to_csv(EVENTS_OUT, index=False)
    print(f'\n  {PROFILE_OUT}')
    print(f'  {EVENTS_OUT}')


if __name__ == '__main__':
    main()
