"""
F1 Prediction Simulation - v2.3 - parameters.py
Every constant the model runs on, where it came from, and what it is worth.

    python -m Simülasyon.parameters

What this is
------------
An audit, not a feature. The model had accumulated numbers of four different
kinds under one appearance: things measured from data, things derived from
other measurements, things chosen by a person to make an output look right,
and things that were no longer used by anything at all. From the outside they
all looked like coefficients.

This module names each one, says which kind it is, records what it was
measured against if anything, and carries the sensitivity - how much the
output actually moves when it moves. A constant nobody can determine and that
changes nothing is not worth arguing about; one that nobody can determine and
that changes everything is the most important thing in the file.

What came out of it
-------------------
Four dead entries, three supported changes, one refusal, and one finding that
matters more than the rest:

  dead        POOLED_THRESHOLD is defined and never read. MIN_GAP is imported
              into simulate.py and never used. Tyre_model/tyre_cliff.py -
              CLIFF_FALLBACK included - is not imported by anything since the
              v1.2 curve replaced the k/W/m model. track['deg'], the end of the
              whole Pirelli rating chain in track_deg.py, reaches one badge in
              the interface and no arithmetic anywhere.

  changed     Safety car timing and pooling strength, all three measured
              against the thing they represent and checked on seasons that did
              not choose them.

  refused     The red-flag clip. It looked like a gag - it pins two thirds of
              the calendar to exactly 0.01 - so pooling was tried against it on
              held-out races. The clip won. It stays, now with evidence.

  the finding NOISE_AUTOCORR is 0.60 and the data says lap-to-lap persistence
              is 0.155, with 93% of a driver's lap-time variance sitting inside
              a stint rather than between stints. The 0.60 is not a correlation
              anyone measured; it was raised until the leader stopped winning
              98% of simulations. Adopting the measured value takes the
              favourite from 32.5% to 38.1% and the field's entropy from 1.60
              to 1.38 - the model becomes markedly more confident, because the
              variance the 0.60 was manufacturing has nowhere else to come
              from.

              So it is not changed here, and it is no longer described as a
              measurement either. It is the single largest open assumption in
              the model and the first thing v2.6's validation should settle.

Versioning
----------
PARAM_SET carries the whole set under a version name. Reverting is one line:
pass the previous name to `apply`. The same name never means two different sets
of numbers.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT = os.path.join(DATA_DIR, 'parameter_inventory.csv')

# --- the sets ---------------------------------------------------------------
# Only the values that changed are listed. Everything absent from a set keeps
# whatever simulate.py declares, so a set is a diff and not a second copy of
# the configuration that can drift away from it.
PARAM_SET = {
    # What simulate.py declared before this audit. It has to list the old
    # numbers explicitly: the module now carries the new ones, so an empty set
    # would revert to nothing and quietly succeed.
    'v2.2-legacy': {
        'SC_TIMING_MEAN_FRACTION': 0.60,
        'SC_TIMING_STD_FRACTION': 0.20,
        'SC_SHRINK_RACES': 6.0,
        'NOISE_AUTOCORR': 0.60,
    },
    'v2.3-measured': {
        # 146 background safety car starts, 2018-2025, with the ones matched
        # to an accident retirement removed because v2.2 generates those in
        # the race itself. The old 60%/20% said events cluster after the
        # opening laps; they cluster before the midpoint.
        'SC_TIMING_MEAN_FRACTION': 0.43,
        'SC_TIMING_STD_FRACTION': 0.32,
        # chosen by predicting 48 held-out races: log loss 0.6985 at 6
        # pseudo-races against 0.6803 at 24. Pooling matters a great deal -
        # no pooling at all scores 1.3848.
        'SC_SHRINK_RACES': 24.0,
        # named so a set is always a complete statement of the four constants
        # this audit touched, rather than a diff against whichever set ran last
        'NOISE_AUTOCORR': 0.60,
    },
    # What the measurements say if every one of them is taken at face value,
    # including the one that is not adopted. Kept so the consequence can be
    # reproduced rather than argued about.
    'v2.3-literal': {
        'SC_TIMING_MEAN_FRACTION': 0.43,
        'SC_TIMING_STD_FRACTION': 0.32,
        'SC_SHRINK_RACES': 24.0,
        'NOISE_AUTOCORR': 0.155,
    },
}

ACTIVE_SET = 'v2.3-measured'


# --- the inventory ----------------------------------------------------------
# kind:
#   measured    read off data that represents this quantity
#   derived     computed from another measurement
#   borrowed    taken from a donor circuit
#   hand-set    chosen by a person, and owed a measurement
#   fallback    used only when nothing else was available
#   safety      a numerical or scope guard, not a physical quantity
#   unused      defined, and read by nothing
#
# sensitivity is the change in the leading car's win probability across the
# candidates tried, at 4,000 simulations. It is a magnitude, not a p-value.
INVENTORY = [
    dict(name='NOISE_AUTOCORR', where='simulate.py',
         purpose='how much of a lap\'s deviation carries into the next',
         unit='correlation', value=0.60, kind='hand-set',
         scope='every lap of every car',
         target='lag-1 correlation of within-stint lap residuals',
         measured=0.155, holdout=0.123, n='61,234 laps, 3,021 stints',
         sensitivity=0.056, decision='unchanged',
         why='The measurement is sound and the change is not adopted, which '
             'needs saying plainly. 93% of a driver\'s lap-time variance is '
             'within a stint, so 0.155 is the right number for the quantity '
             'this parameter names. But 0.60 was never that quantity: it was '
             'raised until the favourite stopped winning 98% of simulations, '
             'and it is standing in for race-level variance the model has no '
             'other source for. Setting 0.155 takes the leader from 32.5% to '
             '38.1%. Removing the stand-in without replacing what it stands '
             'in for would be a worse model that looked better audited.'),

    dict(name='SC_TIMING_MEAN_FRACTION', where='simulate.py',
         purpose='where in the race a background neutralization starts',
         unit='fraction of race distance', value=0.60, kind='hand-set',
         scope='every simulated race', target='observed safety car starts',
         measured=0.43, holdout=None, n='146 background starts',
         sensitivity=0.026, decision='changed to 0.43',
         why='The old comment said events cluster after the opening laps have '
             'settled. They do the opposite: 38% of background starts fall in '
             'the first quarter of the race. Measured on starts only, so a '
             'six-lap safety car counts once, and with accident-linked events '
             'removed because v2.2 produces those from the crash itself.'),

    dict(name='SC_TIMING_STD_FRACTION', where='simulate.py',
         purpose='spread of that start lap', unit='fraction of race distance',
         value=0.20, kind='hand-set', scope='every simulated race',
         target='observed safety car starts', measured=0.32, holdout=None,
         n='146 background starts', sensitivity=0.013,
         decision='changed to 0.32',
         why='Wider than assumed. A normal this broad puts mass outside the '
             'race and the sampler clips it, so some pile-up at the ends is '
             'expected; an empirical distribution would be the better model '
             'and is left for a version with a reason to build one.'),

    dict(name='SC_SHRINK_RACES', where='simulate.py',
         purpose='how hard a circuit\'s own safety car rate is pulled toward '
                 'the calendar', unit='pseudo-races', value=6.0,
         kind='hand-set', scope='every circuit with few races',
         target='predicting whether a held-out race has a safety car',
         measured=24.0, holdout=0.6803, n='124 train, 48 held-out races',
         sensitivity=0.023, decision='changed to 24.0',
         why='Chosen on log loss over races the estimate had not seen. Six '
             'was a guess and a conservative one; four times as much pooling '
             'predicts better. No pooling at all scores 1.3848 against 0.6803, '
             'which is the real result here - per-circuit rates on four races '
             'are mostly noise.'),

    dict(name='RF_PROB_CAP', where='simulate.py',
         purpose='upper clip on a circuit\'s red flag probability',
         unit='per race', value=0.15, kind='safety',
         scope='10 of 36 circuits bind at it',
         target='predicting held-out red flags', measured=None,
         holdout=0.2074, n='48 held-out races', sensitivity=None,
         decision='unchanged, now with evidence',
         why='This looked like the clearest gag in the file: it pins two '
             'thirds of the calendar to exactly the floor. Pooling toward the '
             'calendar was tried as the alternative and lost on held-out '
             'races - 0.2074 for the clip against 0.2462 for pooling and '
             '0.2575 for the calendar rate alone. With three or four races '
             'per circuit and a 11% base rate, hard bounds beat a smooth '
             'shrink. Kept, and no longer on the suspicion list.'),

    dict(name='RF_PROB_FLOOR', where='simulate.py',
         purpose='lower clip on the same', unit='per race', value=0.01,
         kind='safety', scope='21 of 36 circuits bind at it',
         target='predicting held-out red flags', measured=None,
         holdout=0.2074, n='48 held-out races', sensitivity=None,
         decision='unchanged, now with evidence',
         why='Same test as the cap. Zero observed red flags at a circuit is '
             'not a probability of zero, and the floor is what says so.'),

    dict(name='SC_QUEUE_GAP', where='simulate.py',
         purpose='spacing between cars once the safety car queue has formed',
         unit='s', value=0.50, kind='hand-set',
         scope='the last lap of every full safety car',
         target='consecutive gaps at a restart', measured=0.72, holdout=None,
         n='153 restarts, weak proxy', sensitivity=0.007,
         decision='unchanged',
         why='The lap table has no cumulative time, so the only available '
             'proxy is the spread of lap times on the last neutralised lap - '
             'which measures how differently cars circulated, not how far '
             'apart they were. Changing a number on a proxy that measures '
             'something else is worse than leaving it. It moves the result by '
             '0.007 across the whole plausible range, so nothing turns on it.'),

    dict(name='AUTO_PASS_MARGIN', where='simulate.py',
         purpose='time gap above which a pass needs no dice',
         unit='s of cumulative time', value=1.20, kind='hand-set',
         scope='every following car on every green lap',
         target='pass opportunities, not completed passes', measured=None,
         holdout=None, n=None, sensitivity=0.008, decision='unchanged',
         why='Worth 0.008 of win probability and six overtakes a race across '
             '0.9 to 1.5, so it sets the overtaking count without setting the '
             'result. Tuning it to hit a target overtake count is the trap '
             'this version is supposed to avoid: the number it would be fitted '
             'to is produced by the pass model as a whole. It needs an '
             'opportunity-denominated measurement, which overtaking.py could '
             'supply and does not yet.'),

    dict(name='STRATEGY_TEMPERATURE', where='simulate.py',
         purpose='how much weight the softmax gives a dearer strategy',
         unit='seconds of cost', value=1.0, kind='hand-set',
         scope='one draw per car per simulation',
         target='observed spread of strategies', measured=None, holdout=None,
         n=None, sensitivity=0.007, decision='unchanged',
         why='Its unit is seconds, so it means what it means only while costs '
             'stay in seconds - worth knowing before anyone rescales them. '
             'Real strategies cannot calibrate it: what teams actually ran was '
             'shaped by weather, safety cars and traffic on the day, and '
             'treating the realised choice as the pre-race optimum would fit '
             'this constant to v2.0\'s job rather than its own.'),

    dict(name='DELTA_KNEE / DELTA_RATIO', where='clean.py',
         purpose='soft-knee compression of the measured pace spread',
         unit='s/lap, ratio', value='1.00 / 2.50', kind='hand-set',
         scope='every driver above 1 s/lap off pole',
         target='pace spread free of traffic', measured=None, holdout=None,
         n=None, sensitivity=None, decision='unchanged, with its cause named',
         why='Identity below the knee, so the order is preserved and the '
             'front of the field is untouched - the monotonicity this has to '
             'have, and it has it. But it is a patch on a known cause: a lap '
             'spent stuck behind another car still counts as clean, and '
             'backmarkers spend far more of the race there. The fix is a '
             'gap-to-car-ahead filter in clean.py, after which the ratio goes '
             'back to 1.0. Compressing the symptom is not measuring the '
             'phenomenon and this entry exists to stop it being mistaken for '
             'one.'),

    dict(name='POOLED_THRESHOLD', where='simulate.py',
         purpose='was the 50% pass-odds advantage', unit='s/lap', value=1.60,
         kind='unused', scope='nothing', target=None, measured=None,
         holdout=None, n=None, sensitivity=0.0, decision='dead, kept as a note',
         why='Defined and read by nothing. track_threshold solves for the '
             'threshold from the measured base rate instead, which is what '
             'replaced it. Left in place with this note rather than deleted, '
             'because the comment above it still explains the model.'),

    dict(name='MIN_GAP', where='tracks.py, imported by simulate.py',
         purpose='was the distance a blocked car was held at', unit='s',
         value=0.35, kind='unused', scope='nothing in the pass path',
         target=None, measured=None, holdout=None, n=None, sensitivity=0.0,
         decision='dead import',
         why='HELD_GAP replaced it in v1.5 with a measured 0.65, and the '
             'import stayed behind. Only 7% of the laps it was supposed to '
             'describe were anywhere near 0.35.'),

    dict(name='CLIFF_FALLBACK', where='Tyre_model/tyre_cliff.py',
         purpose='W and m when a cliff fit was rejected', unit='s, s/lap',
         value='per compound', kind='unused', scope='nothing',
         target=None, measured=None, holdout=None, n=None, sensitivity=0.0,
         decision='dead module',
         why='tyre_cliff.py is not imported by anything. The v1.2 rewrite '
             'replaced the two-segment k/W/m model with a fitted quadratic '
             'whose cliff comes from when teams actually stopped, and the '
             'whole module went with it. Calibrating its fallback would be '
             'tuning a number that cannot reach the simulation.'),

    dict(name='DEG_DONOR', where='deg_overrides.py',
         purpose='borrow a degradation curve from another circuit',
         unit='circuit name', value='British <- Belgian', kind='borrowed',
         scope='one circuit, not this one', target=None, measured=None,
         holdout=None, n='1 entry', sensitivity=0.0,
         decision='unchanged, out of scope for this race',
         why='The only entry is Silverstone borrowing Spa, and the target '
             'race is Monza, so it does not fire here at all. Evaluating a '
             'donor properly means hiding a circuit\'s own data and scoring '
             'the borrowed prediction against it, which is worth doing when '
             'the target race is one that uses a donor.'),

    dict(name='Pirelli ratings', where='track_deg.py',
         purpose='abrasion and stress, 1-5, into a degradation severity',
         unit='rating', value='per circuit', kind='hand-set',
         scope='reaches one badge and no arithmetic', target=None,
         measured=None, holdout=None, n=None, sensitivity=0.0,
         decision='unused by the engine, and now labelled so',
         why='The chain ends at track["deg"], which is displayed in the '
             'parameter panel and read by nothing in the lap loop - the v1.2 '
             'tyre curves took over. So there is no double counting, which was '
             'the thing to check, but the panel was implying a number mattered '
             'when it did not.'),
]


def sensitivity_table():
    """The inventory sorted by how much each constant is actually worth."""
    table = pd.DataFrame(INVENTORY)
    table['rank'] = pd.cut(table['sensitivity'].fillna(0.0),
                           [-0.001, 0.005, 0.02, 1.0],
                           labels=['low', 'medium', 'high'])
    return table


def apply(module, set_name=ACTIVE_SET):
    """
    Put a named set onto simulate.py, and hand back what was there before.

    Reverting is passing the previous name back, which is the whole of the
    rollback story §11 asks for. A set is a diff: anything it does not name is
    left exactly as the module declares it.
    """
    if set_name not in PARAM_SET:
        raise KeyError(f'no such parameter set: {set_name}')
    before = {}
    for key, value in PARAM_SET[set_name].items():
        before[key] = getattr(module, key)
        setattr(module, key, value)
    return before


def report():
    table = sensitivity_table()
    print(f'\n=== parameter inventory, set "{ACTIVE_SET}" ===')
    print(f'{len(table)} entries reviewed\n')

    for rank in ('high', 'medium', 'low'):
        rows = table[table['rank'] == rank]
        if rows.empty:
            continue
        print(f'--- {rank} sensitivity ---')
        for _, r in rows.iterrows():
            sens = ('-' if r['sensitivity'] is None
                    else f'{r["sensitivity"]:.3f}')
            print(f'  {r["name"]:<26} {str(r["value"]):>16}  '
                  f'{r["kind"]:<9} dP_win {sens:>6}  -> {r["decision"]}')
        print()

    dead = table[table['kind'] == 'unused']
    print(f'--- read by nothing: {len(dead)} ---')
    for _, r in dead.iterrows():
        print(f'  {r["name"]:<26} {r["where"]}')

    changed = table[table['decision'].str.startswith('changed')]
    print(f'\n--- changed in this set: {len(changed)} ---')
    for _, r in changed.iterrows():
        print(f'  {r["name"]:<26} {r["value"]} -> {r["measured"]}  '
              f'({r["n"]})')

    print(f'\n--- the open assumption ---')
    rho = table[table['name'] == 'NOISE_AUTOCORR'].iloc[0]
    print(f'  {rho["name"]}: {rho["value"]} in use, {rho["measured"]} '
          f'measured over {rho["n"]}')
    print(f'  {rho["why"]}')

    table.to_csv(OUT, index=False)
    print(f'\nSaved: {OUT}')
    return table


if __name__ == '__main__':
    report()
