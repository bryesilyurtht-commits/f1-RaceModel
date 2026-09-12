"""
F1 Prediction Simulation - inspect_stints.py
Diagnostic: shows where the long stints in stint_index.csv actually came from.

Not part of the pipeline. Run it when a stint cap looks impossible - a 50-lap
SOFT at Zandvoort while MEDIUM caps at 33, say. That ordering cannot happen on
a real tyre, so something upstream is wrong and this finds out what.

Two things are checked against each other:

  stint_length  the raw tyre-age span, measured over every lap of the stint
  n_laps_clean  laps that survived cleaning, so no neutralization or pit laps

A big gap between them means most of the stint was run under a safety car, in
the wet, or behind a red flag - none of which say anything about how long the
tyre lasts. A 50-lap stint with 12 clean laps is not a 50-lap stint.

Usage:
    python inspect_stints.py
    python inspect_stints.py --race Dutch --compound SOFT
"""

import os
import sys

import numpy as np
import pandas as pd

pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
TOP_N = 15


def load():
    df = pd.read_csv(os.path.join(DATA_DIR, 'stint_index.csv'))
    if 'usable' in df.columns:
        df = df[df['usable']]
    return df[df['Compound'].isin(COMPOUNDS)]


def main():
    race_filter = compound_filter = None
    if '--race' in sys.argv:
        race_filter = sys.argv[sys.argv.index('--race') + 1].lower()
    if '--compound' in sys.argv:
        compound_filter = sys.argv[sys.argv.index('--compound') + 1].upper()

    df = load()
    df['clean_share'] = df['n_laps_clean'] / df['stint_length']

    print('=' * 100)
    print('COMPOUND ORDERING CHECK')
    print('=' * 100)
    print('HARD should outlast MEDIUM, MEDIUM should outlast SOFT. Any track')
    print('breaking that is measuring something other than the tyre.\n')

    piv = df.pivot_table(index='Race', columns='Compound',
                         values='stint_length', aggfunc='max')
    for c in COMPOUNDS:
        if c not in piv.columns:
            piv[c] = np.nan
    bad = piv[(piv['SOFT'] > piv['MEDIUM']) | (piv['MEDIUM'] > piv['HARD'])]
    print(f'Tracks with an impossible ordering: {len(bad)} of {len(piv)}\n')
    if not bad.empty:
        print(bad[COMPOUNDS].to_string())

    print('\n' + '=' * 100)
    print('RAW SPAN vs CLEAN LAPS, longest stints overall')
    print('=' * 100)
    print('clean_share near 1 means the stint was genuinely raced. Low values')
    print('mean the tyre age spans laps that were never racing laps.\n')

    cols = ['Season', 'Race', 'Driver', 'Compound', 'Stint', 'stint_length',
            'n_laps_clean', 'clean_share', 'race_laps', 'tyre_life_start',
            'tyre_life_end', 'stint_first_lap', 'stint_last_lap']
    cols = [c for c in cols if c in df.columns]

    sub = df
    if race_filter:
        sub = sub[sub['Race'].str.lower().str.contains(race_filter)]
    if compound_filter:
        sub = sub[sub['Compound'] == compound_filter]

    print(sub.nlargest(TOP_N, 'stint_length')[cols].round(2).to_string(index=False))

    print('\n' + '=' * 100)
    print('THE SAME LIST, RANKED BY CLEAN LAPS INSTEAD')
    print('=' * 100)
    print('This is the honest version of "longest stint": laps actually raced\n')
    print(sub.nlargest(TOP_N, 'n_laps_clean')[cols].round(2).to_string(index=False))

    print('\n' + '=' * 100)
    print('PER COMPOUND: max raw span vs max clean laps')
    print('=' * 100)
    print(df.groupby('Compound')
            .agg(max_span=('stint_length', 'max'),
                 p95_span=('stint_length', lambda s: s.quantile(0.95)),
                 max_clean=('n_laps_clean', 'max'),
                 p95_clean=('n_laps_clean', lambda s: s.quantile(0.95)),
                 median_clean_share=('clean_share', 'median'))
            .round(2).to_string())

    print('\n' + '=' * 100)
    print('STINTS WHERE MOST LAPS WERE NOT RACING LAPS')
    print('=' * 100)
    junk = df[(df['stint_length'] >= 25) & (df['clean_share'] < 0.6)]
    print(f'Long stints with under 60% clean laps: {len(junk)}\n')
    if not junk.empty:
        print(junk.nlargest(TOP_N, 'stint_length')[cols].round(2).to_string(index=False))
        print('\nThese are the ones inflating the caps. A red-flagged or wet')
        print('race leaves a tyre nominally 50 laps old having raced 15 of them.')

    if race_filter:
        print('\n' + '=' * 100)
        print(f'ALL STINTS AT {race_filter.upper()}, BY COMPOUND')
        print('=' * 100)
        for c in COMPOUNDS:
            cc = sub[sub['Compound'] == c] if not compound_filter else sub
            if cc.empty:
                continue
            print(f'\n--- {c}: {len(cc)} stints ---')
            print(cc.groupby('Season')
                    .agg(n=('Stint', 'size'),
                         max_span=('stint_length', 'max'),
                         max_clean=('n_laps_clean', 'max'),
                         median_clean=('n_laps_clean', 'median'))
                    .to_string())


if __name__ == '__main__':
    main()