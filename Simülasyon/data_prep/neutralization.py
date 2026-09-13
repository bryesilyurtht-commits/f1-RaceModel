"""
F1 Prediction Simulation - neutralization.py
Measures safety car and virtual safety car frequency per track.

TrackStatus is a per-lap string that can carry several codes at once:
    1 green   2 yellow   4 safety car   5 red flag   6 VSC deployed   7 VSC ending

A lap counts as neutralised if any car on track reports that code, so the flag
is read across the whole field rather than one driver. Contiguous neutralised
laps are collapsed into one event, which is what the simulation samples.

Outputs:
    data/neutralization_by_race.csv    race level
    data/neutralization.csv            track level, ready for simulate.py
"""

import os

import numpy as np
import pandas as pd

from Simülasyon.data_prep.dataset import load_laps, iter_races

# --- config -----------------------------------------------------------------

# The repo root, found from this file rather than written down. The path was
# hardcoded to one machine, which is fine until the project runs anywhere
# else - a checkout, a colleague's laptop, or the Linux box Streamlit Cloud
# serves it from, where that path simply does not exist.
# Three levels up, not two: this file lives in Simülasyon/data_prep/,
# so the repo root is one directory further than it used to be.
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEASONS = [2022, 2023, 2024, 2025]

SC_CODE = '4'
VSC_CODES = ('6', '7')
RED_CODE = '5'

MIN_DRIVERS = 8
WET_LAMBDA_MULTIPLIER = 1.8    # roadmap: wet races neutralise far more often

# A street circuit has no run-off. Anywhere else a car in the barrier can be
# recovered under a virtual safety car or a full one, because the marshals have
# somewhere to stand. Between walls they do not: the traffic has to stop
# completely before anyone can go near it, so an incident that would be a
# safety car at a permanent circuit gets promoted to a red flag here.
#
# That is the reason for treating the two kinds of circuit as separate
# populations rather than pooling them. It is a claim about the world and the
# measurement can refute it - main() prints the two rates side by side so it
# can be checked rather than assumed.
#
# Miami and Melbourne are part street, part permanent. They are deliberately
# left out: if the measurement puts their red-flag rate up with the street
# circuits, adding them is a v2.3 question.
STREET_CIRCUITS = [
    'Monaco Grand Prix',
    'Azerbaijan Grand Prix',
    'Singapore Grand Prix',
    'Saudi Arabian Grand Prix',
    'Las Vegas Grand Prix',
]

# --- event extraction -------------------------------------------------------


def lap_flags(race_df):
    """Per race lap: was SC out, was VSC out, was it red flagged."""
    df = race_df[['LapNumber', 'TrackStatus']].copy()
    df['TrackStatus'] = df['TrackStatus'].astype(str)

    grouped = df.groupby('LapNumber')['TrackStatus'].apply(lambda s: ''.join(s))
    sc = grouped.apply(lambda s: SC_CODE in s)
    vsc = grouped.apply(lambda s: any(c in s for c in VSC_CODES))
    red = grouped.apply(lambda s: RED_CODE in s)

    # a lap under SC should not also be counted as VSC
    vsc = vsc & ~sc
    return pd.DataFrame({'sc': sc, 'vsc': vsc, 'red': red}).sort_index()


def count_events(flag_series):
    """Collapses contiguous True runs into events. Returns (count, [durations])."""
    arr = flag_series.to_numpy().astype(bool)
    if not arr.any():
        return 0, []

    durations, run = [], 0
    for v in arr:
        if v:
            run += 1
        elif run:
            durations.append(run)
            run = 0
    if run:
        durations.append(run)
    return len(durations), durations


# --- collection -------------------------------------------------------------


def collect():
    rows = []
    laps = load_laps(SEASONS)

    for season, name, race_laps, race_df in iter_races(laps):
        if race_df['Driver'].nunique() < MIN_DRIVERS:
            continue

        flags = lap_flags(race_df)
        sc_n, sc_dur = count_events(flags['sc'])
        vsc_n, vsc_dur = count_events(flags['vsc'])
        red_n, _ = count_events(flags['red'])

        compounds = race_df['Compound'].astype(str).str.upper()
        known = compounds.isin({'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'})
        wet_share = (compounds[known].isin({'INTERMEDIATE', 'WET'}).mean()
                     if known.any() else 0.0)

        rows.append({
            'Season': season,
            'Race': name,
            'race_laps': race_laps,
            'sc_events': sc_n,
            'sc_laps': int(flags['sc'].sum()),
            'sc_mean_duration': float(np.mean(sc_dur)) if sc_dur else np.nan,
            'vsc_events': vsc_n,
            'vsc_laps': int(flags['vsc'].sum()),
            'vsc_mean_duration': float(np.mean(vsc_dur)) if vsc_dur else np.nan,
            'red_events': red_n,
            'wet_share': wet_share,
        })

        print(f'  {season} {name:32s} SC {sc_n} ({flags["sc"].sum():2d} laps)  '
              f'VSC {vsc_n} ({flags["vsc"].sum():2d})  '
              f'red {red_n}  wet {wet_share:.0%}')

    return pd.DataFrame(rows)


def category_of(race):
    return 'street' if race in STREET_CIRCUITS else 'permanent'


def category_rates(hist):
    """
    Red flags per race for each kind of circuit, pooled.

    Pooled rather than averaged over circuits: sum of red flags divided by sum
    of races. Averaging per-circuit rates would give a track with two races the
    same say as one with four, and with an event this rare that is most of the
    noise.
    """
    out = {}
    hist = hist.copy()
    hist['category'] = hist['Race'].map(category_of)
    for cat, grp in hist.groupby('category'):
        out[cat] = {
            'rf_count': int(grp['red_events'].sum()),
            'rf_races': int(len(grp)),
            'rf_lambda': float(grp['red_events'].sum() / max(len(grp), 1)),
        }
    return out


def aggregate(hist):
    rows = []
    for race, grp in hist.groupby('Race'):
        dry = grp[grp['wet_share'] <= 0.15]
        src = dry if len(dry) >= 2 else grp

        # Red flags are counted over every race, wet included, unlike the SC
        # and VSC rates above. A red flag is most often thrown *because* the
        # track is undriveable, so dropping the wet races would remove the
        # cases the number is meant to describe.
        rf_count = int(grp['red_events'].sum())
        rf_races = int(len(grp))

        rows.append({
            'Race': race,
            'sc_lambda': float(src['sc_events'].mean()),
            'sc_duration': float(src['sc_mean_duration'].mean(skipna=True))
            if src['sc_mean_duration'].notna().any() else 4.0,
            'vsc_lambda': float(src['vsc_events'].mean()),
            'vsc_duration': float(src['vsc_mean_duration'].mean(skipna=True))
            if src['vsc_mean_duration'].notna().any() else 1.5,
            'red_lambda': float(src['red_events'].mean()),
            'rf_count': rf_count,
            'rf_races': rf_races,
            'rf_lambda': float(rf_count / max(rf_races, 1)),
            'category': category_of(race),
            'p_any_sc': float((src['sc_events'] > 0).mean()),
            'race_laps': int(src['race_laps'].median()),
            'n_seasons': len(grp),
            'n_dry': len(dry),
        })

    out = pd.DataFrame(rows).sort_values('sc_lambda', ascending=False)
    return out.reset_index(drop=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Seasons: {SEASONS}\n')

    hist = collect()
    if hist.empty:
        print('Nothing collected.')
        return

    tracks = aggregate(hist)

    print('\n===== field-wide =====')
    print(f'SC per race  : {hist["sc_events"].mean():.2f}')
    print(f'VSC per race : {hist["vsc_events"].mean():.2f}')
    print(f'Races with SC: {(hist["sc_events"] > 0).mean():.0%}')
    print(f'SC duration  : {hist["sc_mean_duration"].mean():.1f} laps')
    print(f'VSC duration : {hist["vsc_mean_duration"].mean():.1f} laps')

    dry = hist[hist['wet_share'] <= 0.15]
    wet = hist[hist['wet_share'] > 0.15]
    if len(wet) >= 2:
        ratio = wet['sc_events'].mean() / max(dry['sc_events'].mean(), 1e-6)
        print(f'\nWet vs dry SC rate: {ratio:.2f}x '
              f'(roadmap assumes {WET_LAMBDA_MULTIPLIER}x, n_wet={len(wet)})')

    print('\n===== red flags =====')
    print(f'Red flags seen : {int(hist["red_events"].sum())} over '
          f'{len(hist)} races  ({hist["red_events"].mean():.3f} per race)')
    print(f'Races with one : {(hist["red_events"] > 0).mean():.1%}')

    cats = category_rates(hist)
    print('\nBy circuit type - the claim is that street circuits red-flag more,')
    print('because an incident that is a safety car elsewhere has to stop the')
    print('race where there is no run-off. If these two are close, that claim')
    print('is wrong and the split should come out of simulate.py.')
    for cat in ('street', 'permanent'):
        c = cats.get(cat)
        if c:
            print(f'  {cat:10s} {c["rf_count"]:3d} red flags / {c["rf_races"]:3d} '
                  f'races = {c["rf_lambda"]:.3f} per race')
    if 'street' in cats and 'permanent' in cats:
        perm = max(cats['permanent']['rf_lambda'], 1e-9)
        print(f'  street circuits run {cats["street"]["rf_lambda"] / perm:.2f}x '
              f'the permanent-circuit rate')

    hot = tracks[tracks['rf_count'] > 0][['Race', 'category', 'rf_count',
                                          'rf_races', 'rf_lambda']]
    if not hot.empty:
        print('\nCircuits that actually saw one:')
        print(hot.round(3).to_string(index=False))
    print(f'\nCircuits with none: {int((tracks["rf_count"] == 0).sum())} of '
          f'{len(tracks)} - which is "not in four years", not "never".')

    print('\n===== per track =====')
    cols = ['Race', 'category', 'sc_lambda', 'sc_duration', 'vsc_lambda',
            'vsc_duration', 'p_any_sc', 'rf_count', 'rf_lambda',
            'n_seasons', 'n_dry']
    print(tracks[cols].round(2).to_string(index=False))

    hist.to_csv(os.path.join(DATA_DIR, 'neutralization_by_race.csv'), index=False)
    tracks.to_csv(os.path.join(DATA_DIR, 'neutralization.csv'), index=False)
    print(f'\nSaved 2 files to {DATA_DIR}')


if __name__ == '__main__':
    main()