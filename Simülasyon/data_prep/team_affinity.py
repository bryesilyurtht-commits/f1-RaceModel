"""
F1 Prediction Simulation - v1.5 - team_affinity.py
Which 2026 circuits suit which 2026 team.

Why a separate module from track_affinity.py
--------------------------------------------
track_affinity.py measures 2022-2025. Those are different cars under different
regulations, and 2026 changed both the chassis and the power unit. The question
was whether that history still says anything, and it was measured:

    2022-25 team affinity   -> 2026 race affinity   corr +0.076,  skill +0.6%
    2022-25 driver affinity -> 2026 race affinity   corr +0.122,  skill +1.5%
    2026 quali affinity     -> 2026 race affinity   corr +0.365,  skill +13.3%

The old cars carry essentially nothing, and simulate.py was taking 40% of its
team term from them. This module builds the team term from 2026 alone.

Definition
----------
    affinity = the team's average position across its OTHER 2026 races
             - the team's average position here

Positive means the circuit suits them. A team that runs 5th everywhere and 5th
here scores zero; the number is suitability, not speed.

Censoring - the decision that mattered most
-------------------------------------------
A retired car is classified behind every car that finished, so reading the
classification scores a team's affinity by its reliability. The last position
the car actually held on track is used instead: a car running 4th when its
engine let go scores 4th here, not 19th. This is not a detail.

    raw classified Position          skill  +3.0%
    last position actually held      skill +12.9%

Retirees' classified positions average 19.8 while their last running position
averages 16.1, and that 3.7-position gap was the bulk of the measured signal
before censoring recovered it.

What is predicted, and how
--------------------------
Qualifying happens before the race, so the target circuit's 2026 qualifying is
available while its race is not. A team's qualifying affinity at a circuit
predicts its race affinity there with corr +0.36. Regressed, one position of
qualifying affinity is worth about half a position of race affinity, and that
shrinkage is fitted here rather than assumed.

What was tried and rejected
---------------------------
The original plan pooled circuits by similarity: weight a circuit's neighbours
in the PC1/PC2 space of track_features.py so that Monza and Spa inform Baku,
and so that a circuit with no 2026 race can be predicted at all. It does not
work, and the measurement is kept here because the idea is a natural one to
have again.

First a trap worth recording. Affinity is a deviation from the team's own
average, so the twelve values of any team sum to exactly zero. That makes the
mean of the other eleven circuits identically -y_i/(n-1): a leave-one-out
kernel is anti-correlated with the truth by construction, and every bandwidth
scored worse than predicting zero for a reason that had nothing to do with
circuits. Subtracting the training mean removes the artefact. With that fixed:

    race channel    centred kernel corr +0.157,  permutation p = 0.050
    quali channel   centred kernel corr -0.169,  permutation p = 0.835

The qualifying channel's correlation is negative. The race channel's +0.157 is
the best of thirteen bandwidths tried, so its p = 0.050 is a selected maximum,
not a finding. Per-team correlations against PC1 average 0.32 where pure noise
on twelve points averages 0.25.

The team x circuit effect itself is real - teammates agree at +0.411, and 56%
of the affinity variance survives as signal - but PC1/PC2 does not explain it.
Similarity pooling stays off until some feature does.

Outputs:
    data/team_affinity_2026.csv
"""

import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simülasyon.data_prep.track_features import FEATURE_SET, standardise

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

SEASON = 2026

# The 2026 calendar and the feature table do not always agree on a name.
RACE_ALIASES = {
    'Barcelona Grand Prix': 'Spanish Grand Prix',
}

# Read the last position the car actually held instead of its classification.
# Off, this is a reliability ranking wearing a circuit-suitability label.
CENSOR_RETIREMENTS = True

MIN_RACES_PER_TEAM = 6      # fewer than this and a season average means little

# Development trend. Haas gained 6.6 positions across the season and Williams
# 5.7, which looks like "the late circuits suit them". Removing it is safe -
# calendar order is nearly independent of circuit character, corr(Round, PC1)
# = -0.12 - but it is also not worth it: taking the slope out costs a little
# skill, because a team's genuine mid-season form is part of what a circuit
# result is measuring.
#
#     detrend off   skill +12.9%
#     detrend on    skill +12.0%
DETREND = False

# See the module docstring. Measured, failed, kept switchable.
SIMILARITY_POOLING = False
BANDWIDTH_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

# Qualifying affinity is a noisy estimate of race affinity, so it is shrunk
# toward zero before use. The factor is fitted by leave-one-circuit-out rather
# than chosen; it comes out near 0.5, meaning a team qualifying two positions
# better than usual here is worth about one position of race affinity.
FIT_SHRINKAGE = True
FALLBACK_SHRINKAGE = 0.5

PARAMS_VERSION = '1.0.0'

# --- loading ----------------------------------------------------------------


def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise SystemExit(f'missing input: {path}')
    return pd.read_csv(path)


def canonical(name):
    return RACE_ALIASES.get(str(name), str(name))


def race_signal(results, laps):
    """
    One finishing position per car, retirements censored to the last position
    the car actually held.
    """
    res = results.copy()
    cp = res['ClassifiedPosition'].astype(str)
    res['classified'] = cp.str.fullmatch(r'\d+')
    res['dns'] = cp.str.upper().isin(['W', 'D', 'E'])

    last = (laps[laps['Position'].notna()]
            .sort_values('LapNumber')
            .groupby(['Race', 'Driver'])['Position'].last()
            .rename('last_pos').reset_index())
    res['Race'] = res['Race'].map(canonical)
    last['Race'] = last['Race'].map(canonical)
    res = res.merge(last, on=['Race', 'Driver'], how='left')

    if CENSOR_RETIREMENTS:
        # a retiree with no lap at all leaves nothing to read and drops out
        res['signal'] = np.where(res['classified'], res['Position'],
                                 res['last_pos'])
    else:
        res['signal'] = pd.to_numeric(res['Position'], errors='coerce')

    res = res[~res['dns'] & res['signal'].notna()].copy()
    return res[['Race', 'Round', 'Team', 'Driver', 'signal']]


def quali_signal(quali, include_target=True):
    """
    Qualifying order. The target race's qualifying is kept: it happens before
    the race, so using it is not leakage, and it is the only 2026 measurement
    the target circuit has.
    """
    q = quali.copy()
    if not include_target and 'is_target' in q.columns:
        q = q[~q['is_target'].astype(bool)]
    q['signal'] = pd.to_numeric(q['quali_pos'], errors='coerce')
    q = q[q['signal'].notna()].copy()
    q['Race'] = q['Race'].map(canonical)
    q['is_target'] = q.get('is_target', False)
    return q[['Race', 'Round', 'Team', 'Driver', 'signal', 'is_target']]


def centre_by_session(signal):
    """
    Expresses a position relative to the field that actually turned up.

    Sessions are not all the same size. Only 19 cars set a time in Australian
    qualifying and only 18 were running at the end in China, so positions
    there run 1..19 and 1..18 while elsewhere they run 1..22. The mean
    position in a 19-car session is 10.0 rather than 11.5, which makes every
    single team look 1.5 positions better than usual - and affinity is exactly
    a measure of "better than usual".

    Left alone this reads as "Australia suits the entire grid", which is not a
    thing a circuit can do. Subtracting the session's own mean removes it, and
    leaves affinity measuring what it claims to.
    """
    out = signal.copy()
    out['field'] = out.groupby('Race')['signal'].transform('size')
    out['signal'] = out['signal'] - out.groupby('Race')['signal'].transform('mean')
    return out


def team_table(signal):
    """Team position per race: the average of whichever of its cars ran."""
    keys = ['Team', 'Race', 'Round']
    agg = {'pos': ('signal', 'mean'), 'cars': ('signal', 'size')}
    if 'is_target' in signal.columns:
        agg['is_target'] = ('is_target', 'max')
    return signal.groupby(keys).agg(**agg).reset_index()


# --- affinity ---------------------------------------------------------------


def detrend(team):
    """
    Takes the season-long development slope out of a team's positions, so a
    team that started 15th and ended 9th is not read as "the late circuits
    suit it".
    """
    out = team.copy()
    out['trend'] = 0.0
    for _, g in out.groupby('Team'):
        if len(g) < 4:
            continue
        b = np.polyfit(g['Round'].to_numpy(float), g['pos'].to_numpy(float), 1)
        fit = np.polyval(b, g['Round'].to_numpy(float))
        out.loc[g.index, 'trend'] = fit - g['pos'].mean()
    out['pos_adj'] = out['pos'] - out['trend']
    return out


def affinity_table(team, reference_mask=None):
    """
    affinity = mean of the team's other races - its position here.

    reference_mask marks the rows that may contribute to a team's season
    average. The target circuit is measured but never used as its own
    reference, otherwise a circuit with no race would still be shaping the
    baseline it is compared against.
    """
    col = 'pos_adj' if 'pos_adj' in team.columns else 'pos'
    if reference_mask is None:
        reference_mask = pd.Series(True, index=team.index)

    rows = []
    for name, g in team.groupby('Team'):
        ref = g[reference_mask.loc[g.index]]
        if len(ref) < MIN_RACES_PER_TEAM:
            continue
        total, n = ref[col].sum(), len(ref)
        for _, r in g.iterrows():
            v = r[col]
            if reference_mask.loc[r.name]:
                # leave the row out of its own reference
                others = (total - v) / (n - 1)
                used = n - 1
            else:
                others, used = total / n, n
            rows.append({'Team': name, 'Race': r['Race'], 'Round': r['Round'],
                         'affinity': others - v, 'n_reference': used,
                         'cars': r.get('cars', np.nan)})
    return pd.DataFrame(rows)


# --- circuit coordinates ----------------------------------------------------


def pc_coordinates(features):
    """
    PC1 and PC2 of the four track features, with the sign pinned.

    An SVD is free to flip either axis, which would silently mirror the map
    between runs. PC1 is oriented so braking-heavy circuits are positive and
    PC2 so straight-line circuits are positive, matching the character map.
    """
    z, _ = standardise(features)
    x = z[FEATURE_SET].to_numpy(float)
    x = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)

    coords = x @ vt[:2].T
    if vt[0, FEATURE_SET.index('braking_share')] < 0:
        coords[:, 0] = -coords[:, 0]
    if vt[1, FEATURE_SET.index('straight_blend')] < 0:
        coords[:, 1] = -coords[:, 1]

    share = s ** 2 / (s ** 2).sum()
    out = pd.DataFrame({'Race': features['Race'],
                        'pc1': coords[:, 0], 'pc2': coords[:, 1]})
    out.attrs['share'] = (float(share[0]), float(share[1]))
    return out


def centred_kernel(train_pc, train_y, target_pc, h):
    """
    Similarity-weighted mean of the team's other circuits, with the plain mean
    of those circuits subtracted.

    The subtraction is not cosmetic. A team's affinities sum to zero, so the
    unsubtracted mean of the other eleven equals -y_i/(n-1) exactly, and any
    kernel built on it is anti-correlated with the truth for arithmetic
    reasons. What survives the subtraction is the only real question: do
    similar circuits say something different from dissimilar ones.
    """
    if len(train_y) == 0:
        return 0.0
    d2 = ((train_pc - target_pc) ** 2).sum(axis=1)
    w = np.exp(-d2 / (2.0 * h * h))
    if w.sum() <= 1e-12:
        return 0.0
    return float((w * train_y).sum() / w.sum() - train_y.mean())


# --- validation -------------------------------------------------------------


def skill(pred, truth, clip_lambda=True):
    """
    Shrunk mean-squared-error skill against predicting zero, plus the
    shrinkage the data itself asks for.

    Reporting raw MSE would punish a predictor that has the right shape at the
    wrong scale, and scale is exactly what a single season cannot pin down.
    Lambda is the regression of truth on prediction: how much of the predicted
    swing actually shows up.
    """
    p = np.asarray(pred, float)
    t = np.asarray(truth, float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if len(p) < 10 or p.std() < 1e-12:
        return {'n': len(p), 'corr': np.nan, 'lambda': 0.0, 'skill': 0.0}

    corr = float(np.corrcoef(p, t)[0, 1])
    # ddof must match on both sides. np.cov defaults to ddof=1 and np.var to
    # ddof=0, which quietly scales lambda by n/(n-1) - 11% on a ten-point
    # check and 0.8% on the real table, small enough to survive unnoticed.
    lam = float(np.cov(p, t, ddof=0)[0, 1] / np.var(p))
    if clip_lambda:
        lam = float(np.clip(lam, 0.0, 1.0))
    mse0 = float((t ** 2).mean())
    mse = float(((lam * p - t) ** 2).mean())
    return {'n': len(p), 'corr': corr, 'lambda': lam,
            'skill': 1 - mse / mse0 if mse0 > 0 else 0.0}


def validate_quali_channel(race_aff, quali_aff):
    """
    Leave one circuit out: predict a team's race affinity there from its
    qualifying affinity there, with the shrinkage fitted on the other circuits
    only. Fitting lambda on all twelve and then scoring on all twelve would be
    grading the model on its own training data.
    """
    j = (race_aff.rename(columns={'affinity': 'race_aff'})
         .merge(quali_aff.rename(columns={'affinity': 'q_aff'})[
             ['Team', 'Race', 'q_aff']], on=['Team', 'Race'], how='inner'))
    if j.empty:
        raise SystemExit('race and qualifying affinity tables did not overlap')

    circuits = sorted(j['Race'].unique())
    preds, truths = [], []
    for c in circuits:
        train = j[j['Race'] != c]
        test = j[j['Race'] == c]
        s = skill(train['q_aff'], train['race_aff'])
        preds.extend((s['lambda'] * test['q_aff']).tolist())
        truths.extend(test['race_aff'].tolist())

    out = skill(preds, truths)
    out['in_sample'] = skill(j['q_aff'], j['race_aff'])
    out['pairs'] = j
    return out


def validate_similarity(aff, pcs, label):
    """The rejected path, re-measured every run so the rejection stays honest."""
    m = aff.merge(pcs, on='Race', how='inner')
    best = None
    print(f'    {label} - centred kernel on PC1/PC2:')
    for h in BANDWIDTH_GRID:
        preds, truths = [], []
        for _, g in m.groupby('Team'):
            pc = g[['pc1', 'pc2']].to_numpy(float)
            y = g['affinity'].to_numpy(float)
            for i in range(len(g)):
                k = np.arange(len(g)) != i
                preds.append(centred_kernel(pc[k], y[k], pc[i], h))
                truths.append(y[i])
        s = skill(preds, truths)
        print(f'      h={h:<5.2f} corr {s["corr"]:+.3f}  skill {s["skill"]:+6.1%}')
        if best is None or s['corr'] > best[1]['corr']:
            best = (h, s)
    return best


# --- main -------------------------------------------------------------------


def build():
    results = read_csv(f'f1_{SEASON}_results.csv')
    laps = read_csv(f'f1_{SEASON}_laps.csv')
    quali = read_csv(f'f1_{SEASON}_quali.csv')

    race = team_table(centre_by_session(race_signal(results, laps)))
    qual = team_table(centre_by_session(quali_signal(quali)))

    if DETREND:
        race, qual = detrend(race), detrend(qual)

    # the target circuit has qualifying but no race, and must not be part of
    # the season average it is measured against
    ref = ~qual['is_target'].astype(bool)
    race_aff = affinity_table(race)
    quali_aff = affinity_table(qual, reference_mask=ref)

    target_races = sorted(qual.loc[~ref, 'Race'].unique())
    return race_aff, quali_aff, target_races


def main():
    print(f'===== team affinity {SEASON} =====')
    race_aff, quali_aff, target_races = build()
    print(f'  race channel : {race_aff["Team"].nunique()} teams, '
          f'{race_aff["Race"].nunique()} circuits, {len(race_aff)} cells')
    print(f'  quali channel: {quali_aff["Team"].nunique()} teams, '
          f'{quali_aff["Race"].nunique()} circuits, {len(quali_aff)} cells')
    print(f'  circuits with qualifying but no race: '
          f'{", ".join(target_races) if target_races else "none"}')
    print(f'  censoring {"on" if CENSOR_RETIREMENTS else "OFF"}, '
          f'detrend {"on" if DETREND else "off"}')

    # --- is there a team x circuit effect at all -----------------------------
    print('\n  --- is the effect real, before any model ---')
    j = race_aff.merge(quali_aff[['Team', 'Race', 'affinity']],
                       on=['Team', 'Race'], suffixes=('_race', '_quali'))
    r = float(np.corrcoef(j['affinity_race'], j['affinity_quali'])[0, 1])
    print(f'    race and qualifying agree in the same cell: corr {r:+.3f}  '
          f'(n={len(j)})')
    print('    two separate sessions pointing the same way is not something')
    print('    noise does; the effect exists whatever explains it.')

    # --- the channel that is actually used -----------------------------------
    print('\n  --- qualifying affinity -> race affinity, leave-one-circuit-out ---')
    v = validate_quali_channel(race_aff, quali_aff)
    ins = v['in_sample']
    print(f'    out of sample  n={v["n"]}  corr {v["corr"]:+.3f}  '
          f'skill over predicting zero {v["skill"]:+.1%}')
    print(f'    in sample      corr {ins["corr"]:+.3f}  '
          f'lambda {ins["lambda"]:.2f}')
    passed = v['skill'] > 0
    print(f'    VERDICT: {"PASS" if passed else "FAIL"}')

    lam = ins['lambda'] if (FIT_SHRINKAGE and passed) else FALLBACK_SHRINKAGE
    if not passed:
        print('    qualifying does not predict race affinity this season; '
              'the file will be zeros.')
        lam = 0.0

    # --- the rejected path, re-measured --------------------------------------
    print('\n  --- similarity pooling (disabled, re-measured) ---')
    features = read_csv('track_features.csv')
    pcs = pc_coordinates(features)
    v1, v2 = pcs.attrs['share']
    print(f'    {len(pcs)} circuits, PC1 {v1:.0%} / PC2 {v2:.0%} of the '
          f'feature variance')
    for label, aff in (('race', race_aff), ('quali', quali_aff)):
        h, s = validate_similarity(aff, pcs, label)
        print(f'      best h={h:.2f}, corr {s["corr"]:+.3f} - '
              f'{"would be used" if SIMILARITY_POOLING else "not used"}')

    # --- output --------------------------------------------------------------
    out = quali_aff.rename(columns={'affinity': 'affinity_quali'})[
        ['Team', 'Race', 'Round', 'affinity_quali', 'n_reference', 'cars']]
    out = out.merge(race_aff.rename(columns={'affinity': 'affinity_race'})[
        ['Team', 'Race', 'affinity_race']], on=['Team', 'Race'], how='left')

    out['affinity'] = (lam * out['affinity_quali']).round(4)
    out['shrinkage'] = round(lam, 4)
    out['raced_2026'] = out['affinity_race'].notna()
    out['source'] = 'quali x shrinkage'
    out['version'] = PARAMS_VERSION
    out = out.sort_values(['Race', 'affinity'], ascending=[True, False])

    path = os.path.join(DATA_DIR, f'team_affinity_{SEASON}.csv')
    out.round(4).to_csv(path, index=False, encoding='utf-8')
    print(f'\n  shrinkage applied: {lam:.3f} '
          f'(one position of qualifying affinity is worth '
          f'{lam:.2f} positions of race affinity)')
    print(f'  saved: {path}  ({len(out)} rows, '
          f'{out["Race"].nunique()} circuits)')

    for race in target_races:
        g = out[out['Race'] == race].sort_values('affinity', ascending=False)
        print(f'\n  {race} (no 2026 race - measured from its qualifying):')
        for _, r in g.iterrows():
            print(f'    {r["Team"]:18s} quali {r["affinity_quali"]:+6.2f}  '
                  f'-> affinity {r["affinity"]:+6.2f}')
    return out


if __name__ == '__main__':
    main()
