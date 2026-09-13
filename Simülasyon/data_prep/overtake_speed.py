"""
F1 Prediction Simulation - v1.5 - overtake_speed.py
Which 2026 cars get past, which hold position, and which are quick in a
straight line.

MEASUREMENT ONLY. Nothing here is wired into simulate.py, no constant
anywhere else is updated, and no other module changes behaviour. The output
is four CSVs and a console summary to be read before any of it is used.

Scope
-----
2026 alone. The cars changed chassis and power unit this season, so a 2024
car's ability to pass says nothing about a 2026 one - the same reasoning that
took the 2022-25 history out of the team affinity term, where it measured a
+0.6% skill against 2026 results. Twelve races is thin, and the thinness is
reported rather than papered over.

The target race is absent from the lap data by construction (fetch.py
withholds it), so nothing here can see the race being predicted. Qualifying
is a different matter: it runs before the race, so the target's qualifying
is used for the speed trap and marked as such.

Why the naive pass rate is not the answer
-----------------------------------------
overtaking.py emits one row per close lap. Read as "attempts", that counts a
single twelve-lap stalemate as twelve failures, and a pass completed on first
contact as one success. The rate it produces is therefore not a per-attempt
probability at all, and it is depressed hardest exactly where passing is
hard - which is where the number would be used.

On the 2022-25 events the two disagree by a factor of three:

    lap-level attempts    6,208      pass rate 0.172
    pursuits              2,144      success   0.497
    laps per pursuit      2.90 mean, 2 median, 25 max

So this module reports both. The directive's per-lap tables are produced
exactly as specified, and alongside them the same events regrouped into
pursuits, where the question "did a faster car get through" has a
well-defined denominator.

The censoring
-------------
A pursuit that ends without a pass has not necessarily failed. It can end
because either car pitted, because the race ran out, because a safety car
neutralised the fight, or because the follower retired. Only one of those is
evidence about passing, and the rest are right-censoring of exactly the kind
the tyre cliff work already deals with.

One case is genuinely ambiguous and is reported both ways rather than
decided here: the follower dropping back out of contact. That can be tyres
going off - censoring - or giving up on a pass that was never coming -
failure. The two readings bracket the truth and the gap between them is
printed.

Beyond that sits a selection effect no estimator fixes. A team only appears
as an attacker when it is behind a slower car, which for a front-running
team means the races that went wrong. Its attacking sample is drawn from its
own bad days. The model below controls for pace advantage and for the
circuit, which is the most that can be done; the residual bias is stated,
not removed.

Speed trap
----------
FastF1's SpeedST is a single-point peak speed, and it moves with DRS, engine
mode, fuel load and whoever happens to be in front. Read raw across a race it
can easily measure "this car had DRS open behind someone" rather than "this
car is quick in a straight line".

Two channels are taken instead of one:

    qualifying   every car on a flying lap, low fuel, DRS available to all,
                 no car ahead to tow. The cleanest read on drag level there
                 is, and it is pre-race information.
    race, early  the directive's window: the opening laps, before DRS
                 activates. Contaminated by fuel load and by the pack still
                 being bunched, but it is the channel that describes the
                 conditions a pass actually happens in.

Both are z-scored within each session, so a 340 km/h trap at Monza and a 290
at Hungary become comparable, and the two are correlated against each other
so the reader can see how much the contamination actually costs.

Outputs:
    data/overtake_pairs_2026.csv       team x team, per-lap and per-pursuit
    data/overtake_attacking_2026.csv   per team, attacking
    data/overtake_defending_2026.csv   per team, defending
    data/overtake_episodes_2026.csv    pursuit level, with censoring reason
    data/speed_trap_2026.csv           per team per race, both channels
    data/overtake_team_profile_2026.csv  the two-axis team map
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simülasyon.data_prep.overtaking import (collect_events, CLOSE_GAP, MIN_ADVANTAGE,
                                   PIT_BLACKOUT)
from Simülasyon.target_race import TARGET_EVENT, TRACK_ALIASES

warnings.filterwarnings('ignore')

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
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

SEASON = 2026

# below this many pursuits a team-pair cell is flagged, never dropped. The
# directive sets it in lap-attempts; it is applied to pursuits as well,
# because a cell holding eight laps can hold as few as three real fights.
MIN_ATTEMPTS = 8
MIN_EPISODES = 5

SPEED_TRAP_EARLY_LAPS = 5     # the pre-DRS window, best-effort not exact
MIN_TRAP_LAPS = 3             # fewer than this and a team-race median is noise

# Ridge weight for the attack/defence fit. Eleven teams give twenty
# coefficients from a few hundred pursuits, and without a penalty a team that
# happened to race one opponent gets an enormous coefficient off three fights.
RIDGE = 2.0

FETCH_SPEED_TRAP = True       # set False to reuse data/speed_trap_2026.csv

PARAMS_VERSION = '1.0.0'

CENSOR_REASONS = ('pass', 'pit', 'neutralised', 'race_end', 'retired',
                  'dropped_back')


# --- 2026 lap adapter -------------------------------------------------------


def load_2026_laps():
    """
    f1_2026_laps.csv in the shape overtaking.py's detection expects.

    dataset.py's cache is built from FastF1 for 2022-25 and holds seconds;
    fetch.py's 2026 file holds timedelta strings. The conversion here is the
    same one dataset.build applies, so collect_events sees identical columns
    either way.
    """
    path = os.path.join(DATA_DIR, f'f1_{SEASON}_laps.csv')
    if not os.path.exists(path):
        raise SystemExit(f'missing {path} - run: python -m Simülasyon.data_prep.fetch')

    df = pd.read_csv(path)
    df['LapTime_s'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    for col in ('PitInTime', 'PitOutTime'):
        df[col + '_s'] = pd.to_timedelta(df[col]).dt.total_seconds()
    df = df.drop(columns=['LapTime', 'PitInTime', 'PitOutTime'])

    df['TrackStatus'] = (df['TrackStatus'].astype(str)
                         .str.replace(r'\.0$', '', regex=True))
    df['Compound'] = df['Compound'].astype(str).str.upper()
    df['Season'] = SEASON
    df['race_laps'] = df.groupby('Race')['LapNumber'].transform('max')

    leaked = [r for r in df['Race'].unique()
              if any(a in str(r).lower() for a in TRACK_ALIASES)]
    if leaked:
        raise SystemExit(f'the target race is in the lap data: {leaked}')
    return df


def driver_teams(laps):
    """Driver -> team. A driver who changed teams keeps the one they ran most."""
    counts = laps.groupby(['Driver', 'Team']).size().reset_index(name='n')
    best = counts.sort_values('n').groupby('Driver').tail(1)
    return dict(zip(best['Driver'], best['Team']))


# --- part 1: the directive's per-lap tables ---------------------------------


def collect_2026_events(laps):
    """overtaking.py's detection, unchanged, applied race by race to 2026."""
    frames = []
    for race, grp in laps.groupby('Race', sort=True):
        events = collect_events(grp)
        if not events:
            print(f'    {race:32s} no usable events')
            continue
        ev = pd.DataFrame(events)
        ev['Race'] = race
        ev['Round'] = int(grp['Round'].iloc[0])
        ev['race_laps'] = int(grp['race_laps'].iloc[0])
        frames.append(ev)
        n = int((ev['advantage'] >= MIN_ADVANTAGE).sum())
        print(f'    {race:32s} close {len(ev):4d}  chances {n:4d}')

    if not frames:
        raise SystemExit('no events collected from the 2026 laps')
    return pd.concat(frames, ignore_index=True)


def pair_table(chances):
    """attacking_team x defending_team, on the directive's per-lap definition."""
    g = chances.groupby(['attacking_team', 'defending_team'])
    out = g.agg(n_attempts=('passed', 'size'),
                n_passes=('passed', 'sum'),
                mean_advantage=('advantage', 'mean')).reset_index()
    out['n_passes'] = out['n_passes'].astype(int)
    out['pass_rate'] = out['n_passes'] / out['n_attempts']
    out['low_confidence'] = out['n_attempts'] < MIN_ATTEMPTS
    return out.sort_values('n_attempts', ascending=False).reset_index(drop=True)


def side_table(chances, side):
    """Per-team totals on one side of the fight."""
    col = 'attacking_team' if side == 'attacking' else 'defending_team'
    g = chances.groupby(col)
    out = g.agg(n_attempts=('passed', 'size'),
                n_passes=('passed', 'sum'),
                mean_advantage=('advantage', 'mean'),
                n_races=('Race', 'nunique')).reset_index()
    out = out.rename(columns={col: 'Team'})
    out['n_passes'] = out['n_passes'].astype(int)
    out['pass_rate'] = out['n_passes'] / out['n_attempts']
    if side == 'defending':
        out = out.rename(columns={'n_attempts': 'n_exposed',
                                  'n_passes': 'n_passed_by'})
        out['passed_rate'] = out['n_passed_by'] / out['n_exposed']
        out = out.drop(columns=['pass_rate'])
    out['low_confidence'] = out.filter(like='n_').iloc[:, 0] < MIN_ATTEMPTS
    return out.sort_values('Team').reset_index(drop=True)


# --- part 2: pursuits and their censoring -----------------------------------


def build_episodes(chances, laps):
    """
    Regroups close laps into pursuits, and says why each one ended.

    A pursuit is a run of consecutive laps with the same attacker behind the
    same defender. It ends the moment that stops being true, and the reason
    it stopped is what separates evidence about passing from noise about pit
    windows and safety cars.
    """
    pit_laps, last_lap, neutral_laps, race_len = {}, {}, {}, {}
    for race, grp in laps.groupby('Race'):
        race_len[race] = int(grp['race_laps'].iloc[0])
        pitted = grp[grp['PitInTime_s'].notna() | grp['PitOutTime_s'].notna()]
        for drv, g in pitted.groupby('Driver'):
            pit_laps[(race, drv)] = set(g['LapNumber'].astype(int))
        for drv, g in grp.groupby('Driver'):
            last_lap[(race, drv)] = int(g['LapNumber'].max())
        flagged = grp[grp['TrackStatus'].astype(str) != '1']
        neutral_laps[race] = set(flagged['LapNumber'].astype(int))

    ev = chances.sort_values(['Race', 'attacker', 'defender', 'lap']).copy()
    keys = ['Race', 'attacker', 'defender']
    gap = ev.groupby(keys)['lap'].diff()
    ev['episode'] = (gap.ne(1) | gap.isna()).cumsum()

    rows = []
    for (race, att, dfn, epi), g in ev.groupby(keys + ['episode']):
        end = int(g['lap'].max())
        passed = bool(g['passed'].any())

        if passed:
            reason = 'pass'
        else:
            nxt = end + 1
            att_pit = nxt in pit_laps.get((race, att), ())
            dfn_pit = nxt in pit_laps.get((race, dfn), ())
            window = range(nxt, nxt + PIT_BLACKOUT + 1)
            att_pit = att_pit or any(l in pit_laps.get((race, att), ())
                                     for l in window)
            dfn_pit = dfn_pit or any(l in pit_laps.get((race, dfn), ())
                                     for l in window)
            att_done = last_lap.get((race, att), 0) <= end
            dfn_done = last_lap.get((race, dfn), 0) <= end
            finished = end >= race_len.get(race, 0) - 1

            if finished:
                reason = 'race_end'
            elif att_done or dfn_done:
                reason = 'retired'
            elif att_pit or dfn_pit:
                reason = 'pit'
            elif nxt in neutral_laps.get(race, ()):
                reason = 'neutralised'
            else:
                reason = 'dropped_back'

        rows.append({
            'Race': race, 'Round': int(g['Round'].iloc[0]),
            'attacker': att, 'defender': dfn,
            'attacking_team': g['attacking_team'].iloc[0],
            'defending_team': g['defending_team'].iloc[0],
            'start_lap': int(g['lap'].min()), 'end_lap': end,
            'laps_in_contact': int(len(g)),
            'mean_advantage': float(g['advantage'].mean()),
            'min_gap': float(g['gap'].min()),
            'passed': passed, 'reason': reason,
        })

    out = pd.DataFrame(rows)
    # the ambiguous case, both readings kept so neither is smuggled in
    out['censored_strict'] = (~out['passed']) & (out['reason'] != 'dropped_back')
    out['censored_loose'] = ~out['passed']
    return out


def hazard_by_lap(episodes, max_lap=4):
    """Pass probability on lap k of a fight, given the fight reached lap k."""
    out = []
    for t in range(1, max_lap + 1):
        at_risk = int((episodes['laps_in_contact'] >= t).sum())
        passes = int((episodes['passed']
                      & (episodes['laps_in_contact'] == t)).sum())
        out.append(passes / at_risk if at_risk else np.nan)
    return out


def stratified_hazard(episodes):
    """
    Whether the falling hazard is time or composition.

    The pooled hazard drops from 0.34 on first contact to 0.15 after, which
    reads like "the first lap is the best chance and it closes". It mostly is
    not. Easy fights - a big pace advantage, a circuit with a long straight -
    resolve immediately, so the pursuits still running on lap three are the
    hard ones by selection. Split by advantage and the drop largely goes
    away; in the bottom third it is not there at all.

    This matters because the two readings ask for different things. Real
    time-dependence would want a first-lap bonus in the lap loop. Composition
    wants what simulate.py already has: a pass probability conditioned on
    pace advantage.
    """
    q = episodes['mean_advantage'].quantile([1 / 3, 2 / 3]).tolist()
    lo = episodes[episodes['mean_advantage'] < q[0]]
    mid = episodes[(episodes['mean_advantage'] >= q[0])
                   & (episodes['mean_advantage'] < q[1])]
    hi = episodes[episodes['mean_advantage'] >= q[1]]
    return [('all', episodes), (f'advantage < {q[0]:.2f}', lo),
            ('advantage middle', mid), (f'advantage > {q[1]:.2f}', hi)]


def kaplan_meier(durations, events):
    """
    Survival of "still stuck behind", against laps spent in contact.

    Censored pursuits stay in the risk set until the lap they end on and then
    leave without counting as failures, which is the whole point: a fight
    broken up by a pit stop is not evidence that the car could not pass.
    """
    d = np.asarray(durations, float)
    e = np.asarray(events, bool)
    if len(d) == 0:
        return pd.DataFrame(columns=['lap', 'at_risk', 'passes', 'hazard',
                                     'survival'])

    rows, surv = [], 1.0
    for t in np.unique(d):
        at_risk = int((d >= t).sum())
        passes = int((e & (d == t)).sum())
        if at_risk == 0:
            continue
        h = passes / at_risk
        surv *= (1 - h)
        rows.append({'lap': int(t), 'at_risk': at_risk, 'passes': passes,
                     'hazard': h, 'survival': surv})
    return pd.DataFrame(rows)


# --- part 2b: attack and defence coefficients -------------------------------


def logistic_ridge(x, y, ridge=RIDGE, iters=60):
    """
    Penalised logistic regression by IRLS, in numpy.

    scipy's optimisers are blocked in this environment, and this is a convex
    problem with a closed-form Newton step, so it does not need them. The
    intercept is never penalised.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, k = x.shape
    beta = np.zeros(k)
    pen = ridge * np.eye(k)
    pen[0, 0] = 0.0

    for _ in range(iters):
        z = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = x.T @ (y - p) - pen @ beta
        hess = (x * w[:, None]).T @ x + pen
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def fit_attack_defence(episodes, teams):
    """
    One coefficient per team for getting past, one for holding position.

    A raw pair rate answers "how often did McLaren get past Haas", which on
    2026 data is three fights. This asks the bigger question - how often does
    any car get past, given who was attacking, who was defending, how much
    quicker the attacker was and which circuit it happened at - and reads the
    team terms off that. Every pursuit informs every coefficient, so a team
    that met one opponent twice is still placed.

    Race dummies rather than an imported circuit rating: the 2022-25
    overtaking table describes different cars, and this way the circuit, the
    weather and the day's safety cars are all absorbed without borrowing any
    of it.
    """
    ep = episodes[episodes['reason'] != 'pit'].copy()
    ep = ep[ep['reason'] != 'retired']
    if len(ep) < 40:
        return None

    races = sorted(ep['Race'].unique())
    idx_t = {t: i for i, t in enumerate(teams)}
    idx_r = {r: i for i, r in enumerate(races)}

    n = len(ep)
    # intercept, advantage, attack (n-1), defence (n-1), race (n-1)
    na, nr = len(teams), len(races)
    x = np.zeros((n, 2 + (na - 1) * 2 + (nr - 1)))
    x[:, 0] = 1.0
    x[:, 1] = ep['mean_advantage'].to_numpy(float)

    for row, (_, r) in enumerate(ep.iterrows()):
        a, d = idx_t[r['attacking_team']], idx_t[r['defending_team']]
        # sum-to-zero coding: the last team is minus the sum of the others,
        # so no team is silently the reference everything is measured against
        if a < na - 1:
            x[row, 2 + a] = 1.0
        else:
            x[row, 2:2 + na - 1] = -1.0
        if d < na - 1:
            x[row, 2 + (na - 1) + d] = 1.0
        else:
            x[row, 2 + (na - 1):2 + 2 * (na - 1)] = -1.0
        ri = idx_r[r['Race']]
        if ri < nr - 1:
            x[row, 2 + 2 * (na - 1) + ri] = 1.0
        else:
            x[row, 2 + 2 * (na - 1):] = -1.0

    y = ep['passed'].to_numpy(float)
    beta = logistic_ridge(x, y)

    att = list(beta[2:2 + na - 1])
    att.append(-sum(att))
    dfn = list(beta[2 + (na - 1):2 + 2 * (na - 1)])
    dfn.append(-sum(dfn))

    # the raw coefficient sits on P(pass), so a team that is hard to get past
    # carries a negative one. Reported that way the column reads backwards -
    # Mercedes, the hardest car on the grid to pass, looks like the worst
    # defender. Negated here so positive means holds position, which is what
    # the word says.
    dfn = [-v for v in dfn]

    fitted = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    ll = float(np.sum(y * np.log(np.clip(fitted, 1e-9, 1))
                      + (1 - y) * np.log(np.clip(1 - fitted, 1e-9, 1))))
    base = float(y.mean())
    ll0 = float(np.sum(y * np.log(base) + (1 - y) * np.log(1 - base)))

    return {
        'teams': list(teams),
        'attack': np.array(att), 'defence': np.array(dfn),
        'advantage_coef': float(beta[1]), 'intercept': float(beta[0]),
        'n': n, 'pseudo_r2': 1 - ll / ll0 if ll0 != 0 else np.nan,
        'episodes': ep,
        'design': x, 'y': y,
    }


def permutation_test(fit, teams, n_draws=200, seed=42):
    """
    How much of the attack/defence spread is just eleven teams and a few
    hundred fights.

    The team labels are shuffled within each race and the spread of the
    coefficients re-measured. If the real spread sits inside that cloud, the
    numbers are describing the sample, not the cars - the same check that
    disqualified the circuit-similarity pooling in v1.4.
    """
    ep = fit['episodes']
    rng = np.random.default_rng(seed)
    real = float(np.std(fit['attack'])), float(np.std(fit['defence']))

    null_a, null_d = [], []
    for _ in range(n_draws):
        sh = ep.copy()
        for col in ('attacking_team', 'defending_team'):
            sh[col] = (sh.groupby('Race')[col]
                       .transform(lambda s: rng.permutation(s.to_numpy())))
        f = fit_attack_defence(sh, teams)
        if f is None:
            continue
        null_a.append(float(np.std(f['attack'])))
        null_d.append(float(np.std(f['defence'])))

    null_a, null_d = np.array(null_a), np.array(null_d)
    return {
        'attack_real': real[0], 'defence_real': real[1],
        'attack_p': float((null_a >= real[0]).mean()),
        'defence_p': float((null_d >= real[1]).mean()),
        'attack_null_mean': float(null_a.mean()),
        'defence_null_mean': float(null_d.mean()),
        'n_draws': len(null_a),
    }


# --- part 3: speed trap -----------------------------------------------------


def fetch_speed_trap():
    """
    SpeedST per car per session, qualifying and race.

    session.laps carries the trap already, so this stays off the telemetry
    path entirely and runs in seconds against a warm cache.
    """
    import logging
    import fastf1
    # FastF1 logs a dozen lines per session at INFO; twenty-five sessions of
    # that buries the summary this module exists to print
    logging.getLogger('fastf1').setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith('fastf1'):
            logging.getLogger(name).setLevel(logging.ERROR)

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    schedule = fastf1.get_event_schedule(SEASON, include_testing=False)
    now = pd.Timestamp.now()

    rows = []
    for _, event in schedule.iterrows():
        name, rnd = event['EventName'], int(event['RoundNumber'])
        if pd.Timestamp(event['EventDate']) > now:
            continue
        is_target = any(a in f"{name} {event.get('Country', '')}".lower()
                        for a in TRACK_ALIASES)

        for kind in ('Q', 'R'):
            # the target's race has not been run from this project's point of
            # view - its qualifying has, and qualifying precedes the race
            if is_target and kind == 'R':
                continue
            try:
                s = fastf1.get_session(SEASON, name, kind)
                s.load(telemetry=False, weather=False, messages=False)
            except Exception as exc:
                print(f'    [{kind}] {name}: {type(exc).__name__}')
                continue

            laps = s.laps
            if laps is None or laps.empty or 'SpeedST' not in laps.columns:
                continue
            d = laps[['Driver', 'Team', 'LapNumber', 'SpeedST']].copy()
            d['SpeedST'] = pd.to_numeric(d['SpeedST'], errors='coerce')
            d = d[d['SpeedST'].notna() & (d['SpeedST'] > 50)]
            if d.empty:
                continue
            d['Race'], d['Round'] = name, rnd
            d['session'] = 'quali' if kind == 'Q' else 'race'
            d['is_target'] = is_target
            rows.append(d)
            print(f'    [{kind}] {name:32s} {len(d):4d} trap readings')

    if not rows:
        raise SystemExit('no speed trap data collected')
    return pd.concat(rows, ignore_index=True)


def summarise_speed_trap(raw):
    """
    Per team per race, both channels, z-scored inside each session.

    The absolute number is meaningless across circuits - Monza's trap sits
    fifty km/h above Hungary's for reasons that have nothing to do with the
    cars - so what is kept is the deviation from that session's own field.
    """
    d = raw.copy()
    early = (d['session'] == 'race') & (d['LapNumber'] <= SPEED_TRAP_EARLY_LAPS)
    d = d[(d['session'] == 'quali') | early]

    g = (d.groupby(['session', 'Race', 'Round', 'Team', 'is_target'])
         .agg(speed_trap_median=('SpeedST', 'median'),
              n_laps=('SpeedST', 'size')).reset_index())
    g = g[g['n_laps'] >= MIN_TRAP_LAPS]

    key = ['session', 'Race']
    mean = g.groupby(key)['speed_trap_median'].transform('mean')
    sd = g.groupby(key)['speed_trap_median'].transform('std')
    g['speed_trap_z'] = (g['speed_trap_median'] - mean) / sd.replace(0, np.nan)
    return g


def impact_preview(fit, perm, base_rate):
    """
    What the coefficients would do to the lap loop, without touching it.

    simulate.py has no team term at all: pass_probability() reads a pace
    advantage and a per-circuit threshold, so two cars a tenth apart pass at
    the same rate whoever is driving them. Adding a team term is the obvious
    use for these numbers, and this says how hard it would hit before anyone
    tries it.

    Raw coefficients are too strong to believe. The permutation test gives
    the spread shuffled labels produce by themselves, and the difference
    between that and the observed spread is the part worth keeping:

        signal variance = observed variance - noise variance
        shrinkage       = signal variance / observed variance

    which is the same empirical-Bayes step the affinity shrinkage came from,
    asked of a spread instead of a regression slope.
    """
    obs = float(np.std(fit['attack']))
    noise = perm['attack_null_mean']
    signal_var = max(obs ** 2 - noise ** 2, 0.0)
    shrink = signal_var / obs ** 2 if obs > 0 else 0.0

    logit0 = np.log(base_rate / (1 - base_rate))
    rows = []
    for i, team in enumerate(fit['teams']):
        raw = fit['attack'][i]
        shrunk = raw * shrink
        rows.append({
            'Team': team,
            'attack_raw': raw, 'attack_shrunk': shrunk,
            'p_raw': 1 / (1 + np.exp(-(logit0 + raw))),
            'p_shrunk': 1 / (1 + np.exp(-(logit0 + shrunk))),
        })
    df = pd.DataFrame(rows).sort_values('attack_raw', ascending=False)
    return df, shrink


def trap_vs_track_character(trap):
    """
    Whether straight-line speed is a property of the car or of the day.

    A team with a genuinely low-drag car should read high at every circuit.
    One that simply runs a skinny wing when the track asks for it should
    swing with the circuit - high at Monza, low at Budapest. The two are
    different things and only the first is a stable car parameter worth
    carrying into a simulation.

    track_features.py already places circuits on PC1 and PC2, so the question
    has a ready answer: regress each team's per-race trap z-score on the
    circuit's PC2 - its straight-line character - and see whether the slope
    is anything. A flat slope means the trap number travels.
    """
    try:
        from Simülasyon.data_prep.team_affinity import pc_coordinates, canonical
    except Exception:
        return None

    feat_path = os.path.join(DATA_DIR, 'track_features.csv')
    if not os.path.exists(feat_path):
        return None
    pcs = pc_coordinates(pd.read_csv(feat_path))

    q = trap[trap['session'] == 'quali'].copy()
    q['Race'] = q['Race'].map(canonical)
    m = q.merge(pcs, on='Race', how='inner')
    if len(m) < 40:
        return None

    rows = []
    for team, g in m.groupby('Team'):
        if len(g) < 6:
            continue
        b = np.polyfit(g['pc2'].to_numpy(float),
                       g['speed_trap_z'].to_numpy(float), 1)
        pred = np.polyval(b, g['pc2'].to_numpy(float))
        y = g['speed_trap_z'].to_numpy(float)
        ss = ((y - y.mean()) ** 2).sum()
        rows.append({
            'Team': team, 'level': float(y.mean()),
            'pc2_slope': float(b[0]),
            'r2': float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else np.nan,
            'within_sd': float(y.std()), 'n': len(g),
        })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    between = float(df['level'].std())
    within = float(df['within_sd'].mean())
    return {'table': df.sort_values('level', ascending=False),
            'between_team_sd': between, 'within_team_sd': within,
            'n_matched': len(m)}


# --- part 4: the two-axis team map ------------------------------------------


def team_profile(fit, trap, attacking, defending):
    """
    The overtaking equivalent of the track character map.

    track_features.py puts circuits on two axes because four correlated
    measurements are really two questions. The same is true here: attacking
    ability, defending ability and straight-line speed are not independent,
    and what the axes turn out to mean is a finding rather than a setting.
    """
    rows = []
    for i, team in enumerate(fit['teams']):
        q = trap[(trap['session'] == 'quali') & (trap['Team'] == team)]
        r = trap[(trap['session'] == 'race') & (trap['Team'] == team)]
        a = attacking[attacking['Team'] == team]
        d = defending[defending['Team'] == team]
        rows.append({
            'Team': team,
            'attack': float(fit['attack'][i]),
            'defence': float(fit['defence'][i]),
            'trap_quali_z': float(q['speed_trap_z'].mean()) if len(q) else np.nan,
            'trap_race_z': float(r['speed_trap_z'].mean()) if len(r) else np.nan,
            'n_attacks': int(a['n_attempts'].iloc[0]) if len(a) else 0,
            'n_defences': int(d['n_exposed'].iloc[0]) if len(d) else 0,
        })
    df = pd.DataFrame(rows)

    feats = ['attack', 'defence', 'trap_quali_z']
    usable = df.dropna(subset=feats)
    if len(usable) < 4:
        return df, None

    x = usable[feats].to_numpy(float)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    x = x - x.mean(0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T

    # sign pinned so the axes read the same way between runs
    if vt[0, feats.index('attack')] < 0:
        coords[:, 0], vt[0] = -coords[:, 0], -vt[0]
    if vt[1, feats.index('trap_quali_z')] < 0:
        coords[:, 1], vt[1] = -coords[:, 1], -vt[1]

    df.loc[usable.index, 'PC1'] = coords[:, 0]
    df.loc[usable.index, 'PC2'] = coords[:, 1]
    share = s ** 2 / (s ** 2).sum()
    return df, {'share': share, 'loadings': vt[:2], 'features': feats}


# --- reporting --------------------------------------------------------------


def head_tail(df, col, n, fmt):
    top = df.nlargest(n, col)
    bot = df.nsmallest(n, col)
    return [fmt(r) for _, r in top.iterrows()], [fmt(r) for _, r in bot.iterrows()]


def main():
    print(f'===== v1.5 overtaking and straight-line speed, {SEASON} =====')
    print('MEASUREMENT ONLY - nothing here is connected to simulate.py.\n')

    laps = load_2026_laps()
    teams_of = driver_teams(laps)
    teams = sorted(set(teams_of.values()))
    print(f'{laps["Race"].nunique()} races, {len(laps):,} laps, '
          f'{len(teams)} teams  (target {TARGET_EVENT} withheld)\n')

    print('--- detecting close laps (overtaking.py, unchanged) ---')
    events = collect_2026_events(laps)
    events['attacking_team'] = events['attacker'].map(teams_of)
    events['defending_team'] = events['defender'].map(teams_of)
    chances = events[events['advantage'] >= MIN_ADVANTAGE].copy()

    print(f'\n  close laps {len(events):,}   '
          f'with a pace advantage >= {MIN_ADVANTAGE} s/lap: {len(chances):,}')

    # ---- part 1, the directive's tables ------------------------------------
    pairs = pair_table(chances)
    attacking = side_table(chances, 'attacking')
    defending = side_table(chances, 'defending')

    print(f'\n===== 1. per-lap tables (as specified) =====')
    print(f'  team-pair cells: {len(pairs)}   '
          f'below {MIN_ATTEMPTS} attempts: {int(pairs["low_confidence"].sum())} '
          f'({pairs["low_confidence"].mean():.0%})')
    print(f'  lap-level pass rate across everything: '
          f'{chances["passed"].mean():.3f}')

    # ---- part 2, pursuits --------------------------------------------------
    episodes = build_episodes(chances, laps)
    print(f'\n===== 2. the same events as pursuits =====')
    print(f'  {len(chances):,} close laps collapse into {len(episodes):,} '
          f'pursuits ({len(chances) / max(len(episodes), 1):.2f} laps each)')
    print(f'  per-lap rate  {chances["passed"].mean():.3f}      '
          f'per-pursuit  {episodes["passed"].mean():.3f}')

    print('\n  how pursuits ended:')
    for reason, n in episodes['reason'].value_counts().items():
        print(f'    {reason:14s} {n:5d}  {n / len(episodes):6.1%}')

    strict = episodes[~episodes['censored_strict']]
    loose = episodes[~episodes['censored_loose'] | episodes['passed']]
    print(f'\n  the ambiguous case, both ways:')
    print(f'    dropping back counts as a failed pass : '
          f'{episodes["passed"].mean():.3f}')
    print(f'    dropping back counts as censoring     : '
          f'{strict["passed"].mean():.3f}')
    print(f'    the truth is between those two.')

    km = kaplan_meier(episodes['laps_in_contact'], episodes['passed'])
    print(f'\n  chance of getting past, by lap of the fight:')
    for _, r in km.head(8).iterrows():
        print(f'    lap {r["lap"]:2.0f} of contact   at risk {r["at_risk"]:5.0f}   '
              f'passes {r["passes"]:4.0f}   hazard {r["hazard"]:.3f}   '
              f'still stuck {r["survival"]:.3f}')
    if len(km) > 1:
        first = km['hazard'].iloc[0]
        later = km[km['lap'] > 1]['passes'].sum() / max(
            km[km['lap'] > 1]['at_risk'].sum(), 1)
        print(f'\n    pooled: {first:.3f} on first contact against {later:.3f} '
              f'after it.')

    print('\n  is that a real first-lap effect, or just the easy fights '
          'leaving early?')
    print(f'    {"stratum":24s}{"n":>5s}{"lap1":>8s}{"lap2":>8s}'
          f'{"lap3":>8s}{"lap4":>8s}')
    for name, d in stratified_hazard(episodes):
        h = hazard_by_lap(d)
        cells = ''.join(f'{v:8.3f}' if np.isfinite(v) else f'{"-":>8s}'
                        for v in h)
        print(f'    {name:24s}{len(d):5d}{cells}')
    print('    Split by pace advantage the fall mostly goes, and in the')
    print('    bottom third it is not there at all. So this is composition,')
    print('    not the first lap being special - which is an argument for')
    print('    conditioning on advantage, which simulate.py already does,')
    print('    and not for adding a first-lap bonus.')

    obs_first = float((episodes.loc[episodes['passed'],
                                    'laps_in_contact'] == 1).mean())
    print(f'\n    check: {obs_first:.0%} of passes land on the first lap of')
    print(f'    contact. A flat per-lap probability of '
          f'{chances["passed"].mean():.3f} over the observed fight lengths')
    print(f'    puts that figure at roughly the same place, so the timing the')
    print(f'    simulation produces is not the thing that is wrong here.')

    # ---- attack and defence ------------------------------------------------
    fit = fit_attack_defence(episodes, teams)
    perm = None
    if fit is None:
        print('\n  too few pursuits to fit attack/defence coefficients.')
    else:
        print(f'\n===== 3. attack and defence, controlled =====')
        print(f'  {fit["n"]} pursuits, pseudo-R2 {fit["pseudo_r2"]:.3f}, '
              f'advantage coefficient {fit["advantage_coef"]:+.2f} per s/lap')
        order = np.argsort(-fit['attack'])
        print(f'\n  {"team":18s}{"attack":>9s}{"defence":>9s}'
              f'{"fights att":>12s}{"fights def":>12s}')
        for i in order:
            t = fit['teams'][i]
            na = int((episodes['attacking_team'] == t).sum())
            nd = int((episodes['defending_team'] == t).sum())
            print(f'  {t:18s}{fit["attack"][i]:+9.2f}{fit["defence"][i]:+9.2f}'
                  f'{na:12d}{nd:12d}')

        perm = permutation_test(fit, teams)
        print(f'\n  is that spread real? labels shuffled within each race, '
              f'{perm["n_draws"]} draws:')
        print(f'    attack  spread {perm["attack_real"]:.3f} against '
              f'chance {perm["attack_null_mean"]:.3f}   p = {perm["attack_p"]:.3f}')
        print(f'    defence spread {perm["defence_real"]:.3f} against '
              f'chance {perm["defence_null_mean"]:.3f}   p = {perm["defence_p"]:.3f}')

    # ---- part 3, speed trap ------------------------------------------------
    trap_path = os.path.join(DATA_DIR, f'speed_trap_{SEASON}.csv')
    if FETCH_SPEED_TRAP or not os.path.exists(trap_path):
        print(f'\n--- speed trap (session.laps only, no telemetry) ---')
        trap = summarise_speed_trap(fetch_speed_trap())
        trap.to_csv(trap_path, index=False, encoding='utf-8')
    else:
        trap = pd.read_csv(trap_path)

    print(f'\n===== 4. speed trap =====')
    q = trap[trap['session'] == 'quali']
    r = trap[trap['session'] == 'race']
    print(f'  qualifying: {q["Race"].nunique()} sessions, {len(q)} team-races')
    print(f'  race, first {SPEED_TRAP_EARLY_LAPS} laps: '
          f'{r["Race"].nunique()} races, {len(r)} team-races')

    season_q = q.groupby('Team')['speed_trap_z'].mean().sort_values(ascending=False)
    season_r = r.groupby('Team')['speed_trap_z'].mean()
    both = pd.DataFrame({'quali': season_q, 'race': season_r}).dropna()
    if len(both) > 3:
        rho = float(np.corrcoef(both['quali'], both['race'])[0, 1])
        print(f'\n  the two channels agree at corr {rho:+.3f} - '
              f'{"the contamination is survivable" if rho > 0.6 else "they are measuring different things"}')

    print(f'\n  {"team":18s}{"quali z":>10s}{"race z":>10s}')
    for t, v in season_q.items():
        rv = season_r.get(t, np.nan)
        print(f'  {t:18s}{v:+10.2f}{rv:+10.2f}')

    # ---- is the trap a car property or a wing choice ------------------------
    stab = trap_vs_track_character(trap)
    if stab is not None:
        print(f'\n===== 5. is straight-line speed a car or a wing setting? =====')
        print(f'  between teams {stab["between_team_sd"]:.3f} sd, '
              f'within a team across circuits {stab["within_team_sd"]:.3f} sd')
        share = stab['between_team_sd'] ** 2 / (stab['between_team_sd'] ** 2
                                                + stab['within_team_sd'] ** 2)
        print(f'  so {share:.0%} of the variation is the car and the rest is '
              f'the circuit.')
        print(f'\n  {"team":18s}{"level":>8s}{"PC2 slope":>11s}{"R2":>7s}{"n":>4s}')
        for _, r in stab['table'].iterrows():
            print(f'  {r["Team"]:18s}{r["level"]:+8.2f}{r["pc2_slope"]:+11.2f}'
                  f'{r["r2"]:7.2f}{int(r["n"]):4d}')
        big = stab['table'].reindex(
            stab['table']['pc2_slope'].abs().sort_values(ascending=False).index)
        t = big.iloc[0]
        print(f'\n  steepest: {t["Team"]} at {t["pc2_slope"]:+.2f} z per PC2 unit -')
        print(f'  {"it trims wing for the fast circuits" if abs(t["pc2_slope"]) > 0.2 else "nobody swings much; the trap travels"}.')

    # ---- what it would do to the lap loop -----------------------------------
    if fit is not None and perm is not None:
        base = float(chances['passed'].mean())
        prev, shrink = impact_preview(fit, perm, base)
        print(f'\n===== 6. what this would do if it were wired in =====')
        print(f'  simulate.py has no team term: pass_probability() reads a pace')
        print(f'  advantage and a circuit threshold, nothing else. Adding the')
        print(f'  attack coefficient to that threshold, at a {base:.3f} baseline:')
        print(f'\n  observed spread {np.std(fit["attack"]):.3f}, '
              f'shuffled labels give {perm["attack_null_mean"]:.3f} on their own')
        print(f'  -> keep {shrink:.0%} of it, discard the rest as sample noise')
        print(f'\n  {"team":18s}{"raw":>8s}{"kept":>8s}{"p raw":>9s}{"p kept":>9s}')
        for _, r in prev.iterrows():
            print(f'  {r["Team"]:18s}{r["attack_raw"]:+8.2f}'
                  f'{r["attack_shrunk"]:+8.2f}{r["p_raw"]:9.3f}'
                  f'{r["p_shrunk"]:9.3f}')
        rr = prev['p_raw'].max() / prev['p_raw'].min()
        rs = prev['p_shrunk'].max() / prev['p_shrunk'].min()
        print(f'\n  best-to-worst per-lap pass chance: {rr:.1f}x raw, '
              f'{rs:.1f}x after shrinking.')
        print(f'  {rr:.1f}x is not a credible spread between two 2026 cars at')
        print(f'  equal pace. That is the number the permutation test is warning')
        print(f'  about, and it is why nothing is wired in from here.')

    # ---- part 4, the map ---------------------------------------------------
    profile, pca = team_profile(fit, trap, attacking, defending) if fit else (None, None)
    if pca is not None:
        print(f'\n===== 7. the two-axis team map =====')
        print(f'  PC1 {pca["share"][0]:.0%} of the variance, '
              f'PC2 {pca["share"][1]:.0%}')
        for i in (0, 1):
            load = ', '.join(f'{f} {pca["loadings"][i, j]:+.2f}'
                             for j, f in enumerate(pca['features']))
            print(f'  PC{i + 1}: {load}')
        print(f'\n  {"team":18s}{"PC1":>8s}{"PC2":>8s}{"attack":>9s}'
              f'{"defence":>9s}{"trap Q":>9s}')
        for _, row in profile.sort_values('PC1', ascending=False).iterrows():
            print(f'  {row["Team"]:18s}{row.get("PC1", np.nan):+8.2f}'
                  f'{row.get("PC2", np.nan):+8.2f}{row["attack"]:+9.2f}'
                  f'{row["defence"]:+9.2f}{row["trap_quali_z"]:+9.2f}')

    # ---- output ------------------------------------------------------------
    pairs.to_csv(os.path.join(DATA_DIR, f'overtake_pairs_{SEASON}.csv'),
                 index=False, encoding='utf-8')
    attacking.to_csv(os.path.join(DATA_DIR, f'overtake_attacking_{SEASON}.csv'),
                     index=False, encoding='utf-8')
    defending.to_csv(os.path.join(DATA_DIR, f'overtake_defending_{SEASON}.csv'),
                     index=False, encoding='utf-8')
    episodes.to_csv(os.path.join(DATA_DIR, f'overtake_episodes_{SEASON}.csv'),
                    index=False, encoding='utf-8')
    if profile is not None:
        profile['version'] = PARAMS_VERSION
        profile.round(4).to_csv(
            os.path.join(DATA_DIR, f'overtake_team_profile_{SEASON}.csv'),
            index=False, encoding='utf-8')

    print(f'\n===== written =====')
    for name in (f'overtake_pairs_{SEASON}.csv',
                 f'overtake_attacking_{SEASON}.csv',
                 f'overtake_defending_{SEASON}.csv',
                 f'overtake_episodes_{SEASON}.csv',
                 f'speed_trap_{SEASON}.csv',
                 f'overtake_team_profile_{SEASON}.csv'):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            print(f'  {name:36s} {os.path.getsize(p):>9,} b')

    print('\nNothing was connected to the simulation. Read the tables first.')
    return pairs, episodes, trap, profile


if __name__ == '__main__':
    main()
