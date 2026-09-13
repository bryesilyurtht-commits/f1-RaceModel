"""
F1 Prediction Simulation - v2.6 - backtest.py
Scoring a prediction against what actually happened.

This module is the measuring instrument, not the experiment. It holds the race
universe, the ground truth, the naive baselines the model has to beat, and the
scores - and it knows nothing about how a prediction was produced. Anything
that returns per-driver probabilities can be handed to it.

What a prediction is
--------------------
A `Prediction`: a driver list, a finishing-position distribution, and the three
event probabilities read off it. The distribution is the primary object; win,
podium and points are views of it, and are stored rather than recomputed
because a baseline may have probabilities without a full distribution.

Why two kinds of position error
-------------------------------
The expected position of a driver whose distribution is bimodal - wins or
retires - is a number they will almost never finish in. Ranking the drivers by
expected position and scoring that rank is a different measurement, and the two
disagree most exactly where the model is least certain. Both are reported, and
each is labelled, so that neither can be quoted as "the MAE".

Why the ranked probability score
--------------------------------
Position error treats a prediction of P2 as equally wrong whether the model
gave P2 60% or 2%. RPS scores the whole distribution: at every position
threshold it compares the predicted "this position or better" against what
happened, so a confident miss costs more than a hedged one. It is the score
that separates an overconfident model from a calibrated one, which is the
question v2.6 exists to answer.

What this module does not do
----------------------------
It does not select races, tune anything, or decide what counts as success. It
reports the numbers and the sample they came from.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RESULTS = 'results_2018_2025.csv'

# The scope, fixed before any result was looked at.
#
# The model is built for the 18-inch regulations. 2018-2021 ran a different
# tyre, and the degradation curve - which is most of the model's per-circuit
# character - does not transfer across that boundary. Those seasons stay in the
# archive as training history for everything that is not a tyre curve, and are
# not scored. Excluding them is a statement about the model's domain, made in
# advance; excluding a race after seeing how it scored would not be.
ERA_FIRST_SEASON = 2022

# A target race needs enough same-era history behind it for the profiles to
# mean anything. Two full seasons is the floor.
MIN_TRAINING_RACES = 40

POINTS_PLACES = 10      # verified against the Points column, all 8 seasons


# --- the universe -----------------------------------------------------------

def load_results(path=None):
    """Every classified finish 2018-2025, with position parsed to a number."""
    frame = pd.read_csv(path or os.path.join(DATA_DIR, RESULTS))
    frame['finish_pos'] = pd.to_numeric(frame['ClassifiedPosition'],
                                        errors='coerce')
    frame['grid_pos'] = pd.to_numeric(frame['GridPosition'], errors='coerce')
    return frame


def universe(results=None):
    """
    Every race in the archive, in order, with why it is in or out.

    The reasons are computed from the archive alone. Nothing here has seen a
    prediction, so a race cannot be dropped for having scored badly.
    """
    results = load_results() if results is None else results
    rows = []
    for (season, rnd), group in results.groupby(['Season', 'RoundNumber']):
        name = str(group['Race'].iloc[0])
        starters = int(group['grid_pos'].notna().sum())
        classified = int(group['finish_pos'].notna().sum())

        earlier = history_before(results, season, rnd)
        era = earlier[earlier['Season'] >= ERA_FIRST_SEASON]
        n_training = era.groupby(['Season', 'RoundNumber']).ngroups

        if season < ERA_FIRST_SEASON:
            state, why = 'excluded', f'before {ERA_FIRST_SEASON}: 13-inch tyre era'
        elif n_training < MIN_TRAINING_RACES:
            state, why = 'excluded', (f'only {n_training} same-era races before '
                                      f'it, floor is {MIN_TRAINING_RACES}')
        elif starters < 10:
            state, why = 'excluded', f'only {starters} starters on record'
        else:
            state, why = 'included', f'{n_training} same-era races before it'

        rows.append(dict(Season=season, Round=rnd, Race=name, starters=starters,
                         classified=classified, training_races=n_training,
                         state=state, reason=why))
    return pd.DataFrame(rows).sort_values(['Season', 'Round']).reset_index(drop=True)


def history_before(results, season, rnd):
    """
    The archive as it stood before this race started.

    Strictly before: the target race's own rows are never in here. This is the
    single place the walk-forward rule is enforced, so there is one line to
    check rather than one per feature.
    """
    return results[(results['Season'] < season)
                   | ((results['Season'] == season)
                      & (results['RoundNumber'] < rnd))].copy()


def outcome(results, season, rnd):
    """
    What happened, per driver.

    Everyone who took the start is scored, retirements included. A model that
    predicts a full finishing order is answerable for the whole order; dropping
    the cars that broke would score it on the easy half of its own claim.

    Points is position <= 10, which was checked against the Points column
    rather than assumed: across all 3438 rows of the archive, scoring a point
    and finishing in the top ten are the same event, with no exceptions.
    """
    race = results[(results['Season'] == season)
                   & (results['RoundNumber'] == rnd)].copy()
    race = race[race['grid_pos'].notna()]

    # A car that started and was not classified finished behind every car that
    # was. Ranking them among themselves would invent a result, so they share
    # the position after the last classified car.
    classified = race['finish_pos'].notna()
    last = race.loc[classified, 'finish_pos'].max() if classified.any() else 0
    race['actual_pos'] = race['finish_pos']
    race.loc[~classified, 'actual_pos'] = last + 1

    race['won'] = (race['actual_pos'] == 1) & classified
    race['podium'] = (race['actual_pos'] <= 3) & classified
    race['points'] = (race['actual_pos'] <= POINTS_PLACES) & classified
    race['finished'] = classified
    return race[['Driver', 'Team', 'grid_pos', 'actual_pos', 'won', 'podium',
                 'points', 'finished', 'Status']].reset_index(drop=True)


# --- the prediction object --------------------------------------------------

class Prediction:
    """
    Per-driver probabilities for one race, from any source.

    `dist` is optional: a baseline that only offers a ranking has no position
    distribution, and asking it for one would mean inventing it. Scores that
    need a distribution skip such a prediction rather than scoring a fabricated
    one, and report how many they skipped.
    """

    def __init__(self, drivers, mean_pos, p_win, p_podium, p_points,
                 dist=None, label='model'):
        self.drivers = list(drivers)
        self.mean_pos = np.asarray(mean_pos, dtype=float)
        self.p_win = np.asarray(p_win, dtype=float)
        self.p_podium = np.asarray(p_podium, dtype=float)
        self.p_points = np.asarray(p_points, dtype=float)
        self.dist = None if dist is None else np.asarray(dist, dtype=float)
        self.label = label

    @property
    def rank(self):
        """The single predicted order, 1..N, by expected position."""
        order = np.argsort(self.mean_pos, kind='stable')
        out = np.empty(len(self.drivers), dtype=float)
        out[order] = np.arange(1, len(self.drivers) + 1)
        return out

    def frame(self):
        return pd.DataFrame({'Driver': self.drivers, 'mean_pos': self.mean_pos,
                             'rank_pos': self.rank, 'P_win': self.p_win,
                             'P_podium': self.p_podium,
                             'P_points': self.p_points})

    @classmethod
    def from_simulation(cls, result, label='model'):
        """
        Read a Prediction out of what simulate.run() returns.

        The column order of `positions` is the order of `dnf_by_driver`, not of
        `summary` - summary is sorted by win probability. Lining the two up by
        row would silently attribute every distribution to the wrong driver, so
        the driver name is carried through instead.
        """
        order = list(result['dnf_by_driver']['Driver'])
        positions = np.asarray(result['positions'])
        n_sims, n_drivers = positions.shape

        dist = np.zeros((n_drivers, n_drivers))
        for column in range(n_drivers):
            counts = np.bincount(positions[:, column].astype(int),
                                 minlength=n_drivers + 1)[1:n_drivers + 1]
            dist[column] = counts / n_sims

        summary = result['summary'].set_index('Driver')
        keep = [d for d in order if d in summary.index]
        index = [order.index(d) for d in keep]
        return cls(keep, summary.loc[keep, 'mean_pos'].to_numpy(),
                   summary.loc[keep, 'P_win'].to_numpy(),
                   summary.loc[keep, 'P_podium'].to_numpy(),
                   summary.loc[keep, 'P_points'].to_numpy(),
                   dist=dist[index], label=label)


# --- naive baselines --------------------------------------------------------

def grid_baseline(entries):
    """
    The grid, unchanged, as the predicted finishing order.

    The reference every position score is read against. It carries no
    probabilities: a grid slot is not a claim about how often anything happens,
    and giving it one would mean inventing a baseline rather than using the
    obvious one.
    """
    drivers = list(entries['Driver'])
    grid = entries['grid_pos'].to_numpy(dtype=float)
    nan = np.full(len(drivers), np.nan)
    return Prediction(drivers, grid, nan, nan, nan, label='grid order')


def grid_rate_baseline(history, entries, prior_weight=30.0):
    """
    What a car starting in this slot has historically done, and nothing else.

    The honest opponent for a probability, which the grid order cannot be. It
    is built only from races before the target, so it grows with the archive
    and never sees the race it is predicting.

    A slot with four observations is shrunk hard toward the field-wide rate:
    P2 winning twice in three early races is not a 67% win chance, and
    `prior_weight` is the number of pseudo-observations of the pooled rate each
    slot carries before its own record counts for anything.
    """
    past = history[history['grid_pos'].notna()].copy()
    past['finish_pos'] = pd.to_numeric(past['ClassifiedPosition'],
                                       errors='coerce')
    classified = past['finish_pos'].notna()
    past['won'] = (past['finish_pos'] == 1) & classified
    past['podium'] = (past['finish_pos'] <= 3) & classified
    past['points'] = (past['finish_pos'] <= POINTS_PLACES) & classified
    past['slot'] = past['grid_pos'].round().astype(int)

    events = ('won', 'podium', 'points')
    pooled = {e: float(past[e].mean()) for e in events}

    # An unclassified car is scored at the back, the same convention outcome()
    # uses, so the baseline's expected position and the truth mean the same
    # thing by the same rule.
    back = past['finish_pos'].max()
    past['filled_pos'] = past['finish_pos'].fillna(back)
    pooled_pos = float(past['filled_pos'].mean())

    rates, mean_pos = {}, {}
    for slot, group in past.groupby('slot'):
        n = len(group)
        rates[slot] = {e: (group[e].sum() + prior_weight * pooled[e])
                          / (n + prior_weight) for e in events}
        mean_pos[slot] = float((group['filled_pos'].sum()
                                + prior_weight * pooled_pos) / (n + prior_weight))

    drivers = list(entries['Driver'])
    win, pod, pts, pos = [], [], [], []
    for slot in entries['grid_pos']:
        key = int(round(slot)) if pd.notna(slot) else None
        rate = rates.get(key)
        win.append(pooled['won'] if rate is None else rate['won'])
        pod.append(pooled['podium'] if rate is None else rate['podium'])
        pts.append(pooled['points'] if rate is None else rate['points'])
        pos.append(mean_pos.get(key, pooled_pos))
    return Prediction(drivers, pos, win, pod, pts, label='grid history')


# --- scores -----------------------------------------------------------------

def position_errors(prediction, truth):
    """
    Both position errors, over the drivers the two sides agree exist.

    `expected` is the error of the distribution's mean, which is not a position
    anyone finishes in. `ranked` is the error of the single order those means
    imply. Both are reported because quoting whichever came out lower is the
    easiest way to flatter a model.
    """
    frame = prediction.frame().merge(truth[['Driver', 'actual_pos']],
                                     on='Driver', how='inner')
    if frame.empty:
        return None
    return {
        'n_drivers': len(frame),
        'mae_expected': float((frame['mean_pos'] - frame['actual_pos'])
                              .abs().mean()),
        'mae_ranked': float((frame['rank_pos'] - frame['actual_pos'])
                            .abs().mean()),
    }


def ranked_probability_score(prediction, truth):
    """
    Mean RPS over drivers, 0 for a distribution that was certain and right.

    Divided by (N-1) so a 20-car race and a 23-car race sit on the same scale.
    Field size changes across the archive, and an unnormalised RPS would make
    the larger fields look worse for nothing but being larger.
    """
    if prediction.dist is None:
        return None
    lookup = dict(zip(truth['Driver'], truth['actual_pos']))
    n = prediction.dist.shape[1]
    scores = []
    for i, driver in enumerate(prediction.drivers):
        if driver not in lookup or pd.isna(lookup[driver]):
            continue
        actual = int(lookup[driver])
        predicted_cdf = np.cumsum(prediction.dist[i])
        actual_cdf = (np.arange(1, n + 1) >= actual).astype(float)
        scores.append(float(((predicted_cdf - actual_cdf) ** 2).sum() / (n - 1)))
    return None if not scores else float(np.mean(scores))


def brier(probabilities, outcomes):
    """
    Mean squared error of a probability against a 0/1 event.

    Zero when every call was certain and right; 0.25 for every call at a half.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    keep = ~np.isnan(p)
    return None if not keep.any() else float(np.mean((p[keep] - y[keep]) ** 2))


def event_brier(prediction, truth):
    """Brier for win, podium and points, over the shared drivers."""
    frame = prediction.frame().merge(
        truth[['Driver', 'won', 'podium', 'points']], on='Driver', how='inner')
    if frame.empty:
        return None
    return {'win': brier(frame['P_win'], frame['won']),
            'podium': brier(frame['P_podium'], frame['podium']),
            'points': brier(frame['P_points'], frame['points'])}


def calibration(probabilities, outcomes, edges=(0, .05, .1, .2, .3, .5, .7, 1.01),
                min_count=20):
    """
    Predicted rate against observed rate, in bands.

    A band with nine observations says nothing, so bands below `min_count` are
    reported with their count and marked thin rather than being read as
    evidence. The drivers in one race are not independent - a safety car moves
    all of them - so a band's count overstates how much it knows, and the
    interval that matters is the race-level bootstrap, not this table.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    keep = ~np.isnan(p)
    p, y = p[keep], y[keep]

    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = (p >= low) & (p < high)
        n = int(band.sum())
        if n == 0:
            continue
        rows.append(dict(band=f'{low:.0%}-{min(high, 1):.0%}', n=n,
                         predicted=float(p[band].mean()),
                         observed=float(y[band].mean()),
                         thin=n < min_count))
    return pd.DataFrame(rows)


def bootstrap_difference(per_race, draws=10_000, seed=42, level=0.95):
    """
    An interval for the mean per-race difference, resampling whole races.

    Races are the independent unit, not drivers. Twenty drivers in one race
    share a safety car, a shower and a first-corner accident, so resampling
    drivers would treat one race as twenty observations and return an interval
    several times too narrow. The count of races is what the interval is about,
    and it is reported beside it.
    """
    values = np.asarray([v for v in per_race if v is not None and not np.isnan(v)],
                        dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    tail = (1 - level) / 2
    return {'n_races': int(len(values)), 'mean': float(values.mean()),
            'lo': float(np.quantile(means, tail)),
            'hi': float(np.quantile(means, 1 - tail)),
            'level': level, 'draws': draws}


# --- walking the evaluation set ---------------------------------------------

def evaluate(predictors, results=None, races=None, progress=None):
    """
    Score every predictor on every included race, one race at a time.

    A predictor is a callable (history, entries, season, rnd) -> Prediction or
    None. It is handed the archive as it stood before the race and the entry
    list, and nothing else: there is no argument through which the result could
    reach it, which is a stronger guarantee than remembering not to look.

    A race a predictor fails on is recorded as a failure, not as a zero. A
    model that crashes on the hard races and is averaged over the rest is a
    model measured on a sample it chose.
    """
    results = load_results() if results is None else results
    if races is None:
        frame = universe(results)
        races = list(frame[frame.state == 'included'][['Season', 'Round']]
                     .itertuples(index=False, name=None))

    rows, failures = [], []
    for i, (season, rnd) in enumerate(races):
        if progress:
            progress(i, len(races), season, rnd)
        truth = outcome(results, season, rnd)
        history = history_before(results, season, rnd)
        entries = truth[['Driver', 'grid_pos']].copy()

        for name, predictor in predictors.items():
            try:
                prediction = predictor(history, entries, season, rnd)
            except Exception as exc:                     # noqa: BLE001
                failures.append(dict(Season=season, Round=rnd, predictor=name,
                                     error=f'{type(exc).__name__}: {exc}'))
                continue
            if prediction is None:
                failures.append(dict(Season=season, Round=rnd, predictor=name,
                                     error='predictor returned nothing'))
                continue

            errors = position_errors(prediction, truth)
            briers = event_brier(prediction, truth)
            merged = prediction.frame().merge(
                truth[['Driver', 'won', 'podium', 'points']], on='Driver')
            rows.append(dict(
                Season=season, Round=rnd, predictor=name,
                n_drivers=errors['n_drivers'],
                mae_expected=errors['mae_expected'],
                mae_ranked=errors['mae_ranked'],
                rps=ranked_probability_score(prediction, truth),
                brier_win=briers['win'], brier_podium=briers['podium'],
                brier_points=briers['points'],
                _probs=merged))
    return pd.DataFrame(rows), pd.DataFrame(failures)


def summarise(scores, metrics=('mae_expected', 'mae_ranked', 'rps',
                               'brier_win', 'brier_podium', 'brier_points')):
    """
    Each predictor's mean over races, equally weighted.

    Races are weighted equally rather than drivers, so a 20-car race and a
    19-car race count the same. Weighting by driver would let the larger fields
    quietly decide the average.
    """
    rows = []
    for name, group in scores.groupby('predictor'):
        row = {'predictor': name, 'races': len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            row[metric] = float(values.mean()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(scores, metric, predictor, against):
    """
    The per-race difference between two predictors, on the races both ran.

    Paired on the race, because race difficulty dominates: a wet race with two
    safety cars is worse for everything, and comparing unpaired means measures
    which predictor got the easier sample.
    """
    a = scores[scores.predictor == predictor].set_index(['Season', 'Round'])
    b = scores[scores.predictor == against].set_index(['Season', 'Round'])
    shared = a.index.intersection(b.index)
    if not len(shared):
        return None, []
    diff = (a.loc[shared, metric] - b.loc[shared, metric]).dropna()
    return bootstrap_difference(diff.to_numpy()), list(diff.index)


def pooled_calibration(scores, predictor, event):
    """Every driver-race this predictor called, in probability bands."""
    rows = [r for r in scores[scores.predictor == predictor]['_probs']]
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    # The predicted column and the outcome column are not named alike: the
    # event is "win", the thing that happened is "won".
    predicted = {'win': 'P_win', 'podium': 'P_podium', 'points': 'P_points'}[event]
    happened = {'win': 'won', 'podium': 'podium', 'points': 'points'}[event]
    return calibration(frame[predicted].to_numpy(),
                       frame[happened].astype(float).to_numpy())


BASELINES = {
    'grid order': lambda history, entries, season, rnd: grid_baseline(entries),
    'grid history': lambda history, entries, season, rnd:
        grid_rate_baseline(history, entries),
}
