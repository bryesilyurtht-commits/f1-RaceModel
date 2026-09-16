"""
F1 Prediction Simulation - representative.py
A typical race, and a typical race for each driver who ever wins one.

A distribution over ten thousand races says who is likely to win. It does not
say what a race looks like, and the two questions have different answers: the
favourite wins a third of the time, so two races in three are won by somebody
else, and none of those are visible in a table of probabilities.

So this picks one race per winner. For every driver who takes at least one of
the ten thousand, it finds the race among *their* wins that best stands for
them - the modal number of retirements and safety cars in the races they win,
then the most likely finishing order among those.

What this is not
----------------
Not a forecast, and not a set of alternative forecasts. Every race here already
happened inside the simulation, and the probability of each is on the Result
tab. A driver with three wins in ten thousand has a representative race, and it
is a race they win 0.03% of the time - the panel showing it has to say so, or
it reads as eleven equally plausible Sundays.

Why the trace can be rebuilt for any race
-----------------------------------------
The lap-by-lap arrays are kept for every simulation, not just the chosen one -
`trace_order`, `trace_pits` and `trace_laptime` are all (laps, sims, drivers).
run() slices one race out of them for its own representative pick, and this
slices a different one out of the same arrays. Nothing is re-simulated and
nothing is stored twice.
"""

import numpy as np

from Simülasyon.diagnostics import build_stints, find_representative


def winners(positions, drivers):
    """
    Every driver who wins at least one simulation, most wins first.

    The share is carried alongside the count because they answer different
    questions and only one of them belongs next to a race: a driver can have a
    representative win and a 0.03% chance of it.
    """
    positions = np.asarray(positions)
    n_sims = positions.shape[0]
    counts = (positions == 1).sum(axis=0)

    rows = [{'driver': str(drivers[i]), 'driver_idx': int(i),
             'wins': int(counts[i]), 'share': float(counts[i] / n_sims)}
            for i in np.flatnonzero(counts > 0)]
    rows.sort(key=lambda row: (-row['wins'], row['driver']))
    return rows


def races_won_by(positions, driver_idx):
    """The simulation indices this driver won."""
    return np.flatnonzero(np.asarray(positions)[:, driver_idx] == 1)


def trace_for(diag, sim, rebuild_compounds):
    """
    One race's lap-by-lap arrays, in the shape run() hands to the interface.

    `rebuild_compounds` is passed in rather than imported: it lives in
    simulate.py, which imports this module's neighbour, and reaching back into
    it from here would close a loop for the sake of one function.
    """
    retired_lap = diag['retired_lap'][sim] if 'retired_lap' in diag else None
    compounds = rebuild_compounds(diag, sim)
    return {
        'sim': int(sim),
        'order': diag['trace_order'][:, sim, :].astype(np.int32),
        'pits': diag['trace_pits'][:, sim, :],
        'neutral': diag['neutral'][sim],
        'retired_lap': retired_lap,
        'lap_times': diag['trace_laptime'][:, sim, :].astype(float),
        'compounds': compounds,
    }


def for_each_winner(result, rebuild_compounds, min_wins=1):
    """
    A representative race for every driver who wins one, ready to render.

    Computed once, when the run finishes, rather than per page view: the
    selection reads the whole position matrix per driver, and doing it again on
    every interaction would make the interface pay for a fixed answer.

    A driver with a single win gets that race, correctly - the search cannot
    find anything more typical than the only sample there is, and the report
    says the pool was one.
    """
    positions = np.asarray(result['positions'])
    diag = result['diag']
    codes = result['pace']['Driver'].to_numpy()

    entries = []
    for row in winners(positions, codes):
        if row['wins'] < min_wins:
            continue
        won = races_won_by(positions, row['driver_idx'])
        sim, report = find_representative(positions, diag, restrict_to=won)
        trace = trace_for(diag, sim, rebuild_compounds)
        entries.append({
            **row,
            'sim': int(sim),
            'report': report,
            'trace': trace,
            'stints': build_stints(trace['compounds'], trace['pits'], codes,
                                   retired_lap=trace['retired_lap']),
        })
    return entries
