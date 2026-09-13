"""
F1 Prediction Simulation - v2.4 - profile_run.py
Where the time and the memory actually go.

    python -m Simülasyon.profile_run

Why this exists before any optimisation
---------------------------------------
The v2.4 brief is explicit that an optimisation which does not show up in a
measurement should not ship. So this runs first and decides what is worth
touching, rather than the other way round.

It separates the stages the brief asks to be separated - checking data,
preparing the model, the simulation core, collecting results, and building the
report - because they have completely different characters. A cold fetch is
minutes of network; the lap loop is seconds of arithmetic; and reporting one
as a speed-up of the other is the classic way to claim a ten-fold improvement
that nobody can feel.

Peak memory is measured too, and it turns out to be the more interesting
number: the per-lap traces are most of the process, and they exist so that one
representative race can be drawn.

What the reference run records
------------------------------
Machine, Python and library versions, the race, cars and laps, the model flags,
the parameter set, the seed, the number of simulations, and whether the data
cache was warm. Without those a timing is not a measurement, it is an anecdote.
"""

import gc
import os
import platform
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Median of this many repeats at the same settings. Three is enough to see
# past one slow moment and cheap enough to run while iterating; the spread is
# reported alongside so a noisy machine is visible rather than averaged away.
REPEATS = 3


def environment(n_sims, module):
    """Everything that has to be true for a timing to mean anything."""
    import streamlit
    return {
        'machine': platform.machine(),
        'processor': platform.processor() or platform.machine(),
        'cpus': os.cpu_count(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'streamlit': streamlit.__version__,
        'race': module.TARGET_EVENT,
        'season': module.SEASON,
        'n_sims': n_sims,
        'seed': module.RANDOM_SEED,
        'parameter_set': getattr(module, 'PARAM_SET_VERSION', '?'),
        'cache_warm': os.path.exists(os.path.join(DATA_DIR,
                                                  'f1_2026_laps_clean.csv')),
    }


def stages(module, n_sims, measure_memory=False):
    """
    One run, timed stage by stage. Memory only when asked for, and then in a
    separate run.

    tracemalloc hooks every allocation, and the first version of this function
    left it on while timing. The result was a 1,000-simulation run at 102 s
    against 10,000 at 142 s - ten times the work for a third more time, which
    is not a thing that happens. It was measuring its own instrumentation, and
    unevenly, because the stages allocate at very different rates.

    So timing runs clean and memory is a second pass. The two numbers are
    never taken from the same run.
    """
    timings = {}
    gc.collect()
    if measure_memory:
        tracemalloc.start()

    t = time.perf_counter()
    pace = module.load_drivers()
    track = module.load_track()
    strategies = module.load_strategies(track)
    timings['load inputs'] = time.perf_counter() - t

    t = time.perf_counter()
    if module.RECOST_STRATEGIES:
        strategies, _ = module.recost_strategies(strategies, track)
    timings['price strategies'] = time.perf_counter() - t

    t = time.perf_counter()
    positions, total, strat_idx, diag = module.run_simulation(
        pace, track, strategies, n_sims, module.RANDOM_SEED)
    timings['simulation core'] = time.perf_counter() - t
    peak_core = (tracemalloc.get_traced_memory()[1] if measure_memory else 0)

    t = time.perf_counter()
    summary = module.summarize(pace, positions, strat_idx, strategies, diag,
                               total)
    module.build_param_lines(pace, track, strategies, diag, positions, total)
    module.source_badges(pace, track)
    timings['collect results'] = time.perf_counter() - t

    t = time.perf_counter()
    rep, _ = module.find_representative(positions, diag)
    module.rebuild_compounds(diag, rep)
    timings['representative race'] = time.perf_counter() - t

    if measure_memory:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        peak = 0

    timings['total'] = sum(v for k, v in timings.items() if k != 'total')
    return timings, {'peak_mb': peak / 1e6, 'core_peak_mb': peak_core / 1e6,
                     'summary_rows': len(summary)}


def trace_footprint(module, n_sims):
    """
    What the per-lap traces cost, since they dominate the memory.

    Three arrays are kept for every lap of every simulation so that one
    representative race can be drawn at the end. Which race that is cannot be
    known until they have all finished, which is why they are all kept - but
    it is worth knowing the price.
    """
    track = module.load_track()
    n_laps = track['n_laps']
    n_drivers = len(module.load_drivers())
    cells = n_laps * n_sims * n_drivers
    return {
        'laps': n_laps, 'drivers': n_drivers, 'sims': n_sims,
        'order_mb': cells * 1 / 1e6,        # int8
        'pits_mb': cells * 1 / 1e6,         # bool
        'laptime_mb': cells * 4 / 1e6,      # float32
        'total_mb': cells * 6 / 1e6,
    }


def hot_functions(module, n_sims=2000, top=12):
    """
    The lap loop's own profile, so the optimisation list is evidence-led.

    cProfile over a smaller run: the shape of where the time goes does not
    change with the number of simulations, and a full run under the profiler
    costs more than it tells.
    """
    import cProfile
    import pstats
    import io as _io

    pace = module.load_drivers()
    track = module.load_track()
    strategies = module.load_strategies(track)
    if module.RECOST_STRATEGIES:
        strategies, _ = module.recost_strategies(strategies, track)

    profiler = cProfile.Profile()
    profiler.enable()
    module.run_simulation(pace, track, strategies, n_sims, module.RANDOM_SEED)
    profiler.disable()

    buffer = _io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats('tottime')
    stats.print_stats(top)
    return buffer.getvalue()


def report(n_sims=10_000, small=1_000):
    from Simülasyon import simulate as S
    S.OPEN_REPORT = False

    env = environment(n_sims, S)
    print('\n=== reference run ===')
    for key, value in env.items():
        print(f'  {key:<16} {value}')

    print(f'\n--- small run first: {small:,} simulations ---')
    warm, _ = stages(S, small)
    for name, seconds in warm.items():
        print(f'  {name:<22} {seconds:7.2f} s')
    per_sim = warm['simulation core'] / small

    print(f'\n--- {n_sims:,} simulations, median of {REPEATS} ---')
    runs = [stages(S, n_sims) for _ in range(REPEATS)]
    names = list(runs[0][0])
    for name in names:
        values = sorted(r[0][name] for r in runs)
        share = values[len(values) // 2] / sorted(
            r[0]['total'] for r in runs)[len(runs) // 2]
        print(f'  {name:<22} {values[len(values) // 2]:7.2f} s   '
              f'[{values[0]:.2f}-{values[-1]:.2f}]   {share:5.1%}')

    core = sorted(r[0]['simulation core'] for r in runs)[len(runs) // 2]
    print(f'  scaling                {core / n_sims / per_sim:5.2f}x the '
          f'{small:,}-run cost per simulation')

    # memory in its own pass, because tracemalloc distorts every timing above
    print(f'\n--- peak memory, separate run ---')
    _, memory = stages(S, n_sims, measure_memory=True)
    print(f'  peak {memory["peak_mb"]:.0f} MB   '
          f'at the end of the lap loop {memory["core_peak_mb"]:.0f} MB')

    footprint = trace_footprint(S, n_sims)
    print(f'\n--- what the per-lap traces cost ---')
    print(f'  {footprint["laps"]} laps x {footprint["sims"]:,} sims x '
          f'{footprint["drivers"]} cars')
    print(f'  order {footprint["order_mb"]:.0f} MB + pits '
          f'{footprint["pits_mb"]:.0f} MB + lap times '
          f'{footprint["laptime_mb"]:.0f} MB = '
          f'{footprint["total_mb"]:.0f} MB')

    print(f'\n--- where the lap loop spends its time ---')
    print(hot_functions(S))
    return env, runs


if __name__ == '__main__':
    report()
