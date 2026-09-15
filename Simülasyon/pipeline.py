"""
F1 Prediction Simulation - v2.4 - pipeline.py
One entry point: what data exists, what is stale, and run the race.

    python -m Simülasyon.pipeline                 what is ready and what is not
    python -m Simülasyon.pipeline --update        rebuild what is stale
    python -m Simülasyon.pipeline --run           check, then predict

The problem this solves
-----------------------
Switching race meant remembering to run fetch, then clean, then team_affinity,
in that order, and knowing which of them the change actually affected. Getting
it wrong produced a confident prediction built on another circuit's grid, which
is the sort of wrong that looks right.

So the dependencies are written down once, here, and the answer to "what do I
need to run" comes from comparing what is on disk against them.

Staleness is not a timestamp
----------------------------
Two things make a derived file stale, and only one of them is about its inputs.

The first is the obvious one: a file it was built from has changed. That is
checked by content hash rather than modification time, because touching a file
is not changing it and a restored backup has an old date and new contents.

The second is that the code that produces it has changed. A cleaning rule or a
coefficient definition can move while every raw byte stays where it was, and
the derived output is then wrong in a way no input hash will ever notice. So
each entry also records a hash of its producer's source.

What this does not claim
------------------------
It is not a historical replay. The project keeps one copy of each data file and
overwrites it, so there is no way to reconstruct what was knowable before last
Sunday's race - a record downloaded today is today's version of it, whatever
date the event has. The cutoff below says which sessions a prediction is
allowed to use; it cannot say what those sessions looked like at the time.

Nothing here schedules anything. Every update is something the user asked for,
in this process, once.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(DATA_DIR, 'manifest.json')

READY, MISSING, STALE, FAILED = 'ready', 'missing', 'stale', 'failed'

# How much of a file is hashed. Reading 200 MB of lap cache on every status
# check costs more than it is worth; the head and the tail together catch a
# rewrite, a truncation and an append, which is what changes here.
HASH_BYTES = 1 << 20

# The dependency graph, written down once.
#
# `produces` is what a step writes, `needs` what it reads, and `module` is the
# source whose hash decides whether the step's own rules have moved. `required`
# marks the inputs a prediction cannot proceed without - everything else can be
# absent and the model has a documented fallback for it.
#
# `optional_produces` is for the rest of what a step writes: byproducts a
# developer tool reads (a stint inspector, a profiling script) but predict()
# never does. It is tracked the same way - staleness, hashes, the manifest -
# so a status report still shows it; it just never blocks a run, because
# nothing a run needs is missing when it is. Confusing the two once meant a
# fresh clone could not predict anything until it happened to run the same
# developer tool that wanted the byproduct.
#
# The order of this list is the order the steps run in.
STEPS = [
    dict(name='season laps', module='data_prep/fetch.py', network=True,
         needs=[], required=True,
         produces=['f1_2026_laps.csv', 'f1_2026_grid.csv',
                   'f1_2026_poles.csv', 'f1_2026_quali.csv',
                   'f1_2026_results.csv'],
         note='downloads the season and withholds the target race'),

    dict(name='historical laps', module='data_prep/dataset.py', network=True,
         needs=[], required=False,
         produces=['laps_2018_2025.csv', 'results_2018_2025.csv'],
         note='2018-2025, only rebuilt on request - it is a long download'),

    dict(name='driver pace', module='data_prep/clean.py', network=False,
         needs=['f1_2026_laps.csv', 'f1_2026_poles.csv'], required=True,
         produces=['driver_pace_2026.csv'],
         # The lap-level file with the clean_lap flag kept. Read by
         # inspect_stints.py, Tyre_model/deg.py and profile_run.py - all
         # developer tools - and by nothing predict() calls.
         optional_produces=['f1_2026_laps_clean.csv'],
         note='delta to pole and lap-to-lap sigma'),

    dict(name='team affinity', module='data_prep/team_affinity.py', network=False,
         needs=['f1_2026_results.csv', 'f1_2026_quali.csv'], required=False,
         produces=['team_affinity_2026.csv'],
         note='which 2026 circuits suit which 2026 team'),

    dict(name='overtaking', module='data_prep/overtaking.py', network=False,
         needs=['laps_2018_2025.csv'], required=False,
         produces=['overtaking.csv'],
         note='pass rates per circuit'),

    dict(name='neutralization', module='data_prep/neutralization.py', network=False,
         needs=['laps_2018_2025.csv'], required=False,
         produces=['neutralization.csv'],
         note='safety car and red flag rates'),

    dict(name='pit analysis', module='data_prep/pit_analysis.py', network=False,
         needs=['laps_2018_2025.csv'], required=True,
         produces=['pit_summary.csv', 'pit_strategies.csv'],
         note='pit loss and the strategy menu'),

    dict(name='tyre curves', module='Tyre_model/fit_tyre_curve.py',
         network=False, needs=['laps_2018_2025.csv'], required=True,
         produces=['tyre_curve_params.json'],
         note='degradation per compound per circuit'),

    dict(name='retirement risk', module='data_prep/retirements.py', network=False,
         needs=['results_2018_2025.csv', 'laps_2018_2025.csv'],
         required=False,
         produces=['dnf_driver_risk.csv', 'dnf_team_risk.csv',
                   'dnf_profile.csv'],
         note='accident and mechanical rates per lap at risk'),

    dict(name='wet profiles', module='data_prep/wet_conditions.py', network=False,
         needs=['laps_2018_2025.csv'], required=False,
         produces=['wet_profiles.csv'],
         note='what the lap data says about running in the rain'),
]


# --- identity ---------------------------------------------------------------

def resolve_race(name=None):
    """
    A race's canonical identity, not just whatever it was called.

    Dutch and Netherlands are the same Sunday and the registry knows it. An
    unknown name raises rather than being matched to the nearest thing that
    looks similar, because a prediction quietly built for a different circuit
    is worse than no prediction.
    """
    from Simülasyon import target_race as TR

    # Normalise both sides the same way, or the canonical name fails to match
    # itself: "Italian Grand Prix" becomes "italian_grand_prix" while the
    # registry holds "Italian Grand Prix" with spaces, and the comparison
    # silently falls through to "unknown race".
    def flatten(text):
        return str(text).strip().lower().replace(' ', '_')

    key = flatten(name or TR.TARGET_RACE)
    if key in TR.RACE_REGISTRY:
        event, aliases = TR.RACE_REGISTRY[key]
        return {'key': key, 'event': event, 'aliases': list(aliases),
                'season': 2026}

    for registry_key, (event, aliases) in TR.RACE_REGISTRY.items():
        if key == flatten(event) or key in [flatten(a) for a in aliases]:
            return {'key': registry_key, 'event': event,
                    'aliases': list(aliases), 'season': 2026}

    known = ', '.join(sorted(TR.RACE_REGISTRY))
    raise KeyError(f'unknown race "{name}". Known keys: {known}')


def data_belongs_to():
    """Which race the files on disk were actually built for."""
    import pandas as pd

    path = os.path.join(DATA_DIR, 'f1_2026_poles.csv')
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    if 'is_target' not in frame.columns:
        return None
    hit = frame[frame['is_target'].astype(bool)]
    return str(hit.iloc[0]['Race']) if not hit.empty else None


def cutoff(race=None):
    """
    What a prediction at this moment is allowed to read.

    The model needs qualifying: the grid and the pole time come from it. Before
    it has run there is no honest prediction to make, and this says so rather
    than falling back to something that looks like an answer.

    It describes the present. It cannot reconstruct the past - see the module
    docstring.
    """
    race = race or resolve_race()
    on_disk = data_belongs_to()
    matches = (on_disk is not None
               and any(a in on_disk.lower() for a in race['aliases']))
    return {
        'race': race['event'],
        'data_belongs_to': on_disk,
        'matches': matches,
        'needs': 'qualifying complete: grid and pole time',
        'usable': matches,
        'historical_replay': False,
    }


# --- manifest ---------------------------------------------------------------

def _hash_file(path):
    """Content digest of a file's head and tail, with its size."""
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with open(path, 'rb') as handle:
        digest.update(handle.read(HASH_BYTES))
        if size > 2 * HASH_BYTES:
            handle.seek(-HASH_BYTES, os.SEEK_END)
            digest.update(handle.read(HASH_BYTES))
    return digest.hexdigest()[:16]


def _hash_module(relative):
    """
    Digest of a step's source, so a change of rules invalidates its output.

    This is the half of staleness that input hashing cannot see. A cleaning
    rule can move while every raw byte stays where it was, and the derived file
    is then built by code that no longer exists.
    """
    path = os.path.join(CODE_DIR, relative)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def load_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    try:
        with open(MANIFEST, encoding='utf-8') as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(manifest):
    write_atomic(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))


def write_atomic(path, text):
    """
    Write to a temporary file, then move it into place.

    A download interrupted halfway through must not leave something that looks
    ready. The move is the only moment anything changes, and it either happens
    or it does not.
    """
    temporary = path + '.partial'
    with open(temporary, 'w', encoding='utf-8') as handle:
        handle.write(text)
    os.replace(temporary, path)


def scan(race=None):
    """
    Every artefact's state, and why.

    An entry is stale when one of its inputs has changed content, or when its
    producer's source has changed. It is missing when it is not there. It is
    ready otherwise - and "ready" says nothing about whether the numbers in it
    are any good, only that the pipeline has nothing left to do to it.
    """
    race = race or resolve_race()
    previous = load_manifest()
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')

    current, entries = {}, []
    for step in STEPS:
        code = _hash_module(step['module'])
        code_changed = (previous.get('_code', {}).get(step['module'])
                        not in (None, code))

        # Required artefacts gate a run; optional ones are tracked the same
        # way and gate nothing - see the note on optional_produces above.
        artefacts = ([(a, step['required']) for a in step['produces']]
                    + [(a, False) for a in step.get('optional_produces', [])])

        for artefact, required in artefacts:
            path = os.path.join(DATA_DIR, artefact)
            exists = os.path.exists(path)
            digest = _hash_file(path) if exists else None
            was = previous.get(artefact, {})

            if not exists:
                state, reason = MISSING, 'not on disk'
            elif code_changed:
                state, reason = STALE, f'{step["module"]} has changed'
            else:
                moved = [n for n in step['needs']
                         if _input_changed(n, previous)]
                if moved:
                    state, reason = STALE, f'input changed: {", ".join(moved)}'
                else:
                    state, reason = READY, 'up to date'

            current[artefact] = {
                'hash': digest, 'step': step['name'],
                'module': step['module'], 'needs': step['needs'],
                'required': required,
                'seen': was.get('seen', now) if digest == was.get('hash')
                        else now,
                'size': os.path.getsize(path) if exists else 0,
            }
            entries.append({'artefact': artefact, 'state': state,
                            'reason': reason, 'step': step['name'],
                            'required': required,
                            'network': step['network'], 'hash': digest})

    current['_code'] = {s['module']: _hash_module(s['module']) for s in STEPS}
    current['_race'] = race['event']
    return entries, current


def _input_changed(artefact, previous):
    """Whether a dependency's content differs from what the manifest recorded."""
    path = os.path.join(DATA_DIR, artefact)
    if not os.path.exists(path):
        return False        # missing inputs are reported on their own row
    was = previous.get(artefact, {}).get('hash')
    return was is not None and was != _hash_file(path)


def blocking(entries):
    """The required artefacts that are not ready. A prediction stops on these."""
    return [e for e in entries if e['required'] and e['state'] != READY]


# --- running ----------------------------------------------------------------

def update(entries, allow_network=False, dry_run=True):
    """
    Rebuild only what the scan says needs rebuilding, in dependency order.

    Nothing runs unless it was asked for. A step that needs the network is
    skipped unless that is explicitly allowed, because a status check should
    never start a download.
    """
    todo, done, skipped = [], [], []
    needs_work = {e['step'] for e in entries if e['state'] != READY}

    for step in STEPS:
        if step['name'] not in needs_work:
            continue
        if step['network'] and not allow_network:
            skipped.append((step['name'], 'needs the network'))
            continue
        todo.append(step)

    if dry_run:
        return {'would_run': [s['name'] for s in todo], 'skipped': skipped,
                'ran': []}

    for step in todo:
        module = step['module'].replace('/', '.').replace('.py', '')
        started = time.perf_counter()
        try:
            import runpy
            runpy.run_module(f'Simülasyon.{module}', run_name='__main__')
            done.append((step['name'], time.perf_counter() - started, None))
        except Exception as exc:                         # noqa: BLE001
            # A failed step must not be reported as done and must not have
            # damaged what was there: every writer goes through write_atomic.
            done.append((step['name'], time.perf_counter() - started,
                         f'{type(exc).__name__}: {exc}'))
            break

    return {'would_run': [s['name'] for s in todo], 'skipped': skipped,
            'ran': done}


def run_summary(module, result, seconds, requested, completed, status):
    """
    What this result is, kept beside it in memory and nowhere else.

    Enough to say whether two results are comparable: the race, the seed, how
    many simulations actually finished, the parameter set, the flags, and the
    versions of everything that does arithmetic.
    """
    import numpy as np
    import pandas as pd

    return {
        'race': module.TARGET_EVENT,
        'season': module.SEASON,
        'cutoff': cutoff(),
        'requested_runs': requested,
        'completed_runs': completed,
        'seed': module.RANDOM_SEED,
        'parameter_set': getattr(module, 'PARAM_SET_VERSION', '?'),
        'flags': {k: getattr(module, k) for k in module.RUNTIME_FLAGS},
        'choices': {k: getattr(module, k) for k in module.RUNTIME_CHOICES},
        'data': {k: v['hash'] for k, v in load_manifest().items()
                 if not k.startswith('_')},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'pandas': pd.__version__},
        'seconds': round(seconds, 2),
        'status': status,
    }


def predict(race=None, n_sims=None, overrides=None, allow_network=False,
            progress=None):
    """
    Check the data, refuse clearly if it is not usable, otherwise run.

    The check happens once, before the first lap. Updating data inside the lap
    loop or inside a Monte Carlo run is how a prediction ends up built on two
    different versions of the same file.
    """
    from Simülasyon import simulate as S

    race = resolve_race(race)
    entries, manifest = scan(race)
    save_manifest(manifest)

    state = cutoff(race)
    if not state['matches']:
        raise RuntimeError(
            f'the data on disk was built for {state["data_belongs_to"]}, not '
            f'{race["event"]}. Set TARGET_RACE in target_race.py and run the '
            f'pipeline with --update --network before predicting.')

    stop = blocking(entries)
    if stop:
        names = ', '.join(f'{e["artefact"]} ({e["state"]})' for e in stop)
        raise RuntimeError(f'required data is not ready: {names}. '
                           f'Run the pipeline with --update.')

    requested = n_sims or S.N_SIMS
    previous = S.N_SIMS
    S.N_SIMS = requested
    started = time.perf_counter()
    try:
        result = S.run(overrides, progress_callback=progress)
        status = 'complete'
    finally:
        S.N_SIMS = previous

    seconds = time.perf_counter() - started
    result['run'] = run_summary(S, result, seconds, requested,
                                result['n_sims'], status)
    return result


# --- report -----------------------------------------------------------------

def report(race=None, do_update=False, allow_network=False, do_run=False):
    race = resolve_race(race)
    entries, manifest = scan(race)

    print(f'\n=== {race["event"]} {race["season"]} ===')
    state = cutoff(race)
    print(f'  data on disk belongs to: {state["data_belongs_to"]}')
    print(f'  matches this race      : {state["matches"]}')
    print(f'  prediction needs       : {state["needs"]}')

    by_state = {}
    for entry in entries:
        by_state.setdefault(entry['state'], []).append(entry)

    print(f'\n  {len(entries)} artefacts: '
          + ', '.join(f'{len(v)} {k}' for k, v in sorted(by_state.items())))
    for entry in entries:
        if entry['state'] == READY:
            continue
        mark = 'REQUIRED' if entry['required'] else 'optional'
        print(f'    {entry["state"]:<8} {entry["artefact"]:<28} '
              f'{mark:<9} {entry["reason"]}')

    stop = blocking(entries)
    if stop:
        print(f'\n  a prediction cannot run: {len(stop)} required artefacts '
              f'are not ready')
    else:
        print(f'\n  everything required is ready')

    plan = update(entries, allow_network=allow_network,
                  dry_run=not do_update)
    if plan['would_run']:
        verb = 'ran' if do_update else 'would run'
        print(f'\n  {verb}: {", ".join(plan["would_run"])}')
    for name, why in plan['skipped']:
        print(f'  skipped {name}: {why}')
    for name, seconds, error in plan['ran']:
        print(f'  {name}: {seconds:.1f} s' + (f'  FAILED {error}' if error
                                              else ''))

    save_manifest(manifest)
    print(f'\n  manifest: {MANIFEST}')

    if do_run and not blocking(entries):
        print(f'\n  running...')
        result = predict(race['key'])
        summary = result['run']
        print(f'  {summary["completed_runs"]:,} simulations in '
              f'{summary["seconds"]:.1f} s, set {summary["parameter_set"]}')
        print(f'  {result["summary"].iloc[0]["Driver"]} '
              f'{result["summary"].iloc[0]["P_win"]:.1%}')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--race', default=None)
    parser.add_argument('--update', action='store_true',
                        help='rebuild what is stale')
    parser.add_argument('--network', action='store_true',
                        help='allow steps that download')
    parser.add_argument('--run', action='store_true',
                        help='predict once the data checks out')
    args = parser.parse_args()
    report(args.race, do_update=args.update, allow_network=args.network,
           do_run=args.run)


if __name__ == '__main__':
    main()
