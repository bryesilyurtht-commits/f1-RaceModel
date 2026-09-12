"""
F1 Prediction Simulation - test_tyre_curve.py
Checks the curve, the store and the API against the cases that matter.

    python test_tyre_curve.py

Covers the six the roadmap asks for: age zero, the cliff boundary, a set that
joins already used, invalid input, the same query twice, and the API agreeing
with the local call. The API is started on a throwaway port inside the test
process, so nothing has to be running first.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tyre_curve import AGE_OFFSET, TyreCurve, TyreCurveError
from tyre_store import UnknownProfile, load
import tyre_api

PASS, FAIL = [], []


def check(name, condition, detail=''):
    (PASS if condition else FAIL).append(name)
    mark = 'ok  ' if condition else 'FAIL'
    print(f'  [{mark}] {name}' + (f'   {detail}' if detail and not condition else ''))


def expect_error(name, fn, exc_types):
    try:
        fn()
    except exc_types as exc:
        check(name, True, str(exc))
        return
    except Exception as exc:                              # noqa: BLE001
        check(name, False, f'wrong exception {type(exc).__name__}: {exc}')
        return
    check(name, False, 'no exception raised')


# --- the curve --------------------------------------------------------------

def test_curve_maths():
    print('\ncurve arithmetic')
    c = TyreCurve('Test', 'MEDIUM', b1=0.04, b2=0.001, tau=10, gamma=0.01,
                  max_age=30)

    check('D(0) = 0', c.loss(0) == 0.0)
    check('incremental at age 0 is 0', c.incremental_loss(0) == 0.0)

    check('below the cliff the cliff term is absent',
          abs(c.loss(5) - (0.04 * 5 + 0.001 * 25)) < 1e-12)

    # continuity: at exactly tau the cliff contributes nothing
    at_tau = c.loss(10)
    plain_at_tau = 0.04 * 10 + 0.001 * 100
    check('continuous at tau', abs(at_tau - plain_at_tau) < 1e-12)

    past = c.loss(11)
    plain_past = 0.04 * 11 + 0.001 * 121
    check('past tau the curve is above the no-cliff value', past > plain_past)
    check('cliff term is gamma*(a-tau)^2',
          abs((past - plain_past) - 0.01 * 1.0) < 1e-12)

    ages = list(range(0, 31))
    losses = [c.loss(a) for a in ages]
    check('non-decreasing in age',
          all(losses[i] >= losses[i - 1] - 1e-12 for i in range(1, len(losses))))

    check('increment equals the difference of totals',
          abs(c.incremental_loss(17) - (c.loss(17) - c.loss(16))) < 1e-12)

    check('slope and increment are different quantities',
          abs(c.slope(20) - c.incremental_loss(20)) > 1e-9)

    flat = TyreCurve('Test', 'HARD', b1=0.0, b2=0.0, tau=None, gamma=0.0,
                     max_age=40)
    check('a flat curve stays at zero', flat.loss(40) == 0.0)


def test_used_set():
    print('\nused set: age carries over, a stop does not reset it')
    c = TyreCurve('Test', 'MEDIUM', b1=0.05, b2=0.0, tau=None, gamma=0.0,
                  max_age=40)
    # a set joining at age 8 and running 5 laps ends at 13
    start, run = 8, 5
    gained = c.loss(start + run) - c.loss(start)
    check('a used set is charged from its own age',
          abs(gained - 0.05 * run) < 1e-12)
    check('a used set costs more than a fresh one at the same lap count',
          c.loss(start + run) > c.loss(run))


def test_invalid_input():
    print('\ninvalid input is refused, not extrapolated')
    c = TyreCurve('Test', 'SOFT', b1=0.05, b2=0.0, tau=None, gamma=0.0,
                  max_age=20)

    expect_error('negative age rejected', lambda: c.loss(-1), TyreCurveError)
    expect_error('fractional age rejected', lambda: c.loss(4.5), TyreCurveError)
    expect_error('age past max_age rejected', lambda: c.loss(21), TyreCurveError)
    expect_error('non-numeric age rejected', lambda: c.loss('twelve'), TyreCurveError)
    expect_error('boolean age rejected', lambda: c.loss(True), TyreCurveError)
    expect_error('curve past max_age rejected', lambda: c.curve(25), TyreCurveError)

    check('age exactly at max_age is allowed', c.loss(20) > 0)

    expect_error('negative coefficient rejected',
                 lambda: TyreCurve('T', 'SOFT', -0.1, 0.0, None, 0.0, 10),
                 TyreCurveError)
    expect_error('gamma without tau rejected',
                 lambda: TyreCurve('T', 'SOFT', 0.1, 0.0, None, 0.5, 10),
                 TyreCurveError)


def test_determinism():
    print('\ndeterminism')
    store = load()
    a = [store.get('Netherlands', 'MEDIUM').loss(a) for a in range(0, 15)]
    b = [store.get('Netherlands', 'MEDIUM').loss(a) for a in range(0, 15)]
    fresh = load(force=True)
    c = [fresh.get('Netherlands', 'MEDIUM').loss(a) for a in range(0, 15)]
    check('same query gives the same answer', a == b)
    check('reloading the file changes nothing', a == c)


def test_store():
    print('\nstore and track matching')
    store = load()
    check('parameter file is not empty', len(store) > 0, f'{len(store)} curves')
    check('age offset is published', store.age_offset == AGE_OFFSET)

    by_alias = store.get('Netherlands', 'SOFT')
    by_name = store.get('Dutch Grand Prix', 'SOFT')
    check('alias and real name resolve to the same curve',
          by_alias.track == by_name.track and by_alias.b1 == by_name.b1)

    expect_error('unknown track raises rather than defaulting',
                 lambda: store.get('Nowhere Grand Prix', 'SOFT'), UnknownProfile)
    expect_error('unknown compound raises',
                 lambda: store.get('Netherlands', 'ULTRASOFT'), UnknownProfile)

    print('\nphysical ordering across compounds')
    bad_wear, bad_cap, checked = [], [], 0
    for track in store.tracks:
        try:
            s = store.get(track, 'SOFT')
            m = store.get(track, 'MEDIUM')
            h = store.get(track, 'HARD')
        except UnknownProfile:
            continue
        checked += 1
        if not (s.max_age <= m.max_age <= h.max_age):
            bad_cap.append(track)
        top = min(s.max_age, m.max_age, h.max_age)
        # a real tolerance on each comparison. Chaining them as
        # `s >= m - tol >= h - tol` cancels the tolerance out and fails on a
        # 3e-18 rounding difference where the order-clip made two curves
        # deliberately equal.
        tol = 1e-9
        for a in range(1, top + 1):
            sl, ml, hl = s.loss(a), m.loss(a), h.loss(a)
            if sl < ml - tol or ml < hl - tol:
                bad_wear.append(track)
                break
    check(f'SOFT <= MEDIUM <= HARD on max_age ({checked} tracks)',
          not bad_cap, str(bad_cap))
    check(f'SOFT >= MEDIUM >= HARD on loss ({checked} tracks)',
          not bad_wear, str(bad_wear))


# --- API --------------------------------------------------------------------

def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read().decode('utf-8'))


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.loads(r.read().decode('utf-8'))


def post_expect_error(name, url, payload, status):
    try:
        post(url, payload)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode('utf-8'))
        check(name, exc.code == status,
              f'got {exc.code}, wanted {status}: {body.get("error")}')
        return
    check(name, False, 'request succeeded')


def test_api():
    print('\nAPI')
    server, store = tyre_api.build_server('127.0.0.1', 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{port}'

    try:
        status, cat = get(base + '/v1/catalog')
        check('catalog responds', status == 200)
        check('catalog lists every profile', cat['count'] == len(store))
        check('catalog states the units',
              cat['units']['loss'] == 'seconds' and cat['units']['age'] == 'laps')
        check('catalog carries the model version', bool(cat['model_version']))
        check('catalog declares determinism', cat['deterministic'] is True)

        status, one = post(base + '/v1/degradation',
                           {'track': 'Netherlands', 'compound': 'MEDIUM',
                            'tyre_age': 12})
        check('degradation responds', status == 200)
        check('response echoes the query',
              one['compound'] == 'MEDIUM' and one['tyre_age'] == 12)

        local = store.get('Netherlands', 'MEDIUM')
        check('API total matches the local call',
              abs(one['degradation_loss_seconds'] - local.loss(12)) < 1e-12)
        check('API increment matches the local call',
              abs(one['incremental_loss_seconds'] - local.incremental_loss(12)) < 1e-12)

        status, cv = post(base + '/v1/curve',
                          {'track': 'Netherlands', 'compound': 'HARD'})
        check('curve responds', status == 200)
        check('curve starts at age 0 with zero loss',
              cv['ages'][0] == 0 and cv['degradation_loss_seconds'][0] == 0.0)
        check('curve runs to max_age', cv['ages'][-1] == cv['max_age'])
        hard = store.get('Netherlands', 'HARD')
        check('every point of the curve matches the local call',
              all(abs(v - hard.loss(a)) < 1e-12
                  for a, v in zip(cv['ages'], cv['degradation_loss_seconds'])))

        status, batch = post(base + '/v1/curves', {'queries': [
            {'track': 'Netherlands', 'compound': 'SOFT'},
            {'track': 'Netherlands', 'compound': 'MEDIUM'},
            {'track': 'Monza', 'compound': 'HARD'},
        ]})
        check('batch responds', status == 200)
        check('batch returns one curve per query', batch['count'] == 3)
        check('batch shares one model version',
              batch['model_version'] == cat['model_version'])
        soft = store.get('Netherlands', 'SOFT')
        check('batch values match the local call',
              all(abs(v - soft.loss(a)) < 1e-12
                  for a, v in zip(batch['curves'][0]['ages'],
                                  batch['curves'][0]['degradation_loss_seconds'])))

        print('\nAPI error handling')
        post_expect_error('unknown track is 404', base + '/v1/degradation',
                          {'track': 'Nowhere GP', 'compound': 'SOFT',
                           'tyre_age': 5}, 404)
        post_expect_error('unknown compound is 404', base + '/v1/degradation',
                          {'track': 'Netherlands', 'compound': 'ULTRA',
                           'tyre_age': 5}, 404)
        post_expect_error('age past the limit is 400', base + '/v1/degradation',
                          {'track': 'Netherlands', 'compound': 'SOFT',
                           'tyre_age': 500}, 400)
        post_expect_error('negative age is 400', base + '/v1/degradation',
                          {'track': 'Netherlands', 'compound': 'SOFT',
                           'tyre_age': -3}, 400)
        post_expect_error('fractional age is 400', base + '/v1/degradation',
                          {'track': 'Netherlands', 'compound': 'SOFT',
                           'tyre_age': 4.5}, 400)
        post_expect_error('missing field is 400', base + '/v1/degradation',
                          {'track': 'Netherlands', 'compound': 'SOFT'}, 400)
        post_expect_error('out-of-scope field is 400', base + '/v1/degradation',
                          {'track': 'Netherlands', 'compound': 'SOFT',
                           'tyre_age': 5, 'seed': 42}, 400)
        post_expect_error('sample count is refused', base + '/v1/curves',
                          {'queries': [{'track': 'Netherlands',
                                        'compound': 'SOFT'}],
                           'n_samples': 10000}, 400)
        post_expect_error('empty batch is 400', base + '/v1/curves',
                          {'queries': []}, 400)
        post_expect_error('oversized batch is 400', base + '/v1/curves',
                          {'queries': [{'track': 'Netherlands',
                                        'compound': 'SOFT'}] * 500}, 400)

        try:
            get(base + '/v1/nothing')
            check('unknown route is 404', False, 'request succeeded')
        except urllib.error.HTTPError as exc:
            check('unknown route is 404', exc.code == 404)

    finally:
        server.shutdown()
        server.server_close()


def main():
    print('=' * 62)
    print('tyre curve test suite')
    print('=' * 62)

    test_curve_maths()
    test_used_set()
    test_invalid_input()
    test_store()
    test_determinism()
    test_api()

    print('\n' + '=' * 62)
    print(f'passed {len(PASS)}   failed {len(FAIL)}')
    if FAIL:
        for name in FAIL:
            print(f'  FAILED: {name}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
