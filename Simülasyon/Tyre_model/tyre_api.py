"""
F1 Prediction Simulation - tyre_api.py
HTTP access to the fitted degradation curves. Standard library only.

    python tyre_api.py                  # serves on 127.0.0.1:8077
    python tyre_api.py --port 9000

Endpoints
---------
    POST /v1/degradation   one age            {track, compound, tyre_age}
    POST /v1/curve         one whole curve    {track, compound, max_age?}
    POST /v1/curves        several curves     {queries: [{track, compound, max_age?}]}
    GET  /v1/catalog       what exists        -

All losses are in seconds and all ages in laps. Every response repeats what was
asked for, so a stored answer can be read months later without the request.

This layer does no arithmetic of its own. It parses, validates, calls
tyre_curve through tyre_store, and serialises - so the number the API returns
and the number the simulation computes locally come from the same code path
and cannot drift apart. Nothing here trains, fits or samples: the parameter
file is read once at startup and the curves are deterministic, so there is no
seed and no sample count to pass.

An unknown track or compound is an error, never a default profile. Handing
back a calendar average under the name of a circuit that was never fitted is
the one failure that would be invisible downstream.
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tyre_curve import TyreCurveError
from tyre_store import UnknownProfile, load

# --- limits -----------------------------------------------------------------

MAX_BODY_BYTES = 256 * 1024
MAX_BATCH_QUERIES = 200
MAX_CURVE_AGE = 200

DEGRADATION_FIELDS = {'track', 'compound', 'tyre_age'}
CURVE_FIELDS = {'track', 'compound', 'max_age'}
BATCH_FIELDS = {'queries'}


class ApiError(Exception):
    def __init__(self, status, message, detail=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


def _reject_unknown_fields(payload, allowed, where):
    extra = set(payload) - allowed
    if extra:
        raise ApiError(400,
                       f'unexpected field(s) in {where}: {sorted(extra)}',
                       f'allowed: {sorted(allowed)}')


def _require(payload, field, where):
    if field not in payload:
        raise ApiError(400, f'{where} requires {field!r}')
    return payload[field]


# --- handlers (pure: dict in, dict out) -------------------------------------


def handle_degradation(store, payload):
    _reject_unknown_fields(payload, DEGRADATION_FIELDS, '/v1/degradation')
    track = _require(payload, 'track', '/v1/degradation')
    compound = _require(payload, 'compound', '/v1/degradation')
    age = _require(payload, 'tyre_age', '/v1/degradation')

    curve = store.get(track, compound)
    return {
        'track': curve.track,
        'compound': curve.compound,
        'tyre_age': age,
        'degradation_loss_seconds': curve.loss(age),
        'incremental_loss_seconds': curve.incremental_loss(age),
        'units': 'seconds',
        'max_age': curve.max_age,
        'source': curve.source,
        'model_version': store.model_version,
    }


def _curve_payload(store, track, compound, max_age):
    curve = store.get(track, compound)
    if max_age is not None:
        if not isinstance(max_age, int) or isinstance(max_age, bool):
            raise ApiError(400, f'max_age must be a whole number, got {max_age!r}')
        if max_age > MAX_CURVE_AGE:
            raise ApiError(400,
                           f'max_age {max_age} exceeds the server limit of '
                           f'{MAX_CURVE_AGE}')
    ages, losses, incs = curve.curve(max_age)
    return {
        'track': curve.track,
        'compound': curve.compound,
        'requested_max_age': max_age,
        'max_age': curve.max_age,
        'ages': ages,
        'degradation_loss_seconds': losses,
        'incremental_loss_seconds': incs,
        'units': 'seconds',
        'source': curve.source,
        'tau': curve.tau,
    }


def handle_curve(store, payload):
    _reject_unknown_fields(payload, CURVE_FIELDS, '/v1/curve')
    track = _require(payload, 'track', '/v1/curve')
    compound = _require(payload, 'compound', '/v1/curve')
    out = _curve_payload(store, track, compound, payload.get('max_age'))
    out['model_version'] = store.model_version
    return out


def handle_curves(store, payload):
    _reject_unknown_fields(payload, BATCH_FIELDS, '/v1/curves')
    queries = _require(payload, 'queries', '/v1/curves')
    if not isinstance(queries, list):
        raise ApiError(400, "'queries' must be a list")
    if not queries:
        raise ApiError(400, "'queries' must not be empty")
    if len(queries) > MAX_BATCH_QUERIES:
        raise ApiError(400,
                       f'{len(queries)} queries exceeds the limit of '
                       f'{MAX_BATCH_QUERIES}')

    results = []
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise ApiError(400, f'queries[{i}] must be an object')
        _reject_unknown_fields(q, CURVE_FIELDS, f'queries[{i}]')
        track = _require(q, 'track', f'queries[{i}]')
        compound = _require(q, 'compound', f'queries[{i}]')
        try:
            results.append(_curve_payload(store, track, compound, q.get('max_age')))
        except (UnknownProfile, TyreCurveError) as exc:
            raise ApiError(400, f'queries[{i}]: {exc}') from exc

    return {'count': len(results), 'curves': results,
            'units': 'seconds', 'model_version': store.model_version}


def handle_catalog(store):
    return {
        'model': store.model,
        'model_version': store.model_version,
        'formula': store.formula,
        'age_offset': store.age_offset,
        'age_definition': store.age_definition,
        'built_utc': store.built_utc,
        'units': {'loss': 'seconds', 'age': 'laps'},
        'deterministic': True,
        'count': len(store),
        'profiles': store.catalog(),
    }


# --- HTTP layer -------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = 'TyreCurveAPI/1.0'
    store = None

    def _send(self, status, body):
        raw = json.dumps(body, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status, message, detail=None):
        body = {'error': message}
        if detail:
            body['detail'] = detail
        self._send(status, body)

    def _read_json(self):
        length = self.headers.get('Content-Length')
        if length is None:
            raise ApiError(411, 'Content-Length is required')
        try:
            n = int(length)
        except ValueError:
            raise ApiError(400, 'Content-Length is not a number')
        if n <= 0:
            raise ApiError(400, 'empty request body')
        if n > MAX_BODY_BYTES:
            raise ApiError(413,
                           f'request body of {n} bytes exceeds the limit of '
                           f'{MAX_BODY_BYTES}')
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f'body is not valid JSON: {exc}') from exc
        if not isinstance(payload, dict):
            raise ApiError(400, 'body must be a JSON object')
        return payload

    def _dispatch(self, method):
        path = self.path.split('?', 1)[0].rstrip('/') or '/'
        routes = {
            ('POST', '/v1/degradation'): handle_degradation,
            ('POST', '/v1/curve'): handle_curve,
            ('POST', '/v1/curves'): handle_curves,
        }

        if method == 'GET' and path == '/v1/catalog':
            return self._send(200, handle_catalog(self.store))

        fn = routes.get((method, path))
        if fn is None:
            known = ['POST /v1/degradation', 'POST /v1/curve',
                     'POST /v1/curves', 'GET /v1/catalog']
            return self._error(404, f'no route for {method} {path}',
                               f'available: {known}')

        payload = self._read_json()
        return self._send(200, fn(self.store, payload))

    def _guarded(self, method):
        try:
            self._dispatch(method)
        except ApiError as exc:
            self._error(exc.status, exc.message, exc.detail)
        except UnknownProfile as exc:
            self._error(404, str(exc).strip('"'),
                        'unknown track/compound pairs are not given a default '
                        'profile; fit one or query a different circuit')
        except TyreCurveError as exc:
            self._error(400, str(exc))
        except Exception as exc:                     # noqa: BLE001
            self._error(500, f'{type(exc).__name__}: {exc}')

    def do_GET(self):
        self._guarded('GET')

    def do_POST(self):
        self._guarded('POST')

    def log_message(self, fmt, *args):
        sys.stderr.write('  %s - %s\n' % (self.address_string(), fmt % args))


def build_server(host, port, path=None):
    store = load(path) if path else load()
    handler = type('BoundHandler', (Handler,), {'store': store})
    return ThreadingHTTPServer((host, port), handler), store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8077)
    ap.add_argument('--params', default=None)
    args = ap.parse_args()

    server, store = build_server(args.host, args.port, args.params)
    print(f'tyre curve API - model {store.model} v{store.model_version}')
    print(f'{len(store)} profiles over {len(store.tracks)} tracks')
    print(f'listening on http://{args.host}:{args.port}/v1/catalog')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping')
        server.shutdown()


if __name__ == '__main__':
    main()
