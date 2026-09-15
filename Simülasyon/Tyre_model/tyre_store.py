"""
F1 Prediction Simulation - tyre_store.py
Loads the fitted parameter file and hands out curves. The only I/O layer.

    from tyre_store import load, get_curve, catalog

    store = load()                       # data/tyre_curve_params.json
    curve = store.get('Netherlands', 'SOFT')
    curve.loss(12)                       # seconds

Track names are matched, not compared. The season entry calls the race
'Netherlands', FastF1 calls it the 'Dutch Grand Prix' and the parameter file
is keyed on the second. Neither string contains the other, so a plain lookup
fails silently and hands back a different circuit's tyre - which is worse than
no answer. Matching is exact first, then normalised, then by alias, and a miss
raises rather than falling back to a global average.
"""

import json
import os
import sys
import threading

# Reached both ways: as a script from this directory, and as
# Simülasyon.Tyre_model.tyre_store from the project root. Putting this
# directory on the path first makes the flat import below work in either case.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tyre_curve import TyreCurve, TyreCurveError    # noqa: E402

# Found from this file's own location, not written down. The path was
# hardcoded to one machine, which stayed invisible because simulate.py always
# passes its own path explicitly - but pit_analysis.py's main() calls load()
# with no argument, and that call is dead on any machine but the one this was
# written on, cloud deployment included.
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'tyre_curve_params.json')

# Names the same circuit appears under across the project's CSVs.
ALIASES = {
    'netherlands': 'dutch', 'zandvoort': 'dutch',
    'monza': 'italian', 'spa': 'belgian', 'monaco': 'monaco',
    'silverstone': 'british', 'suzuka': 'japanese', 'interlagos': 'brazilian',
    'baku': 'azerbaijan', 'austin': 'united states', 'cota': 'united states',
    'hungaroring': 'hungarian', 'barcelona': 'spanish', 'catalunya': 'spanish',
    'melbourne': 'australian', 'shanghai': 'chinese', 'jeddah': 'saudi',
    'spielberg': 'austrian', 'montreal': 'canadian', 'imola': 'emilia',
    'lusail': 'qatar', 'yas': 'abu dhabi', 'vegas': 'las vegas',
}

_NOISE = ('grand prix', 'gp', 'grand-prix')


class UnknownProfile(KeyError):
    """No curve exists for this track and compound."""


def _normalise(name):
    s = str(name).strip().lower()
    for n in _NOISE:
        s = s.replace(n, ' ')
    return ' '.join(s.split())


class TyreStore:
    """Immutable once loaded. Safe to share across threads and simulations."""

    def __init__(self, payload):
        self.model_version = payload.get('model_version', 'unknown')
        self.model = payload.get('model', 'unknown')
        self.formula = payload.get('formula', '')
        self.age_offset = payload.get('age_offset', 0)
        self.age_definition = payload.get('age_definition', '')
        self.built_utc = payload.get('built_utc', '')

        self._curves = {}
        for d in payload.get('curves', []):
            c = TyreCurve.from_dict(d)
            self._curves[(_normalise(c.track), c.compound.upper())] = c

        self._tracks = sorted({t for t, _ in self._curves})

    # --- lookup -------------------------------------------------------------

    def _resolve_track(self, track):
        key = _normalise(track)
        if any((key, c) in self._curves for c in ('SOFT', 'MEDIUM', 'HARD')):
            return key

        alias = ALIASES.get(key)
        if alias:
            for known in self._tracks:
                if alias in known:
                    return known

        for known in self._tracks:
            if key and (key in known or known in key):
                return known
        return None

    def get(self, track, compound):
        comp = str(compound).strip().upper()
        resolved = self._resolve_track(track)
        if resolved is None:
            raise UnknownProfile(
                f'no tyre profile for track {track!r}; '
                f'known tracks: {len(self._tracks)}')
        curve = self._curves.get((resolved, comp))
        if curve is None:
            have = sorted(c for (t, c) in self._curves if t == resolved)
            raise UnknownProfile(
                f'no {comp} profile for {track!r}; this circuit has {have}')
        return curve

    def has(self, track, compound):
        try:
            self.get(track, compound)
            return True
        except UnknownProfile:
            return False

    def catalog(self):
        """Every profile on offer, with its age limit and provenance."""
        rows = []
        for (_, comp), c in sorted(self._curves.items(),
                                   key=lambda kv: (kv[1].track, kv[1].compound)):
            rows.append({
                'track': c.track,
                'compound': c.compound,
                'min_age': 0,
                'max_age': c.max_age,
                'has_cliff': c.tau is not None,
                'source': c.source,
            })
        return rows

    @property
    def tracks(self):
        return list(self._tracks)

    def __len__(self):
        return len(self._curves)


# --- module-level cache -----------------------------------------------------

_lock = threading.Lock()
_cache = {}


def load(path=DEFAULT_PATH, force=False):
    """
    Reads the parameter file once and caches it.

    The simulation asks for curves inside a 10,000-race loop, so re-reading
    and re-parsing the JSON on every call would dominate the run. The cache is
    keyed on the path, and force=True is there for the fit script's own tests.
    """
    key = os.path.abspath(path)
    with _lock:
        if not force and key in _cache:
            return _cache[key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} not found. Build it with:  python fit_tyre_curve.py')
        with open(path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        store = TyreStore(payload)
        _cache[key] = store
        return store


def get_curve(track, compound, path=DEFAULT_PATH):
    return load(path).get(track, compound)


def catalog(path=DEFAULT_PATH):
    return load(path).catalog()


if __name__ == '__main__':
    store = load()
    print(f'model {store.model} v{store.model_version}  built {store.built_utc}')
    print(f'{len(store)} curves over {len(store.tracks)} tracks')
    print(f'age offset {store.age_offset}\n')

    for name in ['Netherlands', 'Dutch Grand Prix', 'Monza', 'Nowhere GP']:
        try:
            c = store.get(name, 'SOFT')
            print(f'  {name:20s} -> {c!r}')
        except (UnknownProfile, TyreCurveError) as exc:
            print(f'  {name:20s} -> {type(exc).__name__}: {exc}')
