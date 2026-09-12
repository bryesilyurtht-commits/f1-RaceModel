"""
F1 Prediction Simulation - track_deg.py
Per-track tyre degradation, built from Pirelli-style 1-5 severity ratings.

Ratings
-------
Pirelli publishes a characteristics graphic before every race weekend, grading
traction, braking, lateral force, tyre stress, asphalt abrasion and downforce
from 1 to 5. There is no consolidated table or API, so the values below are
transcribed from those previews to the best available knowledge.
They are NOT scraped from an official feed - treat them as a starting point and
correct any entry you can verify.

Two ratings drive degradation and they are not the same thing:
    abrasion  how much rubber the surface takes off (mechanical wear)
    stress    lateral load and energy put through the tyre (thermal deg)

Silverstone and Spa are the classic split: stress 5, abrasion 2. Under the 2026
low-downforce rules the stress side dropped, which is why both measured LOW in
deg.py. Weighting the two lets you tune how much of that carries over.

severity = ABRASION_WEIGHT * abrasion + (1 - ABRASION_WEIGHT) * stress
deg_mult = (severity / FIELD_SEVERITY) ** SEVERITY_EXPONENT

measured_mult holds the 2026 value from deg.py where the race has been run, so
the model can be checked against reality without overwriting the ratings.
"""

# --- tuning -----------------------------------------------------------------

ABRASION_WEIGHT = 0.6        # 1.0 = wear only, 0.0 = lateral stress only
SEVERITY_EXPONENT = 1.8      # >1 stretches the spread between mild and harsh
FIELD_SEVERITY = 3.0         # severity that maps to deg_mult = 1.0

FIELD_MEDIAN_DEG_PCT = 0.0655   # % of a lap per lap of tyre age, 2026 measured

# --- ratings ----------------------------------------------------------------
# abrasion, stress: 1-5   |   measured_mult: from deg.py, None if not yet run

TRACK_DEG = {
    'bahrain':       {'abrasion': 5, 'stress': 4, 'measured_mult': None},
    'saudi':         {'abrasion': 2, 'stress': 4, 'measured_mult': None},
    'jeddah':        {'abrasion': 2, 'stress': 4, 'measured_mult': None},
    'australian':    {'abrasion': 2, 'stress': 3, 'measured_mult': 0.85},
    'melbourne':     {'abrasion': 2, 'stress': 3, 'measured_mult': 0.85},
    'japanese':      {'abrasion': 3, 'stress': 5, 'measured_mult': 0.35},
    'suzuka':        {'abrasion': 3, 'stress': 5, 'measured_mult': 0.35},
    'chinese':       {'abrasion': 2, 'stress': 3, 'measured_mult': 0.14},
    'shanghai':      {'abrasion': 2, 'stress': 3, 'measured_mult': 0.14},
    'miami':         {'abrasion': 3, 'stress': 3, 'measured_mult': 0.77},
    'emilia':        {'abrasion': 3, 'stress': 4, 'measured_mult': None},
    'imola':         {'abrasion': 3, 'stress': 4, 'measured_mult': None},
    'monaco':        {'abrasion': 1, 'stress': 1, 'measured_mult': 1.19},
    'monte carlo':   {'abrasion': 1, 'stress': 1, 'measured_mult': 1.19},
    'spanish':       {'abrasion': 4, 'stress': 5, 'measured_mult': 2.57},
    'barcelona':     {'abrasion': 4, 'stress': 5, 'measured_mult': 2.57},
    'canadian':      {'abrasion': 1, 'stress': 2, 'measured_mult': 0.09},
    'montreal':      {'abrasion': 1, 'stress': 2, 'measured_mult': 0.09},
    'austrian':      {'abrasion': 3, 'stress': 4, 'measured_mult': 1.98},
    'spielberg':     {'abrasion': 3, 'stress': 4, 'measured_mult': 1.98},
    'british':       {'abrasion': 2, 'stress': 5, 'measured_mult': 0.48},
    'silverstone':   {'abrasion': 2, 'stress': 5, 'measured_mult': 0.48},
    'hungarian':     {'abrasion': 2, 'stress': 3, 'measured_mult': 1.58},
    'hungaroring':   {'abrasion': 2, 'stress': 3, 'measured_mult': 1.58},
    'belgian':       {'abrasion': 2, 'stress': 5, 'measured_mult': 0.38},
    'spa':           {'abrasion': 2, 'stress': 5, 'measured_mult': 0.38},
    'dutch':         {'abrasion': 3, 'stress': 4, 'measured_mult': 1.36},
    'netherlands':   {'abrasion': 3, 'stress': 4, 'measured_mult': 1.36},
    'zandvoort':     {'abrasion': 3, 'stress': 4, 'measured_mult': 1.36},
    'italian':       {'abrasion': 2, 'stress': 4, 'measured_mult': None},
    'monza':         {'abrasion': 2, 'stress': 4, 'measured_mult': None},
    'azerbaijan':    {'abrasion': 1, 'stress': 2, 'measured_mult': None},
    'baku':          {'abrasion': 1, 'stress': 2, 'measured_mult': None},
    'singapore':     {'abrasion': 2, 'stress': 2, 'measured_mult': None},
    'united states': {'abrasion': 3, 'stress': 4, 'measured_mult': None},
    'austin':        {'abrasion': 3, 'stress': 4, 'measured_mult': None},
    'mexico':        {'abrasion': 2, 'stress': 2, 'measured_mult': None},
    'brazilian':     {'abrasion': 3, 'stress': 4, 'measured_mult': None},
    'interlagos':    {'abrasion': 3, 'stress': 4, 'measured_mult': None},
    'las vegas':     {'abrasion': 1, 'stress': 2, 'measured_mult': None},
    'qatar':         {'abrasion': 4, 'stress': 5, 'measured_mult': None},
    'lusail':        {'abrasion': 4, 'stress': 5, 'measured_mult': None},
    'abu dhabi':     {'abrasion': 2, 'stress': 3, 'measured_mult': None},
    'yas':           {'abrasion': 2, 'stress': 3, 'measured_mult': None},
}

DEFAULT_ENTRY = {'abrasion': 3, 'stress': 3, 'measured_mult': None}


def severity(abrasion, stress):
    """Blended 1-5 severity score."""
    return ABRASION_WEIGHT * abrasion + (1 - ABRASION_WEIGHT) * stress


def get_track_deg(event_name, base_lap_time=None, use_measured=False):
    """
    Degradation parameters for an event.

    use_measured=True returns the 2026 measured multiplier when one exists,
    falling back to the rating-based one otherwise.

    Returns dict with:
        abrasion, stress, severity   the inputs
        deg_mult                     multiplier on the field median
        deg_pct                      % of a lap per lap of tyre age
        deg_abs                      s per lap, only if base_lap_time is given
        source                       'rating' or 'measured'
        measured_mult                2026 value, None if the race has not run
    """
    key = event_name.lower()
    entry = None
    for track, params in TRACK_DEG.items():
        if track in key:
            entry = params
            break

    if entry is None:
        print(f"[track_deg] '{event_name}' not found, using defaults.")
        entry = DEFAULT_ENTRY

    sev = severity(entry['abrasion'], entry['stress'])
    rating_mult = (sev / FIELD_SEVERITY) ** SEVERITY_EXPONENT

    if use_measured and entry['measured_mult'] is not None:
        mult, source = entry['measured_mult'], 'measured'
    else:
        mult, source = rating_mult, 'rating'

    deg_pct = FIELD_MEDIAN_DEG_PCT * mult
    out = {
        'abrasion': entry['abrasion'],
        'stress': entry['stress'],
        'severity': round(sev, 2),
        'deg_mult': round(mult, 3),
        'deg_pct': round(deg_pct, 5),
        'source': source,
        'measured_mult': entry['measured_mult'],
    }
    if base_lap_time is not None:
        out['deg_abs'] = round(deg_pct / 100.0 * base_lap_time, 5)
    return out


def compare_to_measured():
    """Prints rating-based multipliers next to the 2026 measured ones."""
    seen, rows = set(), []
    for track, e in TRACK_DEG.items():
        sig = (e['abrasion'], e['stress'], e['measured_mult'])
        if sig in seen:
            continue
        seen.add(sig)
        sev = severity(e['abrasion'], e['stress'])
        rating = (sev / FIELD_SEVERITY) ** SEVERITY_EXPONENT
        rows.append((track, e['abrasion'], e['stress'], sev, rating, e['measured_mult']))

    rows.sort(key=lambda r: -r[4])
    print(f"{'track':14s} {'abr':>4s} {'str':>4s} {'sev':>5s} {'rating':>7s} {'measured':>9s} {'diff':>7s}")
    for t, a, s, sev, rating, meas in rows:
        m = f'{meas:9.2f}' if meas is not None else f'{"-":>9s}'
        d = f'{rating - meas:+7.2f}' if meas is not None else f'{"-":>7s}'
        print(f'{t:14s} {a:4d} {s:4d} {sev:5.1f} {rating:7.2f} {m} {d}')


if __name__ == '__main__':
    compare_to_measured()
    print()
    for name in ['Netherlands', 'Barcelona', 'Monza', 'Qatar']:
        print(f'{name:12s} -> {get_track_deg(name, base_lap_time=75.0)}')