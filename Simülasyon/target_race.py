"""
F1 Prediction Simulation - target_race.py
The one place that says which 2026 Grand Prix is being predicted.

Change TARGET_RACE below and every module that needs to know which circuit is
the target - fetch.py, simulate.py, tyre curve lookup, track params, track
racing, track affinity - follows it. Nothing else needs editing.

After changing TARGET_RACE, re-run fetch.py so the laps/grid split matches the
new target (its laps get withheld from training, its grid gets downloaded
instead of the placeholder from the old target):

    python -m Simülasyon.data_prep.fetch

Then run the simulation as usual:

    python -m Simülasyon.simulate
"""

# --- the switch ---------------------------------------------------------

TARGET_RACE = 'italian'          # <-- change this one line to switch GP

# The backtest overrides the line above, because scoring the model across a
# season means switching circuit fifty times in one process rather than once by
# hand. Unset, which is every normal run, the line above is what applies.
import os as _os
TARGET_RACE = _os.environ.get('F1_TARGET_RACE') or TARGET_RACE

# --- registry -------------------------------------------------------------
# key: short name used above.
# event: the name simulate.py, tracks.py, fuel_effect.py and track_deg.py
#        match against (substring, case-insensitive).
# aliases: every string fetch.py should treat as this circuit when matching
#          FastF1's EventName/Country - country names included, since FastF1
#          calls this race "Dutch Grand Prix" but the country is "Netherlands".
#
# 2026 calendar, in round order.

RACE_REGISTRY = {
    'australian':  ('Australian Grand Prix', ['australian', 'australia', 'melbourne']),
    'chinese':     ('Chinese Grand Prix', ['chinese', 'china', 'shanghai']),
    'japanese':    ('Japanese Grand Prix', ['japanese', 'japan', 'suzuka']),
    'bahrain':     ('Bahrain Grand Prix', ['bahrain', 'sakhir']),
    'saudi':       ('Saudi Arabian Grand Prix', ['saudi', 'jeddah']),
    'miami':       ('Miami Grand Prix', ['miami']),
    'emilia':      ('Emilia Romagna Grand Prix', ['emilia', 'imola']),
    'canadian':    ('Canadian Grand Prix', ['canadian', 'canada', 'montreal']),
    'monaco':      ('Monaco Grand Prix', ['monaco', 'monte carlo']),
    'spanish':     ('Spanish Grand Prix', ['spanish', 'spain', 'barcelona']),
    'austrian':    ('Austrian Grand Prix', ['austrian', 'austria', 'spielberg']),
    'british':     ('British Grand Prix', ['british', 'britain', 'silverstone']),
    'belgian':     ('Belgian Grand Prix', ['belgian', 'belgium', 'spa']),
    'hungarian':   ('Hungarian Grand Prix', ['hungarian', 'hungary', 'hungaroring']),
    'dutch':       ('Dutch Grand Prix', ['dutch', 'netherlands', 'zandvoort']),
    'italian':     ('Italian Grand Prix', ['italian', 'italy', 'monza']),
    'azerbaijan':  ('Azerbaijan Grand Prix', ['azerbaijan', 'baku']),
    'singapore':   ('Singapore Grand Prix', ['singapore', 'marina bay']),
    'us':          ('United States Grand Prix', ['united states', 'usa', 'austin', 'cota']),
    'mexico':      ('Mexico City Grand Prix', ['mexico', 'mexico city']),
    'brazil':      ('São Paulo Grand Prix', ['brazil', 'brazilian', 'sao paulo',
                                                  'são paulo', 'interlagos']),
    'vegas':       ('Las Vegas Grand Prix', ['las vegas', 'vegas']),
    'qatar':       ('Qatar Grand Prix', ['qatar', 'lusail']),
    'abu_dhabi':   ('Abu Dhabi Grand Prix', ['abu dhabi', 'yas marina']),
}


def resolve(key=None):
    """
    (event_name, aliases) for a registry key. Raises with the list of known
    keys when the key is wrong, rather than silently falling back to the
    defaults every per-track lookup already has - a typo here should stop the
    run, not quietly predict the wrong circuit.
    """
    key = key or TARGET_RACE
    try:
        return RACE_REGISTRY[key]
    except KeyError:
        known = ', '.join(sorted(RACE_REGISTRY))
        raise SystemExit(f"unknown TARGET_RACE '{key}'. Known keys: {known}")


TARGET_EVENT, TRACK_ALIASES = resolve()
