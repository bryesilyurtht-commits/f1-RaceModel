"""
F1 Prediction Simulation - track parameters
Per-track fuel burn effect (s/lap) and race distance.

fuel_effect = lap time gained per lap as the car burns fuel.
Higher on tracks dominated by acceleration and braking, lower on fast tracks.
Precursor to config/tracks.yaml (v0.6).
"""

# key: lowercase substring matched against the event name / country
#
# Order matters: the first key found inside the event name wins. Keep the
# more specific keys above the ones they contain, otherwise 'italian' would
# swallow nothing but 'emilia' style names could be caught by a shorter key
# added later.
TRACK_PARAMS = {
    'bahrain':      {'fuel_effect': 0.050, 'n_laps': 57},
    'sakhir':       {'fuel_effect': 0.050, 'n_laps': 57},
    'saudi':        {'fuel_effect': 0.035, 'n_laps': 50},
    'jeddah':       {'fuel_effect': 0.035, 'n_laps': 50},
    'australian':   {'fuel_effect': 0.045, 'n_laps': 58},
    'melbourne':    {'fuel_effect': 0.045, 'n_laps': 58},
    'japanese':     {'fuel_effect': 0.040, 'n_laps': 53},
    'suzuka':       {'fuel_effect': 0.040, 'n_laps': 53},
    'chinese':      {'fuel_effect': 0.045, 'n_laps': 56},
    'shanghai':     {'fuel_effect': 0.045, 'n_laps': 56},
    'miami':        {'fuel_effect': 0.045, 'n_laps': 57},
    'emilia':       {'fuel_effect': 0.045, 'n_laps': 63},
    'imola':        {'fuel_effect': 0.045, 'n_laps': 63},
    'monaco':       {'fuel_effect': 0.060, 'n_laps': 78},
    'monte carlo':  {'fuel_effect': 0.060, 'n_laps': 78},
    'spanish':      {'fuel_effect': 0.045, 'n_laps': 66},
    'barcelona':    {'fuel_effect': 0.045, 'n_laps': 66},
    'canadian':     {'fuel_effect': 0.050, 'n_laps': 70},
    'montreal':     {'fuel_effect': 0.050, 'n_laps': 70},
    'austrian':     {'fuel_effect': 0.045, 'n_laps': 71},
    'styrian':      {'fuel_effect': 0.045, 'n_laps': 71},
    'spielberg':    {'fuel_effect': 0.045, 'n_laps': 71},
    'red bull ring': {'fuel_effect': 0.045, 'n_laps': 71},
    'british':      {'fuel_effect': 0.038, 'n_laps': 52},
    'silverstone':  {'fuel_effect': 0.038, 'n_laps': 52},
    'french':       {'fuel_effect': 0.040, 'n_laps': 53},
    'paul ricard':  {'fuel_effect': 0.040, 'n_laps': 53},
    'hungarian':    {'fuel_effect': 0.058, 'n_laps': 70},
    'hungaroring':  {'fuel_effect': 0.058, 'n_laps': 70},
    'belgian':      {'fuel_effect': 0.032, 'n_laps': 44},
    'spa':          {'fuel_effect': 0.032, 'n_laps': 44},
    'dutch':        {'fuel_effect': 0.045, 'n_laps': 72},
    'netherlands':  {'fuel_effect': 0.045, 'n_laps': 72},
    'zandvoort':    {'fuel_effect': 0.045, 'n_laps': 72},
    'italian':      {'fuel_effect': 0.030, 'n_laps': 53},
    'monza':        {'fuel_effect': 0.030, 'n_laps': 53},
    'azerbaijan':   {'fuel_effect': 0.038, 'n_laps': 51},
    'baku':         {'fuel_effect': 0.038, 'n_laps': 51},
    'singapore':    {'fuel_effect': 0.058, 'n_laps': 62},
    'marina bay':   {'fuel_effect': 0.058, 'n_laps': 62},
    'united states': {'fuel_effect': 0.045, 'n_laps': 56},
    'austin':       {'fuel_effect': 0.045, 'n_laps': 56},
    'mexico':       {'fuel_effect': 0.042, 'n_laps': 71},
    'brazilian':    {'fuel_effect': 0.048, 'n_laps': 71},
    'sao paulo':    {'fuel_effect': 0.048, 'n_laps': 71},
    'são paulo':    {'fuel_effect': 0.048, 'n_laps': 71},
    'interlagos':   {'fuel_effect': 0.048, 'n_laps': 71},
    'las vegas':    {'fuel_effect': 0.033, 'n_laps': 50},
    'qatar':        {'fuel_effect': 0.042, 'n_laps': 57},
    'lusail':       {'fuel_effect': 0.042, 'n_laps': 57},
    'abu dhabi':    {'fuel_effect': 0.048, 'n_laps': 58},
    'yas':          {'fuel_effect': 0.048, 'n_laps': 58},
    'madrid':       {'fuel_effect': 0.045, 'n_laps': 57},
    'madring':      {'fuel_effect': 0.045, 'n_laps': 57},
    'portuguese':   {'fuel_effect': 0.045, 'n_laps': 66},
    'portimao':     {'fuel_effect': 0.045, 'n_laps': 66},
    'turkish':      {'fuel_effect': 0.042, 'n_laps': 58},
    'istanbul':     {'fuel_effect': 0.042, 'n_laps': 58},
}

DEFAULT_PARAMS = {'fuel_effect': 0.045, 'n_laps': 60}


def normalise(name):
    """
    Lowercases and strips the accents that make 'São Paulo' miss 'sao paulo'.
    Kept deliberately small: no external dependency, just the characters that
    actually show up in F1 event names.
    """
    key = str(name).lower()
    for src, dst in (('ã', 'a'), ('á', 'a'), ('à', 'a'), ('â', 'a'),
                     ('é', 'e'), ('è', 'e'), ('ê', 'e'),
                     ('í', 'i'), ('î', 'i'),
                     ('ó', 'o'), ('ô', 'o'), ('õ', 'o'),
                     ('ú', 'u'), ('ü', 'u'), ('ñ', 'n'), ('ç', 'c')):
        key = key.replace(src, dst)
    return key


def get_track_params(event_name, quiet=False):
    """Returns {'fuel_effect', 'n_laps'} for an event, falling back to defaults."""
    key = normalise(event_name)
    for track, params in TRACK_PARAMS.items():
        if normalise(track) in key:
            return dict(params)
    if not quiet:
        print(f"[track_params] '{event_name}' not found, using defaults.")
    return dict(DEFAULT_PARAMS)


def is_known(event_name):
    """True when the event matches a real entry rather than the defaults."""
    key = normalise(event_name)
    return any(normalise(t) in key for t in TRACK_PARAMS)


if __name__ == '__main__':
    names = ['Netherlands', 'Monza', 'Monaco', 'French Grand Prix',
             'São Paulo Grand Prix', 'Sao Paulo Grand Prix',
             'Mexico City Grand Prix', 'Unknown GP']
    for name in names:
        p = get_track_params(name, quiet=True)
        flag = '' if is_known(name) else '  <- default'
        print(f'{name:26s} -> {p}{flag}')