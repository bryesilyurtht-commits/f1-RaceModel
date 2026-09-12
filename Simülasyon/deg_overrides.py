"""
F1 Prediction Simulation - deg_overrides.py
Hand-set substitutions for tracks whose degradation cannot be measured.

Only the tyre degradation curve is borrowed. Fuel burn, race distance, pit
loss, overtaking and every other track parameter stay the track's own - those
are measured elsewhere and are not in question here.

Why this file exists
--------------------
tyre_profile.py measures degradation from within-lap contrasts: at a given
lap, one car is on an old tyre and another on a fresh one. Some tracks never
produce that contrast in useful amounts - short races, few clean laps, wet or
safety-car heavy history, or a field that pits in one narrow window. The fit
still returns a number, and the number is close to zero because there was
nothing to estimate from.

Silverstone is the case that forced this. It came out among the flattest
tracks on the calendar, which contradicts everything known about it: high
lateral load, abrasive surface, a track that historically eats tyres. Spa is
the closest match on the characteristics that drive wear, so its curve is
borrowed.

This is a hand-entered value, not a measurement. It belongs on the v2.3 list
of constants to be replaced by something measured. The provenance column in
the final table must say so, otherwise a borrowed curve reads as evidence.
"""

# receiver track -> donor track, matched on the canonical event name
DEG_DONOR = {
    'British Grand Prix': 'Belgian Grand Prix',
}

# free text, carried into the output so the choice is auditable later
DONOR_REASON = {
    'British Grand Prix': 'measured slope implausibly flat; few clean laps, '
                          'short race, wet/SC heavy history. Spa matched on '
                          'lateral load and abrasion.',
}


def donor_for(race_name):
    """Returns the donor track name, or None when the track uses its own curve."""
    return DEG_DONOR.get(race_name)


def resolve(race_name):
    """
    Returns (source_track, provenance).

    provenance is 'measured' when the track keeps its own curve and
    'borrowed:<donor>' when it does not.
    """
    donor = DEG_DONOR.get(race_name)
    if donor is None:
        return race_name, 'measured'
    return donor, f'borrowed:{donor}'


def reason_for(race_name):
    return DONOR_REASON.get(race_name, '')


if __name__ == '__main__':
    for name in ['British Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix']:
        src, prov = resolve(name)
        print(f'{name:24s} -> {src:24s} [{prov}]')
        if reason_for(name):
            print(f'{"":24s}    {reason_for(name)}')