"""
F1 Prediction Simulation - tracks.py
Overtaking difficulty and dirty air penalty per circuit.

overtake_difficulty  1-5, how hard it is to pass a car of similar pace.
                     1 = long straights and heavy braking zones (Monza, Spa)
                     5 = no realistic passing place at all (Monaco)

dirty_air_penalty    seconds per lap lost while glued to the car ahead. The
                     penalty now fades linearly with the gap, reaching zero at
                     DIRTY_AIR_RANGE, rather than switching on and off.

overtake_difficulty is now measured, not guessed: overtaking.py counts on-track
passes conditioned on the follower actually being quicker, and simulate.py reads
the resulting per-lap pass probability straight from data/overtaking.csv. The
ratings below are only a fallback for circuits with no measurement yet.
"""

# --- tuning -----------------------------------------------------------------

DIRTY_AIR_RANGE = 2.5          # seconds; the penalty fades linearly to zero here
MIN_GAP = 0.35                 # seconds a blocked car is held behind the one ahead

# HELD_GAP replaces MIN_GAP as the distance a car that failed to pass settles
# at, and unlike MIN_GAP it is measured rather than chosen.
#
# Among 3,492 laps of 2022-25 where a quicker car was inside a second and did
# not get through, the gap it actually sat at was:
#
#     p10 0.378   p25 0.504   p50 0.659   p75 0.799   p90 0.915   mean 0.646
#
# Only 7% of them were anywhere near 0.35 s; 88% were further back than that.
# Pinning every blocked car at 0.35 put the whole field in the one place real
# blocked cars are not.
#
# It did not matter while pass probability ignored the gap - the number was
# only feeding dirty air. It matters now: simulate.py prices a lap by how
# close the follower is, and a car parked at 0.35 s collects a +1.02 logit
# bonus every lap, which turns a 0.222 base rate into 0.442 and doubles the
# passes. The gap term was measured on real distances, so the simulation has
# to hold cars at real distances for it to mean anything.
HELD_GAP = 0.65
ATTACK_GAP = 1.00              # seconds; inside this the follower is attacking

# ATTACK_GAP has to sit above MIN_GAP. When the two were the same value a car
# that failed to pass was pinned at exactly MIN_GAP behind, which no longer
# counted as close, so it never attacked again and the whole field froze in grid
# order for the rest of the race.

# fallback pass probability per lap when overtaking.csv has no row for a track
FALLBACK_PASS_RATE = 0.18

# --- ratings ----------------------------------------------------------------
# key: lowercase substring matched against the event name

TRACKS = {
    'bahrain':       {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'saudi':         {'overtake_difficulty': 2, 'dirty_air_penalty': 0.20},
    'jeddah':        {'overtake_difficulty': 2, 'dirty_air_penalty': 0.20},
    'australian':    {'overtake_difficulty': 3, 'dirty_air_penalty': 0.30},
    'melbourne':     {'overtake_difficulty': 3, 'dirty_air_penalty': 0.30},
    'japanese':      {'overtake_difficulty': 3, 'dirty_air_penalty': 0.35},
    'suzuka':        {'overtake_difficulty': 3, 'dirty_air_penalty': 0.35},
    'chinese':       {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'shanghai':      {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'miami':         {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'emilia':        {'overtake_difficulty': 4, 'dirty_air_penalty': 0.35},
    'imola':         {'overtake_difficulty': 4, 'dirty_air_penalty': 0.35},
    'monaco':        {'overtake_difficulty': 5, 'dirty_air_penalty': 0.45},
    'monte carlo':   {'overtake_difficulty': 5, 'dirty_air_penalty': 0.45},
    'spanish':       {'overtake_difficulty': 3, 'dirty_air_penalty': 0.40},
    'barcelona':     {'overtake_difficulty': 3, 'dirty_air_penalty': 0.40},
    'canadian':      {'overtake_difficulty': 2, 'dirty_air_penalty': 0.20},
    'montreal':      {'overtake_difficulty': 2, 'dirty_air_penalty': 0.20},
    'austrian':      {'overtake_difficulty': 1, 'dirty_air_penalty': 0.20},
    'spielberg':     {'overtake_difficulty': 1, 'dirty_air_penalty': 0.20},
    'british':       {'overtake_difficulty': 2, 'dirty_air_penalty': 0.35},
    'silverstone':   {'overtake_difficulty': 2, 'dirty_air_penalty': 0.35},
    'hungarian':     {'overtake_difficulty': 4, 'dirty_air_penalty': 0.45},
    'hungaroring':   {'overtake_difficulty': 4, 'dirty_air_penalty': 0.45},
    'belgian':       {'overtake_difficulty': 1, 'dirty_air_penalty': 0.25},
    'spa':           {'overtake_difficulty': 1, 'dirty_air_penalty': 0.25},
    'dutch':         {'overtake_difficulty': 4, 'dirty_air_penalty': 0.40},
    'netherlands':   {'overtake_difficulty': 4, 'dirty_air_penalty': 0.40},
    'zandvoort':     {'overtake_difficulty': 4, 'dirty_air_penalty': 0.40},
    'italian':       {'overtake_difficulty': 1, 'dirty_air_penalty': 0.15},
    'monza':         {'overtake_difficulty': 1, 'dirty_air_penalty': 0.15},
    'azerbaijan':    {'overtake_difficulty': 1, 'dirty_air_penalty': 0.15},
    'baku':          {'overtake_difficulty': 1, 'dirty_air_penalty': 0.15},
    'singapore':     {'overtake_difficulty': 4, 'dirty_air_penalty': 0.40},
    'united states': {'overtake_difficulty': 2, 'dirty_air_penalty': 0.30},
    'austin':        {'overtake_difficulty': 2, 'dirty_air_penalty': 0.30},
    'mexico':        {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'brazilian':     {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'interlagos':    {'overtake_difficulty': 2, 'dirty_air_penalty': 0.25},
    'las vegas':     {'overtake_difficulty': 1, 'dirty_air_penalty': 0.15},
    'qatar':         {'overtake_difficulty': 3, 'dirty_air_penalty': 0.35},
    'lusail':        {'overtake_difficulty': 3, 'dirty_air_penalty': 0.35},
    'abu dhabi':     {'overtake_difficulty': 3, 'dirty_air_penalty': 0.30},
    'yas':           {'overtake_difficulty': 3, 'dirty_air_penalty': 0.30},
}

DEFAULT_ENTRY = {'overtake_difficulty': 3, 'dirty_air_penalty': 0.30}


def get_track_racing(event_name):
    """
    Returns dict with:
        overtake_difficulty   1-5
        overtake_threshold    s/lap of pace advantage needed to pass
        dirty_air_penalty     s/lap lost while following closely
    """
    key = event_name.lower()
    entry = None
    for track, params in TRACKS.items():
        if track in key:
            entry = params
            break

    if entry is None:
        print(f"[tracks] '{event_name}' not found, using defaults.")
        entry = DEFAULT_ENTRY

    difficulty = entry['overtake_difficulty']
    # only used when overtaking.csv has nothing for this circuit
    fallback_rate = FALLBACK_PASS_RATE * (1.6 - 0.25 * difficulty)
    return {
        'overtake_difficulty': difficulty,
        'fallback_pass_rate': round(max(fallback_rate, 0.02), 3),
        'dirty_air_penalty': entry['dirty_air_penalty'],
    }


if __name__ == '__main__':
    for name in ['Monza', 'Netherlands', 'Monaco', 'Spa', 'Unknown GP']:
        r = get_track_racing(name)
        print(f"{name:14s} diff {r['overtake_difficulty']}  "
              f"fallback p_pass {r['fallback_pass_rate']:.3f}  "
              f"dirty air {r['dirty_air_penalty']:.2f} s/lap")