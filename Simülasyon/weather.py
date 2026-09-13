"""
F1 Prediction Simulation - v2.1 - weather.py
Rain, a track that takes time to wet and time to dry, and the tyre that suits it.

Two variables, not one
----------------------
Rain and a wet track are different things and collapsing them is the mistake
this module exists to avoid. Rain stopping does not dry the circuit, and rain
starting does not flood it - there is a lag in both directions, and that lag is
most of what makes a changeable race interesting. So a rainfall path is drawn
first, and a wetness index follows it with delay.

What is measured and what is not
--------------------------------
wet_conditions.py went looking, and the honest split is:

    measured    running on inters costs +13.5% of the lap against the same
                race's own dry median, over 3,167 green laps and 13 races;
                lap-to-lap spread is 3.1x the dry figure
    measured    full wets cost +25.2%, but over 237 laps and 3 races
    measured    an inter on a dry track costs about 13.8% of a lap, and a full
                wet beats an inter by about 0.6% of a lap in the wettest
                conditions anyone actually ran in - both from laps where cars
                were out on both categories at once, and both very thin
    measured    neutralizations start 2.24x as often per eligible lap in the
                wet, which is not the 1.8x the project has been carrying
    NOT         where the crossover actually sits. 32 lap-instants over 11
                races is not a curve, and no amount of fitting makes it one
    NOT         any physical scale for wetness. There is no water depth in the
                data and this index is not pretending to be one
    NOT         how a wet tyre wears. A drying track makes an inter quicker
                with age, which is the opposite sign to wear, and nothing here
                separates the two

So the shape between the anchors is a scenario. It is labelled a scenario in
the parameter panel, in the interface and here, and the anchors it is pinned
to carry their sample sizes with them.

The index
---------
0 is the dry reference and 1 is this model's very-wet reference. It is not
water depth and it is not a fraction of the track. 0.75 is roughly the wettest
condition the data contains; above that the model is extrapolating and says so.
"""

import numpy as np

# --- the rainfall path ------------------------------------------------------
# Intensity classes rather than millimetres. There is no measurement behind a
# millimetre figure and printing one would be an invented precision.
NONE, LIGHT, MODERATE, HEAVY = 0, 1, 2, 3
INTENSITY_NAMES = ('none', 'light', 'moderate', 'heavy')

# How much of the index one lap of each intensity adds, and what one lap of
# drying takes off. Scenario values: the roadmap's own worked example moves the
# index by about a tenth of its range per lap, and these sit around that.
# Drying runs every lap and is simply outrun by anything heavier than light.
WETTING_PER_LAP = (0.00, 0.10, 0.22, 0.40)
DRYING_PER_LAP = 0.06

# Where the measurements sit on the index. These two are what everything else
# is pinned to, so they are named rather than buried in a table.
W_INTER = 0.50        # conditions in which the field ran inters: +13.5%/lap
W_WET = 1.00          # this model's very-wet reference: +25.2%/lap
W_OBSERVED_MAX = 0.75  # beyond here the model is extrapolating

# --- categories -------------------------------------------------------------
# The dry three keep their indices, so every existing table and every existing
# strategy still means what it meant. The wet pair is appended.
DRY_COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')
CATEGORY_NAMES = DRY_COMPOUNDS + ('INTERMEDIATE', 'WET')
SOFT, MEDIUM, HARD, INTERMEDIATE, WET = range(5)
WET_CATEGORIES = (INTERMEDIATE, WET)


def is_wet_category(index):
    return np.asarray(index) >= INTERMEDIATE


# --- condition penalty ------------------------------------------------------
# What the afternoon costs everybody, whatever is fitted. Measured at two
# points against each race's own dry median, and straight between them; the
# straight line is the scenario, the two knots are not.
#
#     (0.00, 0)        dry, by definition
#     (0.50, 0.135)    3,167 green inter laps over 13 races
#     (1.00, 0.252)    237 green wet laps over 3 races - thin, and labelled
CONDITION_KNOTS = ((0.00, 0.000), (W_INTER, 0.135), (W_WET, 0.252))

# --- mismatch penalty -------------------------------------------------------
# What the wrong tyre costs on top, as a fraction of the lap.
#
# Two of these knots are measurements and the rest are not:
#
#   INTERMEDIATE at 0.00 = 0.138   an inter on a dry track was +12.44 s/lap
#                                  over 5 lap-instants. Thin, but it is the
#                                  strongest single number in the whole set
#                                  and the sign is not in doubt
#   INTERMEDIATE at 0.75 = 0.006   a full wet beat an inter by 0.49 s/lap in
#                                  the wettest conditions run, 40 lap-instants
#                                  over 7 races
#
# Everything else is shape. The dry column is the one with no measurement at
# all worth the name - nobody runs slicks in standing water long enough to be
# timed doing it - so it is set steep and its top end is openly a guess. It
# has to be steep: a model where a slick is merely slow in the wet will race
# one to the flag rather than pay for a stop.
MISMATCH_KNOTS = {
    'DRY': ((0.00, 0.000), (0.25, 0.015), (0.50, 0.100),
            (0.75, 0.280), (1.00, 0.500)),
    'INTERMEDIATE': ((0.00, 0.138), (0.25, 0.022), (0.50, 0.000),
                     (0.75, 0.006), (1.00, 0.030)),
    'WET': ((0.00, 0.280), (0.25, 0.090), (0.50, 0.017),
            (0.75, 0.000), (1.00, 0.000)),
}

# Above this a category is not being raced, it is being survived. A car on a
# tyre this wrong stops whatever the pit loss says - the alternative is a model
# that runs slicks through a downpour because twenty seconds looked expensive.
UNRACEABLE_PCT = 0.35

# --- lap-to-lap spread ------------------------------------------------------
# Measured, as multiples of the same race's dry figure. The wet does not just
# lower the average, it widens the distribution, and a model that only slowed
# everyone down would make a wet race more predictable rather than less.
SIGMA_MULTIPLIER = {'DRY': 1.0, 'INTERMEDIATE': 3.14, 'WET': 4.47}

# --- wear -------------------------------------------------------------------
# Assumptions, and flagged as such everywhere they surface. The measured age
# slope for an inter is negative because the track was drying underneath it,
# and there is nothing in lap data that separates a drying track from a wearing
# tyre. These are linear rates in seconds per lap of age.
WET_WEAR_PER_LAP = {INTERMEDIATE: 0.030, WET: 0.020}

# Longest stint each category may run. Not a measured life: the longest inter
# stint in the data is 57 laps and the p90 is 33, but that is the longest
# anyone needed on a drying track, not the longest the tyre had. Set near the
# p90 so a car cannot sit on one set through a whole wet race for free.
WET_STINT_CAP = {INTERMEDIATE: 40, WET: 25}

# --- neutralization ---------------------------------------------------------
# Measured: 23 events over 567 wet laps against 176 over 9,740 dry ones.
NEUTRAL_WET_RATIO = 2.24
# and the share of laps in the pooled measurement that were already wet, which
# is what stops the multiplier being applied on top of itself
NEUTRAL_WET_LAP_SHARE = 0.055

# --- wet racing -------------------------------------------------------------
# Overtaking in the wet, with opportunities in the denominator rather than
# successful passes alone. There is not enough of it to measure separately at
# this circuit, so one pooled adjustment is used and labelled.
WET_PASS_MULTIPLIER = 0.70     # scenario
WET_DIRTY_AIR_MULTIPLIER = 1.60  # spray is not dirty air, but it acts like it

# --- scenarios --------------------------------------------------------------
# Fixed paths first, sampling second. A scenario the reader can name is worth
# more here than a distribution nobody can check, and the acceptance tests all
# run against these rather than against a draw.
SCENARIOS = {
    'dry': 'No rain. The new component is inert and the race is the dry one.',
    'wet_throughout': 'Wet from the formation lap to the flag.',
    'drying': 'Wet start, rain already stopping, track dries through the race.',
    'rain_arrives': 'Dry start, rain from about the midpoint, stays.',
    'shower': 'Dry start, a burst of rain, then it stops and the track dries.',
    'light_shower': 'A passing drizzle that barely crosses the crossover.',
    'sampled': 'Start lap, length and intensity drawn per simulation.',
}


def _interp(knots, w):
    xs = np.array([k[0] for k in knots])
    ys = np.array([k[1] for k in knots])
    return np.interp(np.clip(np.asarray(w, dtype=float), 0.0, 1.0), xs, ys)


def condition_pct(w):
    """Fraction of the lap the conditions cost everybody, fitted tyre aside."""
    return _interp(CONDITION_KNOTS, w)


def mismatch_pct(category, w):
    """
    Fraction of the lap the fitted category costs on top, for how wet it is.

    `category` may be an array of indices; the three dry compounds share one
    column because their differences from each other are already carried by
    the measured compound offsets and would be counted twice here.
    """
    category = np.asarray(category)
    w = np.asarray(w, dtype=float)
    out = _interp(MISMATCH_KNOTS['DRY'], w) * np.ones_like(w, dtype=float)
    out = np.where(category == INTERMEDIATE,
                   _interp(MISMATCH_KNOTS['INTERMEDIATE'], w), out)
    out = np.where(category == WET, _interp(MISMATCH_KNOTS['WET'], w), out)
    return out


def sigma_multiplier(category):
    """How much wider lap times run on each category. Measured."""
    category = np.asarray(category)
    out = np.full(category.shape, SIGMA_MULTIPLIER['DRY'], dtype=float)
    out = np.where(category == INTERMEDIATE,
                   SIGMA_MULTIPLIER['INTERMEDIATE'], out)
    out = np.where(category == WET, SIGMA_MULTIPLIER['WET'], out)
    return out


def wet_wear(category, age):
    """
    Seconds a wet set has lost to age. Linear, assumed, and zero for dry
    compounds - those keep the fitted curves they already had.
    """
    category, age = np.asarray(category), np.asarray(age, dtype=float)
    out = np.zeros(np.broadcast(category, age).shape)
    for index, rate in WET_WEAR_PER_LAP.items():
        out = np.where(category == index, rate * age, out)
    return out


def best_category(w, dry_choice):
    """
    The category that costs least at this wetness, with `dry_choice` standing
    in for whichever dry compound the plan would fit.

    The condition penalty is common to all of them, so it cancels and this is
    a comparison of mismatch alone.
    """
    w = np.asarray(w, dtype=float)
    dry_choice = np.asarray(dry_choice)
    shape = np.broadcast(w, dry_choice).shape
    w = np.broadcast_to(w, shape)
    dry_choice = np.broadcast_to(dry_choice, shape)
    options = np.stack([mismatch_pct(dry_choice, w),
                        mismatch_pct(np.full(shape, INTERMEDIATE), w),
                        mismatch_pct(np.full(shape, WET), w)])
    pick = options.argmin(axis=0)
    return np.where(pick == 0, dry_choice,
                    np.where(pick == 1, INTERMEDIATE, WET))


def unraceable(category, w):
    """Whether this tyre is past the point of being raced at this wetness."""
    return mismatch_pct(category, w) > UNRACEABLE_PCT


# --- building a race's weather ---------------------------------------------

def _fixed_rain(scenario, n_laps):
    """The named scenarios, as an intensity per lap."""
    rain = np.zeros(n_laps, dtype=np.int8)
    mid = n_laps // 2
    if scenario == 'dry':
        pass
    elif scenario == 'wet_throughout':
        rain[:] = MODERATE
    elif scenario == 'drying':
        rain[:max(1, n_laps // 8)] = LIGHT
    elif scenario == 'rain_arrives':
        rain[mid:] = MODERATE
    elif scenario == 'shower':
        rain[n_laps // 4:n_laps // 4 + max(3, n_laps // 8)] = HEAVY
    elif scenario == 'light_shower':
        rain[n_laps // 3:n_laps // 3 + max(4, n_laps // 6)] = LIGHT
    else:
        raise KeyError(f'unknown weather scenario: {scenario}')
    return rain


def _initial_wetness(scenario):
    """
    How wet the track already is on lap one.

    A race that starts wet started wet; the rain fell before the formation lap
    and the index has to reflect that rather than climbing from zero while the
    field is already on inters.
    """
    return {'wet_throughout': 0.75, 'drying': 0.70}.get(scenario, 0.0)


def advance(w, intensity):
    """
    One lap of the track getting wetter or drier.

    Drying runs every lap and is simply outrun by anything above light rain,
    which is what gives both lags: the track keeps filling for a while after
    the rain eases, and keeps drying for a while after it starts.
    """
    gain = np.asarray(WETTING_PER_LAP)[np.asarray(intensity)]
    return np.clip(np.asarray(w, dtype=float) + gain - DRYING_PER_LAP, 0.0, 1.0)


def build_paths(scenario, n_sims, n_laps, rng):
    """
    One rainfall path and one wetness path per simulation.

    Every car in a simulation sees the same weather, which is the point: rain
    is not a per-car random effect, and giving each car its own would make a
    wet race a lottery rather than a shared problem.

    `rng` is the weather's own generator, not the race's. That is what makes a
    dry scenario exactly inert: it consumes none of the race's draws, so v2.1
    can be switched on over a dry afternoon and give the old race back bit for
    bit.

    It does not make two wet scenarios comparable draw for draw, and it would
    be wrong to claim so. A wet race neutralises 2.24x as often per lap, so the
    safety-car sample legitimately differs the moment it rains and everything
    after it follows. That is the model working, not the weather stealing the
    race's dice.
    """
    rain = np.zeros((n_sims, n_laps), dtype=np.int8)
    start = np.zeros(n_sims)

    if scenario == 'sampled':
        # Half the races stay dry. The rest get a burst somewhere in the
        # middle two-thirds, with a length and an intensity of their own.
        wet_race = rng.random(n_sims) < 0.50
        begin = (rng.integers(n_laps // 6, max(n_laps // 6 + 1,
                                               5 * n_laps // 6), size=n_sims))
        length = rng.integers(3, max(4, n_laps // 2), size=n_sims)
        strength = rng.integers(LIGHT, HEAVY + 1, size=n_sims)
        for sim in np.flatnonzero(wet_race):
            a = int(begin[sim])
            b = min(n_laps, a + int(length[sim]))
            rain[sim, a:b] = strength[sim]
        # a race that is raining on lap one was raining before it
        start = np.where(rain[:, 0] > 0, 0.70, 0.0)
    else:
        rain[:] = _fixed_rain(scenario, n_laps)
        start[:] = _initial_wetness(scenario)

    wetness = np.zeros((n_sims, n_laps))
    w = start
    for lap in range(n_laps):
        w = advance(w, rain[:, lap])
        wetness[:, lap] = w
    return rain, wetness


def predict(w_now, rain_now, n_steps):
    """
    What the track is expected to do next, from what can be seen now.

    The assumption is the one the roadmap asks for and the only one available:
    whatever is falling now keeps falling for the moment. It is wrong as often
    as the weather changes, and it is not quietly corrected against the path
    the simulator drew - a decision that could see the real future would not be
    a decision.
    """
    w = np.asarray(w_now, dtype=float)
    out = np.empty(w.shape + (n_steps,))
    for step in range(n_steps):
        w = advance(w, rain_now)
        out[..., step] = w
    return out


def neutral_lambda(pooled, wet_share):
    """
    This race's neutralization rate, given how much of it is wet.

    The measured rate already pools wet and dry races, so multiplying it by the
    wet ratio would charge part of the wet twice. The dry-only rate is backed
    out of the pooled one first, using the share of laps in the measurement
    that were wet, and the ratio is applied to that.

    There is no second safety-car process here. This scales the one that
    already exists, which is what keeps a wet race from having two independent
    sources of yellow flags overlapping each other.
    """
    wet_share = np.asarray(wet_share, dtype=float)
    dry_only = pooled / (1.0 + NEUTRAL_WET_LAP_SHARE * (NEUTRAL_WET_RATIO - 1))
    return dry_only * (1.0 + wet_share * (NEUTRAL_WET_RATIO - 1))


def provenance():
    """
    Which of these numbers were measured and which were chosen.

    This is the table the interface shows. It exists because a scenario that
    looks like a measurement is worse than no scenario at all, and the reader
    cannot tell them apart from the output alone.
    """
    return [
        ('Inter pace penalty', 'measured', '+13.5% of the lap',
         '3,167 green laps, 13 races'),
        ('Wet pace penalty', 'thin', '+25.2% of the lap',
         '237 green laps, 3 races'),
        ('Inter lap-time spread', 'measured', f'{SIGMA_MULTIPLIER["INTERMEDIATE"]:.1f}x dry',
         'same laps'),
        ('Wet lap-time spread', 'thin', f'{SIGMA_MULTIPLIER["WET"]:.1f}x dry',
         'same laps'),
        ('Inter on a dry track', 'thin', f'+{MISMATCH_KNOTS["INTERMEDIATE"][0][1]:.1%} of the lap',
         '5 lap-instants where both were running'),
        ('Wet beats inter, very wet', 'thin',
         f'{MISMATCH_KNOTS["INTERMEDIATE"][3][1]:.1%} of the lap',
         '40 lap-instants, 7 races'),
        ('Neutralisation in the wet', 'measured', f'{NEUTRAL_WET_RATIO:.2f}x per lap',
         '23 events / 567 wet laps vs 176 / 9,740 dry'),
        ('Crossover shape', 'scenario', 'straight between the anchors',
         '32 lap-instants is not a curve'),
        ('Wetness index', 'scenario', '0 dry to 1 very wet',
         'no water measurement exists in the data'),
        ('Wetting and drying rates', 'scenario',
         f'+{WETTING_PER_LAP[MODERATE]:.2f}/lap moderate, -{DRYING_PER_LAP:.2f}/lap',
         'chosen so the lag is visible, never measured'),
        ('Slick in the wet', 'scenario',
         f'+{MISMATCH_KNOTS["DRY"][2][1]:.0%} of the lap at half wet',
         'nobody runs one long enough to be timed'),
        ('Wet tyre wear', 'assumption',
         f'{WET_WEAR_PER_LAP[INTERMEDIATE]:.3f} s per lap of age',
         'a drying track and a wearing tyre have opposite signs'),
        ('Wet stint cap', 'assumption',
         f'{WET_STINT_CAP[INTERMEDIATE]} laps inter, {WET_STINT_CAP[WET]} wet',
         'longest run was 57 and 25, set by weather not by the tyre'),
        ('Wet overtaking', 'scenario', f'{WET_PASS_MULTIPLIER:.2f}x pass odds',
         'not enough wet racing at this circuit to measure'),
    ]


# --- the decision to change category ---------------------------------------
# Reuses v2.0's traffic kernels rather than growing a second set. A pass in the
# rain is still a pass, and two models of the same thing drift apart.
from Simülasyon.reactive_strategy import _follow_steps, _chased_steps  # noqa: E402

# How far ahead a weather change is priced. Ten laps is the roadmap's starting
# preference and it is not a measured optimum; the agreement between 5, 10 and
# 15 is reported so the reader can see how much it matters.
SWITCH_HORIZON = 10
HORIZON_CHECKS = (5, 10, 15)

# Laps of the remaining stint charged beyond the horizon, so that waiting
# cannot win by hiding the consequence just past the edge of the window.
TERMINAL_LAPS = 5

# Laps of traffic priced after rejoining, matching v2.0's window.
TRAFFIC_LAPS = 3

REASON_STAY, REASON_SWITCH, REASON_WAIT, REASON_UNRACEABLE = range(4)
SWITCH_REASONS = ('stay', 'weather_switch', 'wait one lap',
                  'current tyre unraceable')


def _run_cost(category, w_path, age0, lap_time, n, cum=None):
    """
    Seconds a car loses over `n` laps on one category: the conditions, the
    wrong tyre for them, and the wear the set has already collected.

    Wear comes from the accumulated table the race itself runs on, which covers
    all five categories. An earlier version priced only the two wet rows and
    left the dry curves out, which quietly deleted half the reason to give up a
    worn slick: staying out was charged for being on the wrong tyre and never
    for being on an old one.

    Age keeps running across a category change in the caller, never here: a set
    that did fifteen laps on a drying track has done fifteen laps, and the wear
    it collected does not come back when it starts raining again.
    """
    cost = np.zeros(np.broadcast(category, age0).shape)
    for h in range(n):
        w = w_path[..., h]
        cost = cost + lap_time * (condition_pct(w) + mismatch_pct(category, w))
        if cum is None:
            cost = cost + wet_wear(category, age0 + h + 1)
            cost = cost - wet_wear(category, age0 + h)
        else:
            top = np.clip(age0 + h + 1, 0, cum.shape[1] - 1).astype(int)
            bottom = np.clip(age0 + h, 0, cum.shape[1] - 1).astype(int)
            cost = cost + cum[category, top] - cum[category, bottom]
    return cost


def decide_switch(st, horizon=SWITCH_HORIZON, traffic=True):
    """
    Whether to change tyre category for the weather, and to what.

    Three options, which is what the roadmap asks for: carry on, change now, or
    wait a lap and reconsider. Waiting is priced with the stop it implies on
    the next lap rather than with no stop at all, because an option that parks
    its cost outside the window always wins and never should.

    The forecast is the one a pit wall has: whatever is falling now keeps
    falling. It is not checked against the path the simulator drew.
    """
    w_now, rain_now = st['wetness'], st['rain']
    fitted, age = st['fitted'], st['age']
    lap_time, pit_loss = st['lap_time'], st['pit_loss']
    laps_left = st['laps_left']

    cum = st.get('cum')
    n = int(min(horizon, laps_left))
    if n <= 0:
        shape = np.shape(fitted)
        return (np.zeros(shape, dtype=bool), fitted,
                np.full(shape, REASON_STAY, dtype=np.int8))

    forecast = predict(w_now, rain_now, n + TERMINAL_LAPS)
    target = best_category(forecast[..., min(1, n - 1)], st['dry_choice'])

    # --- carry on ---
    stay = _run_cost(fitted, forecast, age, lap_time, n, cum)

    # --- change now, and change next lap ---
    options, costs = [], []
    for k in (0, 1):
        if k >= n:
            continue
        before = _run_cost(fitted, forecast, age, lap_time, k, cum) if k else 0.0
        after = _run_cost(target, forecast[..., k:], np.zeros_like(age),
                          lap_time, n - k, cum)
        costs.append(before + pit_loss + after)
        options.append(k)

    # --- what happens just past the edge of the window ---
    # Without this, waiting collects the benefit of a fresh tyre inside the
    # window and leaves its cost one lap outside it.
    tail = int(min(TERMINAL_LAPS, max(laps_left - n, 0)))
    if tail:
        stay = stay + _run_cost(fitted, forecast[..., n:], age + n,
                                lap_time, tail, cum)
        for i, k in enumerate(options):
            costs[i] = costs[i] + _run_cost(target, forecast[..., n:],
                                            np.full_like(age, n - k),
                                            lap_time, tail, cum)

    if traffic and st.get('exit_gap') is not None:
        stay = stay + st['stay_traffic']
        for i in range(len(costs)):
            costs[i] = costs[i] + st['switch_traffic']

    stack = np.stack([stay] + costs)
    pick = stack.argmin(axis=0)

    # A tyre this wrong is not a choice any more. Pit loss does not enter it:
    # the alternative is a model that races slicks through a downpour because
    # twenty seconds looked expensive.
    forced = unraceable(fitted, w_now) & (target != fitted)
    switch_now = ((pick == 1) & (len(options) > 0)) | forced

    reason = np.where(forced, REASON_UNRACEABLE,
                      np.where(pick == 0, REASON_STAY,
                               np.where(pick == 1, REASON_SWITCH,
                                        REASON_WAIT))).astype(np.int8)
    return switch_now, target, reason


def horizon_agreement(st):
    """
    How much the answer depends on where the window was drawn.

    The roadmap asks for this rather than for a defence of ten laps, and it is
    the right thing to ask: a decision that flips between a five and a fifteen
    lap horizon is being made by the horizon.
    """
    picks = [decide_switch(st, horizon=h, traffic=False)[0]
             for h in HORIZON_CHECKS]
    same = np.ones(np.shape(picks[0]), dtype=bool)
    for other in picks[1:]:
        same &= other == picks[0]
    return float(np.mean(same)) if same.size else 1.0


def traffic_terms(ctx, t_self, pace_self, pace_fresh, exit_time, totals, paces,
                  dead, front, rear, live_front, live_rear, shift, shift_rear,
                  self_index, lap_len):
    """
    What staying out and what changing cost in traffic, on v2.0's terms.

    Same four cars, same kernels, same expected values - a pass in the rain is
    still a pass, and a second model of the same thing would drift away from
    the first. The only difference is that there is one candidate here rather
    than eleven, so the rejoin point is found once.
    """
    rows = np.arange(t_self.size)
    laps = np.full(t_self.size, float(TRAFFIC_LAPS))

    gap_front = np.maximum(t_self - totals[rows, front], 0.0)
    adv_front = paces[rows, front] - pace_self
    gap_rear = np.maximum(totals[rows, rear] - t_self, 0.0)
    adv_rear = pace_self - paces[rows, rear]

    stay = (np.where(live_front, _follow_steps(ctx, gap_front, adv_front,
                                               shift, TRAFFIC_LAPS), 0.0)
            + np.where(live_rear, _chased_steps(ctx, gap_rear, adv_rear,
                                                shift_rear, TRAFFIC_LAPS), 0.0))

    behind = exit_time[:, None] - (totals + paces)
    usable = ~dead & (np.abs(behind) < lap_len[:, None])
    usable[rows, self_index] = False
    ahead = np.where(usable & (behind > 0.0), behind, np.inf)
    back = np.where(usable & (behind < 0.0), -behind, np.inf)
    i_ahead, i_rear = ahead.argmin(axis=1), back.argmin(axis=1)
    g_ahead, g_rear = ahead[rows, i_ahead], back[rows, i_rear]
    got_ahead, got_rear = np.isfinite(g_ahead), np.isfinite(g_rear)

    switch = (np.where(got_ahead,
                       _follow_steps(ctx, np.where(got_ahead, g_ahead, 0.0),
                                     paces[rows, i_ahead] - pace_fresh,
                                     shift, TRAFFIC_LAPS), 0.0)
              + np.where(got_rear,
                         _chased_steps(ctx, np.where(got_rear, g_rear, 0.0),
                                       pace_fresh - paces[rows, i_rear],
                                       shift, TRAFFIC_LAPS), 0.0))
    return stay, switch
