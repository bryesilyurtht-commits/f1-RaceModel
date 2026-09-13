"""
F1 Prediction Simulation - v2.0 - reactive_strategy.py
The pit-timing decision, priced against rivals and pit-exit traffic.

What v1.1 did
-------------
A car inside a band around its planned stop asked one question every lap: does
the next lap on this tyre cost more than the lap it would displace on a fresh
one? That is a good question and it is entirely about the car itself. Two cars
on identical tyres at identical ages stopped on the same lap whether one of
them was about to rejoin into clear air and the other into the middle of a
train.

What v2.0 adds
--------------
The same stop, chosen by comparing the alternatives. Inside the window the car
enumerates every pit lap still available to it, prices each one to a horizon
all of them share, and takes the cheapest. The price has three parts:

    tyre        laps on the old set, then laps on the new one, read off the
                measured curves - the v1.1 question, kept
    pit loss    the real one for each candidate, not cancelled: a stop under a
                safety car now and a stop at green in four laps are not the
                same stop
    traffic     up to four rivals, in expectation, over a bounded horizon

Nothing here draws a random number. Every term is an expected value computed
from state the car can already see, which is what lets v2.0 be switched off and
reproduce v1.1 bit for bit under the same seed.

What the car is allowed to know
-------------------------------
Lap times that have happened, current gaps and order, compounds and tyre ages,
stops that have been made, the flag that is out right now, and its own plan.

Not: a rival's assigned strategy, the pit lap it will choose, the safety car
schedule the simulator drew before the race, or any lap time not yet run. The
decision layer is handed arrays containing none of those, which is a stronger
guarantee than a promise not to look at them.

Known limitation, stated rather than hidden
-------------------------------------------
The engine carries cumulative race time, not position around the lap. Two cars
a lap apart have a large time gap and can be side by side on the road. Pit-exit
neighbours are therefore found in time, and a rival further away than one lap
time is dropped rather than guessed at. Where the leaders lap the tail this
under-counts traffic for the cars being lapped. Fixing it needs a track-position
model, which is not this version's work.
"""

import numpy as np


# How many laps of detailed traffic are priced after a candidate stop. This is
# a computation boundary, not a measurement: three laps is long enough for an
# undercut to be caught or cleared and short enough that it does not depend on
# guessing when a rival will stop. Beyond it only the car's own tyre and pace
# run on to the common horizon.
TRAFFIC_LAPS = 3

# The car ahead now, the car behind now, and the two it would rejoin between.
# A fifth would need a rival's plan to be worth anything.
MAX_RIVALS = 4

# Floating-point slack, not a decision threshold. This version has no "must
# gain at least a second" rule: ties go to the candidate nearest the original
# plan, and that is the whole tie-break.
TIE_TOLERANCE = 1e-9

# Why the chosen lap is not the planned one. These name which term moved the
# answer. They are not claims about how the race then went.
REASON_NAMES = ('on plan', 'tyre, earlier', 'tyre, later',
                'traffic, earlier', 'traffic, later', 'no valid candidate')
(ON_PLAN, TYRE_EARLY, TYRE_LATE,
 TRAFFIC_EARLY, TRAFFIC_LATE, NO_CANDIDATE) = range(6)

# Intent labels. An early stop while the car ahead is still out is an undercut
# attempt; whether it worked needs the rival's stop to have happened and is
# deliberately not recorded here.
INTENT_NAMES = ('none', 'undercut', 'overcut')
NO_INTENT, UNDERCUT, OVERCUT = range(3)

# How recently a rival must have stopped for staying out to read as an overcut
OVERCUT_MEMORY = 3


class Context:
    """
    Everything the decision needs that does not change from lap to lap.

    Built once per race, before the first lap. Holding the tyre table and the
    pass model here is what keeps the per-lap work down to array arithmetic.
    `pass_prob` arrives as a callable so this module never imports simulate.
    """

    def __init__(self, n_laps, cum_loss, cap_by_compound, offsets, pit_loss,
                 pass_prob, pass_model, dirty_max, dirty_range, attack_gap,
                 held_gap, band, traffic_laps=TRAFFIC_LAPS,
                 max_rivals=MAX_RIVALS):
        self.n_laps = int(n_laps)
        self.cum = cum_loss
        self.max_age = cum_loss.shape[1] - 1
        self.cap = np.asarray(cap_by_compound, dtype=float)
        self.off = np.asarray(offsets, dtype=float)
        self.pit_loss = float(pit_loss)
        self.pass_prob = pass_prob
        self.pass_model = pass_model
        self.dirty_max = float(dirty_max)
        self.dirty_range = float(dirty_range)
        self.attack_gap = float(attack_gap)
        self.held_gap = float(held_gap)
        self.band = int(band)
        self.traffic_laps = int(traffic_laps)
        self.max_rivals = int(max_rivals)
        self.n_cand = 2 * int(band) + 1

    def cum_at(self, compound, age):
        """Accumulated tyre loss, with the age clamped to the fitted range."""
        return self.cum[compound, np.clip(age, 0, self.max_age)]


def _advance(ctx, g, advantage, close):
    """Where the gap is next lap: settled if held, closing if not."""
    closing = np.where(advantage > 0.0,
                       np.maximum(g - advantage, ctx.attack_gap),
                       g - advantage)
    return np.where(close, np.minimum(g, ctx.held_gap), closing)


def _follow_steps(ctx, gap, advantage, shift, n_steps, trace=False):
    """
    Expected seconds lost sitting behind one car, lap by lap.

    The two costs are mutually exclusive by construction, which is what keeps a
    held-up lap from being charged twice. Inside ATTACK_GAP the car is
    fighting: the engine pins it to the car ahead, so what it loses is exactly
    the pace advantage it cannot spend. Between ATTACK_GAP and DIRTY_AIR_RANGE
    it is not fighting yet and pays the dirty-air penalty the engine would give
    it. Beyond that, nothing.

    A pass is not free either. The same per-lap probability the race rolls is
    used as a probability here, so a car with a 30% chance of getting through is
    charged 70% of another held lap rather than waved past.

    With `trace`, returns the running total after each lap instead of the
    final one. Every candidate's pre-stop phase is a prefix of the same chase -
    the car ahead does not change because the stop is planned for later - so
    one pass answers all eleven of them.

    gap        seconds the car is behind the rival, positive
    advantage  its pace advantage over the rival, s/lap; positive means catching
    """
    gap = np.asarray(gap, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    shape = np.broadcast(gap, advantage).shape
    cost = np.zeros(shape)
    still_behind = np.ones(shape)
    g = np.broadcast_to(gap, shape).astype(float).copy()
    steps = np.zeros(shape + (n_steps,)) if trace else None

    for t in range(n_steps):
        close = (g <= ctx.attack_gap) & (g >= 0.0)
        p = np.where(close, ctx.pass_prob(advantage, ctx.pass_model,
                                          gap=np.clip(g, 0.0, None),
                                          team_shift=shift), 0.0)
        dirty = ctx.dirty_max * np.clip(1.0 - g / ctx.dirty_range, 0.0, 1.0)
        per_lap = np.where(close, np.maximum(advantage, 0.0), dirty)

        cost = cost + still_behind * per_lap
        if trace:
            steps[..., t] = cost

        g = _advance(ctx, g, advantage, close)
        still_behind = still_behind * np.where(close, 1.0 - p, 1.0)

    out = steps if trace else cost
    return np.where(np.isfinite(out), out, 0.0)


def _chased_steps(ctx, gap, advantage, shift, n_steps, trace=False):
    """
    What a car closing from behind actually costs, which is almost nothing.

    Someone in the mirrors is not a time loss and is not priced as one. The
    cost appears only after the pass: the car is then the follower and pays
    dirty air for the laps the rival takes to pull clear. That is the
    "subsequent traffic effect" and the whole of it - no points are awarded for
    holding a position, because this version prices time and nothing else.

    The tail is capped at TRAFFIC_LAPS however long the window is. Being
    briefly stuck in someone's wake is a short-lived thing, and extending it
    over a dozen laps would be inventing a cost out of a modelling convenience.

    gap        seconds the rival is behind, positive
    advantage  the rival's pace advantage, s/lap; positive means it is catching
    """
    gap = np.asarray(gap, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    shape = np.broadcast(gap, advantage).shape
    cost = np.zeros(shape)
    still_behind = np.ones(shape)
    g = np.broadcast_to(gap, shape).astype(float).copy()
    steps = np.zeros(shape + (n_steps,)) if trace else None

    for t in range(n_steps):
        close = (g <= ctx.attack_gap) & (g >= 0.0)
        p = np.where(close, ctx.pass_prob(advantage, ctx.pass_model,
                                          gap=np.clip(g, 0.0, None),
                                          team_shift=shift), 0.0)

        # Rebuilt each step rather than accumulated once. Accumulating it is
        # bit-identical and strictly less arithmetic, and it was tried: over
        # three interleaved pairs of 10,000-simulation runs it came out 1.4%
        # faster, against a run-to-run spread of a third. That is not a
        # measured gain, and v2.4's rule is that an optimisation which does
        # not show up in a measurement does not ship.
        tail = np.zeros(shape)
        for u in range(min(ctx.traffic_laps, n_steps - t - 1)):
            d = ctx.held_gap + u * np.maximum(advantage, 0.0)
            tail = tail + ctx.dirty_max * np.clip(1.0 - d / ctx.dirty_range,
                                                  0.0, 1.0)
        cost = cost + still_behind * p * tail
        if trace:
            steps[..., t] = cost

        g = _advance(ctx, g, advantage, close)
        still_behind = still_behind * np.where(close, 1.0 - p, 1.0)

    out = steps if trace else cost
    return np.where(np.isfinite(out), out, 0.0)


def decide(ctx, lap, st, log_sim=None):
    """
    One lap's pit decisions for every car currently inside a decision window.

    `st` carries only observable state. Every car's decision is computed from
    the same snapshot and none of them is applied here, so a car sitting
    earlier in the arrays cannot influence one sitting later: the ordering
    invariance the roadmap asks for is a property of the shape of this function
    rather than something bolted on and tested for afterwards.

    Returns `pit_now`, the lap chosen, reason and intent codes, and - for one
    nominated simulation only - the arithmetic behind each call.
    """
    sim, drv = st['sim'], st['drv']
    n_sel = sim.size
    if n_sel == 0:
        return {
            'pit_now': np.zeros(0, dtype=bool),
            'chosen_lap': np.zeros(0, dtype=np.int64),
            'planned_lap': np.zeros(0, dtype=np.int64),
            'reason': np.zeros(0, dtype=np.int8),
            'intent': np.zeros(0, dtype=np.int8),
            'valid': np.zeros(0, dtype=bool),
            'log': [],
        }

    lap_no = lap + 1                      # 1-based, the way pit laps are kept
    cand = np.arange(ctx.n_cand)          # 0 = now, 1 = next lap, ...
    rows = np.arange(n_sel)

    total, pace, retired = st['total'], st['pace'], st['retired']
    order, pos_of = st['order'], st['pos']
    team_shift = st['team_shift']

    t_self = total[sim, drv]
    pace_self = pace[sim, drv]
    age = st['age'][sim, drv].astype(np.int64)
    c_old = st['c_old'][sim, drv].astype(np.int64)
    c_new = st['c_new'][sim, drv].astype(np.int64)
    planned = st['planned'][sim, drv].astype(np.int64)
    horizon = st['horizon'][sim, drv].astype(np.int64)
    pit_now_cost = st['pit_cost_now'][sim, drv]
    shift = team_shift[drv]

    pit_lap = lap_no + cand[None, :]                      # (n_sel, n_cand)

    # --- which candidates are even legal ---------------------------------
    # The plan survives this version: the number of stops and the compound
    # order are fixed and only the lap moves. A candidate that would make the
    # rest of that plan unrunnable - a stint past what the curve was measured
    # over, a stop colliding with the next planned one, a stint of zero laps -
    # is dropped here rather than chosen and then rescued by the cap rule,
    # which is how a car ends up taking a stop nobody planned.
    old_len = age[:, None] + cand[None, :]
    new_len = horizon[:, None] - pit_lap

    ok = pit_lap <= (planned[:, None] + ctx.band)
    ok &= new_len >= 1
    ok &= old_len <= ctx.cap[c_old][:, None]
    ok &= new_len <= ctx.cap[c_new][:, None]

    # --- the car's own time to the common horizon ------------------------
    # Every candidate is priced over exactly the same laps, lap_no to horizon,
    # so base pace, fuel and track evolution are identical across them and
    # cancel. What is left is what the choice actually changes: which laps run
    # on which tyre, and what the stop costs.
    old_cost = (ctx.cum_at(c_old[:, None], old_len)
                - ctx.cum_at(c_old, age - 1)[:, None])
    new_cost = ctx.cum_at(c_new[:, None], np.maximum(new_len, 0))
    off_cost = (ctx.off[c_old][:, None] * (cand[None, :] + 1)
                + ctx.off[c_new][:, None] * np.maximum(new_len, 0))

    # The pit loss does not cancel. Stopping now, with a flag out, is cheaper
    # than stopping in three laps - and the base case for three laps from now
    # is a green race, because the decision is not allowed to know that a
    # neutralization is coming.
    pit_cost = np.where(cand[None, :] == 0, pit_now_cost[:, None], ctx.pit_loss)

    own = old_cost + new_cost + off_cost + pit_cost

    # --- traffic ---------------------------------------------------------
    # base_self is the car stripped of its current tyre and compound, so its
    # pace on any other tyre is one addition away.
    pen_now = ctx.cum_at(c_old, age) - ctx.cum_at(c_old, age - 1)
    base_self = pace_self - pen_now - ctx.off[c_old]
    pace_fresh = base_self + ctx.off[c_new] + ctx.cum_at(c_new, 1)

    tot_all = total[sim]                                  # (n_sel, n_drivers)
    pace_all = pace[sim]
    n_drivers = total.shape[1]
    lap_len = np.maximum(pace_self, 1.0)

    self_mask = np.zeros((n_sel, n_drivers), dtype=bool)
    self_mask[rows, drv] = True
    unusable = retired[sim] | self_mask

    # current neighbours, taken from the engine's own running order
    pos = pos_of[sim, drv]
    front_now = order[sim, np.maximum(pos - 1, 0)]
    rear_now = order[sim, np.minimum(pos + 1, n_drivers - 1)]
    live_front = (pos > 0) & ~retired[sim, front_now]
    live_rear = (pos < n_drivers - 1) & ~retired[sim, rear_now]

    gap_front = np.maximum(t_self - total[sim, front_now], 0.0)
    adv_front = pace[sim, front_now] - pace_self
    gap_rear = np.maximum(total[sim, rear_now] - t_self, 0.0)
    adv_rear = pace_self - pace[sim, rear_now]
    shift_rear = team_shift[rear_now]

    # Detailed traffic runs over one window that every candidate shares: from
    # this lap to TRAFFIC_LAPS after the last candidate stop. That matters more
    # than it looks. Pricing three laps after each candidate's own stop gives
    # the early candidates a short window and the late ones a long one, and
    # then compares the totals as though they measured the same thing. They do
    # not, and the comparison quietly rewards whichever end of the window the
    # traffic happened to be thinner at.
    #
    # So every candidate is charged for laps lap_no .. lap_no + n_cand - 1 +
    # TRAFFIC_LAPS: k + 1 of them behind the car it is following now, the rest
    # behind whatever it rejoins between. Same laps, same count, every time.
    n_window = ctx.n_cand + ctx.traffic_laps

    pre_front = np.where(
        live_front[:, None],
        _follow_steps(ctx, gap_front, adv_front, shift, ctx.n_cand, trace=True),
        0.0)
    pre_rear = np.where(
        live_rear[:, None],
        _chased_steps(ctx, gap_rear, adv_rear, shift_rear, ctx.n_cand,
                      trace=True),
        0.0)

    traffic = np.zeros((n_sel, ctx.n_cand))
    exit_front = np.full((n_sel, ctx.n_cand), -1, dtype=np.int64)
    exit_rear = np.full((n_sel, ctx.n_cand), -1, dtype=np.int64)

    for k in range(ctx.n_cand):
        n_post = n_window - (k + 1)
        cost_a = pre_front[:, k]
        cost_b = pre_rear[:, k]

        # --- where the car rejoins ---------------------------------------
        # Both sides are carried to the same moment - the end of lap lap_no + k
        # - so this compares two positions at one instant rather than two gaps
        # measured at different times. A rival's baseline is that it carries
        # on: same compound, same tyre age, the pace it has been showing. That
        # is an assumption, and the only one available that does not require
        # knowing its plan.
        t_exit = (t_self + (k + 1) * (base_self + ctx.off[c_old])
                  + old_cost[:, k] + pit_cost[:, k])
        behind = t_exit[:, None] - (tot_all + (k + 1) * pace_all)

        # Beyond one lap time the sign of a time gap says nothing about where a
        # car is on the road, so those rivals are dropped rather than invented.
        # This is the lapped-traffic limitation, made explicit.
        reachable = ~unusable & (np.abs(behind) < lap_len[:, None])

        ahead = np.where(reachable & (behind > 0.0), behind, np.inf)
        rear = np.where(reachable & (behind < 0.0), -behind, np.inf)
        i_ahead = ahead.argmin(axis=1)
        i_rear = rear.argmin(axis=1)
        g_ahead = ahead[rows, i_ahead]
        g_rear = rear[rows, i_rear]
        got_ahead = np.isfinite(g_ahead)
        got_rear = np.isfinite(g_rear)

        exit_front[:, k] = np.where(got_ahead, i_ahead, -1)
        exit_rear[:, k] = np.where(got_rear, i_rear, -1)

        cost_c = np.where(
            got_ahead,
            _follow_steps(ctx, np.where(got_ahead, g_ahead, 0.0),
                          pace_all[rows, i_ahead] - pace_fresh, shift, n_post),
            0.0)
        cost_d = np.where(
            got_rear,
            _chased_steps(ctx, np.where(got_rear, g_rear, 0.0),
                          pace_fresh - pace_all[rows, i_rear],
                          team_shift[i_rear], n_post),
            0.0)

        # --- one rival, one charge ---------------------------------------
        # A car can be both the one in front now and the one the stop would
        # drop it behind. It is the same car and it is charged once: the larger
        # of the two phases stands, the other is dropped. Adding them would
        # build a cost no single rival could impose.
        dup = got_ahead & live_front & (i_ahead == front_now)
        cost_a = np.where(dup, np.maximum(cost_a, cost_c), cost_a)
        cost_c = np.where(dup, 0.0, cost_c)

        dup = got_ahead & live_rear & (i_ahead == rear_now)
        cost_b = np.where(dup, np.maximum(cost_b, cost_c), cost_b)
        cost_c = np.where(dup, 0.0, cost_c)

        dup = got_rear & live_front & (i_rear == front_now)
        cost_a = np.where(dup, np.maximum(cost_a, cost_d), cost_a)
        cost_d = np.where(dup, 0.0, cost_d)

        dup = got_rear & live_rear & (i_rear == rear_now)
        cost_b = np.where(dup, np.maximum(cost_b, cost_d), cost_b)
        cost_d = np.where(dup, 0.0, cost_d)

        traffic[:, k] = cost_a + cost_b + cost_c + cost_d

    # --- choose ----------------------------------------------------------
    cost = np.where(ok, own + traffic, np.inf)
    any_valid = ok.any(axis=1)

    distance = np.abs(pit_lap - planned[:, None]).astype(float)
    best = cost.min(axis=1, keepdims=True)
    pick = np.where(cost <= best + TIE_TOLERANCE, distance, np.inf).argmin(axis=1)

    # the same comparison with the rivals taken out, which is what separates
    # "the tyre chose this" from "the traffic chose this"
    tyre_only = np.where(ok, own, np.inf)
    best_t = tyre_only.min(axis=1, keepdims=True)
    pick_tyre = np.where(tyre_only <= best_t + TIE_TOLERANCE,
                         distance, np.inf).argmin(axis=1)

    chosen_lap = lap_no + pick
    moved = chosen_lap - planned
    traffic_drove = pick != pick_tyre

    reason = np.full(n_sel, ON_PLAN, dtype=np.int8)
    reason = np.where(moved < 0,
                      np.where(traffic_drove, TRAFFIC_EARLY, TYRE_EARLY), reason)
    reason = np.where(moved > 0,
                      np.where(traffic_drove, TRAFFIC_LATE, TYRE_LATE), reason)
    reason = np.where(any_valid, reason, NO_CANDIDATE).astype(np.int8)

    # --- intent ----------------------------------------------------------
    nearest = np.where(live_front, front_now, rear_now)
    nearest_live = live_front | live_rear
    my_stops = st['stops'][sim, drv]
    rival_stops = st['stops'][sim, nearest]
    rival_last_pit = st['last_pit'][sim, nearest]

    intent = np.full(n_sel, NO_INTENT, dtype=np.int8)
    intent = np.where((moved < 0) & nearest_live & (rival_stops <= my_stops),
                      UNDERCUT, intent)
    intent = np.where((moved > 0) & nearest_live & (rival_last_pit >= 0)
                      & (lap_no - rival_last_pit <= OVERCUT_MEMORY),
                      OVERCUT, intent)
    intent = np.where(any_valid, intent, NO_INTENT).astype(np.int8)

    out = {
        'pit_now': any_valid & (pick == 0),
        'chosen_lap': chosen_lap,
        'planned_lap': planned,
        'reason': reason,
        'intent': intent,
        'valid': any_valid,
        'log': [],
    }

    if log_sim is not None:
        out['log'] = _log(ctx, lap_no, st, sim, drv, log_sim, ok, own, traffic,
                          cost, pick, planned, chosen_lap, reason, intent,
                          front_now, rear_now, live_front, live_rear,
                          exit_front, exit_rear)
    return out


def _log(ctx, lap_no, st, sim, drv, log_sim, ok, own, traffic, cost, pick,
         planned, chosen_lap, reason, intent, front_now, rear_now,
         live_front, live_rear, exit_front, exit_rear):
    """
    The arithmetic behind one simulation's calls, for the diagnostics panel.

    One race, not ten thousand. Every decision of every car of every simulation
    is tens of megabytes of text nobody reads; one race is a thing a person can
    actually check the model against.
    """
    names = st.get('names')
    entries = []
    for r in np.flatnonzero(sim == log_sim):
        k = int(pick[r])
        roles = []
        if live_front[r]:
            roles.append((int(front_now[r]), 'ahead now'))
        if live_rear[r]:
            roles.append((int(rear_now[r]), 'behind now'))
        if exit_front[r, k] >= 0:
            roles.append((int(exit_front[r, k]), 'ahead at exit'))
        if exit_rear[r, k] >= 0:
            roles.append((int(exit_rear[r, k]), 'behind at exit'))

        seen, rivals = set(), []
        for who, role in roles:
            if who in seen:
                continue
            seen.add(who)
            rivals.append((names[who] if names is not None else who, role))

        entries.append({
            'lap': int(lap_no),
            'driver': names[drv[r]] if names is not None else int(drv[r]),
            'planned': int(planned[r]),
            'chosen': int(chosen_lap[r]),
            'pit_now': k == 0,
            'reason': REASON_NAMES[int(reason[r])],
            'intent': INTENT_NAMES[int(intent[r])],
            'candidates': [(int(lap_no + c), round(float(own[r, c]), 3),
                            round(float(traffic[r, c]), 3),
                            round(float(cost[r, c]), 3))
                           for c in np.flatnonzero(ok[r])],
            'rivals': rivals,
        })
    return entries
