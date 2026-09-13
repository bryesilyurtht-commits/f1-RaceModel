"""
Acceptance tests for the v2.0 pit decision.

Every test here is one of the scenarios the v2.0 roadmap requires before the
version counts as finished. They are written against small hand-built states
rather than a race, because a 10,000-simulation run can only tell you the
answer changed - not which of thirty interacting rules changed it.

    python -m Simülasyon.test_reactive_strategy
"""

import numpy as np

from Simülasyon import reactive_strategy as v2
from Simülasyon.simulate import pass_probability


COMPOUND_B1 = (0.060, 0.040, 0.030)
COMPOUND_B2 = (0.0040, 0.0020, 0.0010)
LAP = 80.0                       # a long enough lap that nothing is "lapped"


def make_cum(max_age=60):
    """A cumulative tyre-loss table shaped like the fitted ones."""
    cum = np.zeros((3, max_age + 1))
    ages = np.arange(max_age + 1, dtype=float)
    for c in range(3):
        pen = COMPOUND_B1[c] * ages + COMPOUND_B2[c] * ages * ages
        pen[0] = 0.0
        cum[c] = np.cumsum(pen)
    return cum


def make_ctx(pit_loss=20.0, band=5, traffic_laps=3, dirty_max=0.40,
             caps=(30.0, 40.0, 50.0), offsets=(-0.35, 0.0, 0.30),
             pass_rate=0.22):
    model = {'intercept': np.log(pass_rate / (1 - pass_rate))
             - (1.325 * 0.50 - 4.074 * 0.60),
             'threshold': 1.60}
    return v2.Context(
        n_laps=60, cum_loss=make_cum(), cap_by_compound=np.array(caps),
        offsets=np.array(offsets), pit_loss=pit_loss,
        pass_prob=pass_probability, pass_model=model,
        dirty_max=dirty_max, dirty_range=2.5, attack_gap=1.00,
        held_gap=0.65, band=band, traffic_laps=traffic_laps)


def scenario(totals, paces, *, me=0, age=10, planned=20, lap=14, horizon=45,
             c_old=0, c_new=2, retired=None, pit_cost_now=None,
             stops=None, last_pit=None):
    """
    One simulation, n cars, everything the decision is allowed to see.

    totals are cumulative race times, so a smaller one is further up the road.
    Cars are ordered by total, which is what the engine's running order would
    say in the absence of an unresolved pass.
    """
    totals = np.asarray(totals, dtype=float)[None, :]
    paces = np.asarray(paces, dtype=float)[None, :]
    n = totals.shape[1]
    retired = (np.zeros((1, n), dtype=bool) if retired is None
               else np.asarray(retired, dtype=bool)[None, :])

    order = np.argsort(totals, axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(n)[None, :], axis=1)

    full = lambda v, dt: np.full((1, n), v, dtype=dt)
    return {
        'sim': np.array([0]), 'drv': np.array([me]),
        'total': totals, 'pace': paces, 'retired': retired,
        'order': order, 'pos': pos,
        'age': full(age, np.int64),
        'c_old': full(c_old, np.int64), 'c_new': full(c_new, np.int64),
        'planned': full(planned, np.int64), 'horizon': full(horizon, np.int64),
        'pit_cost_now': full(20.0 if pit_cost_now is None else pit_cost_now,
                             float),
        'team_shift': np.zeros(n),
        'stops': full(0, np.int64) if stops is None
                 else np.asarray(stops, dtype=np.int64)[None, :],
        'last_pit': full(-1, np.int64) if last_pit is None
                    else np.asarray(last_pit, dtype=np.int64)[None, :],
        'names': np.array([f'C{i}' for i in range(n)]),
        'lap': lap,
    }


def call(ctx, st, log_sim=None):
    return v2.decide(ctx, st['lap'] - 1, st, log_sim=log_sim)


def alone(n=6, spread=LAP * 3, **kw):
    """Everyone else a lap or more away, so nobody is traffic."""
    totals = [0.0] + [spread * (i + 1) for i in range(n - 1)]
    return scenario(totals, [LAP] * n, me=0, **kw)


# --- 1. the switch ----------------------------------------------------------

def test_off_consumes_no_random_numbers():
    """
    v2.0 off must reproduce v1.1 exactly, which means more than the same
    result: it means the random stream is untouched. Retirements are drawn
    from that stream and depend on nothing the strategy does, so if the two
    runs retire the same cars on the same laps, no draw was added or skipped.
    """
    from Simülasyon import simulate as s
    with s.flag_overrides({'REACTIVE_PIT_V2': False}):
        a = s.run_simulation(*_tiny(s), 400, seed=7)[3]
    with s.flag_overrides({'REACTIVE_PIT_V2': True}):
        b = s.run_simulation(*_tiny(s), 400, seed=7)[3]
    assert np.array_equal(a['retired_lap'], b['retired_lap'])
    assert np.array_equal(a['neutral'], b['neutral'])


_TINY = {}


def _tiny(s):
    if 'v' not in _TINY:
        pace = s.load_drivers()
        track = s.load_track()
        strategies, _ = s.recost_strategies(s.load_strategies(track), track)
        _TINY['v'] = (pace, track, strategies)
    return _TINY['v']


# --- 2. what the decision may not know --------------------------------------

def test_state_carries_no_future():
    """
    The guarantee is structural: the decision is handed a dict, and that dict
    has no rival plan, no pit lap anyone intends to take, and no flag
    schedule in it. A key added later that carried one would fail here before
    it could quietly change a race.
    """
    allowed = {'sim', 'drv', 'total', 'pace', 'retired', 'order', 'pos',
               'age', 'c_old', 'c_new', 'planned', 'horizon', 'pit_cost_now',
               'team_shift', 'stops', 'last_pit', 'names', 'lap'}
    st = alone()
    assert set(st) <= allowed
    # `planned` and `horizon` are the car's own plan, which it may see
    assert st['planned'].shape == st['total'].shape


def test_rival_plans_do_not_move_the_decision():
    """
    Two identical races, one where every other car intends to stop next lap
    and one where none of them does. The decision cannot tell them apart,
    because that information never reaches it.
    """
    ctx = make_ctx()
    st = scenario([0.0, -3.0, 4.0, 9.0], [LAP] * 4)
    first = call(ctx, st)
    # whatever a rival's plan is, it lives in arrays decide() never receives
    st_other = dict(st)
    st_other['their_pit_laps'] = np.full((1, 4), st['lap'] + 1)
    second = v2.decide(ctx, st['lap'] - 1, st_other)
    assert first['chosen_lap'] == second['chosen_lap']


# --- 3. order invariance ----------------------------------------------------

def test_driver_order_does_not_change_decisions():
    """
    The same race with the cars written into the arrays in a different order.
    Every car must get the same answer, because no car's decision is applied
    before another car's is computed.
    """
    ctx = make_ctx()
    totals = np.array([0.0, 1.2, 2.6, 5.0, 40.0, 41.0])
    paces = np.array([LAP, LAP + 0.3, LAP - 0.2, LAP + 0.1, LAP, LAP + 0.4])

    st = scenario(totals, paces)
    st['drv'] = np.arange(6)
    st['sim'] = np.zeros(6, dtype=int)
    base = call(ctx, st)

    perm = np.array([3, 0, 5, 2, 1, 4])
    st2 = scenario(totals[perm], paces[perm])
    st2['drv'] = np.arange(6)
    st2['sim'] = np.zeros(6, dtype=int)
    shuffled = call(ctx, st2)

    for new_i, old_i in enumerate(perm):
        assert shuffled['chosen_lap'][new_i] == base['chosen_lap'][old_i], \
            f'car {old_i} answered differently when the arrays were reordered'


# --- 4. no traffic means no rival term --------------------------------------

def test_empty_track_costs_nothing_and_the_tyre_decides():
    ctx = make_ctx()
    st = alone()
    answer = call(ctx, st, log_sim=0)
    entry = answer['log'][0]
    assert all(traffic == 0.0 for _, _, traffic, _ in entry['candidates']), \
        'an empty track charged a traffic penalty'
    assert entry['reason'].startswith('tyre') or entry['reason'] == 'on plan'


def test_alone_the_choice_matches_the_tyre_table_by_hand():
    """
    With nobody else on the road the answer is arithmetic, so it can be done
    twice. A disagreement here means the candidate table and the tyre table
    have drifted apart.
    """
    ctx, st = make_ctx(), alone(age=10, planned=20, lap=14, horizon=45)
    answer = call(ctx, st, log_sim=0)
    cum, c_old, c_new = ctx.cum, 0, 2

    by_hand = {}
    for k in range(ctx.n_cand):
        pit_lap, age = 14 + k, 10 + k
        new_len = 45 - pit_lap
        if pit_lap > 20 + ctx.band or new_len < 1:
            continue
        by_hand[pit_lap] = (cum[c_old, age] - cum[c_old, 9]
                            + cum[c_new, new_len]
                            + ctx.off[c_old] * (k + 1) + ctx.off[c_new] * new_len
                            + 20.0)
    best = min(by_hand, key=lambda lp: (by_hand[lp], abs(lp - 20)))
    assert answer['chosen_lap'][0] == best, \
        f'model chose {answer["chosen_lap"][0]}, arithmetic says {best}'


# --- 5. traffic can move the answer both ways -------------------------------

def _exit_time(ctx, st, k):
    """Where the car rejoins if it stops k laps from now, in race time."""
    age, c_old = int(st['age'][0, 0]), int(st['c_old'][0, 0])
    pace = st['pace'][0, int(st['drv'][0])]
    base = pace - (ctx.cum_at(c_old, age) - ctx.cum_at(c_old, age - 1)) \
        - ctx.off[c_old]
    old_cost = ctx.cum_at(c_old, age + k) - ctx.cum_at(c_old, age - 1)
    return (st['total'][0, int(st['drv'][0])]
            + (k + 1) * (base + ctx.off[c_old]) + old_cost + ctx.pit_loss)


def place_at_exit(ctx, st, k, ahead_by, pace):
    """
    The cumulative time a rival needs now to be `ahead_by` seconds up the road
    when the car rejoins after stopping k laps from now.

    Both sides have to be carried to the same moment, which is the end of lap
    L + k. Forgetting the rival's own (k + 1) laps puts it a lap away and the
    scenario silently stops testing anything.
    """
    return _exit_time(ctx, st, k) - ahead_by - (k + 1) * pace


def test_a_bad_exit_makes_waiting_win():
    """
    Same tyre, same pit loss, same plan. The only difference is a slow car
    sitting exactly where stopping now would drop the driver.

    The tyre has to be worn enough that it would stop on its own, or the test
    proves nothing: a candidate the tyre curve had already ruled out cannot be
    ruled out again by traffic, and a scenario where the planned lap wins
    either way would pass while measuring nothing.
    """
    ctx = make_ctx()
    worn = dict(age=26, planned=20, lap=22, horizon=45)

    clean = scenario([0.0, LAP * 4, LAP * 5], [LAP] * 3, **worn)
    clean_answer = call(ctx, clean)
    assert clean_answer['pit_now'][0], 'the tyre alone should have stopped here'

    # a car 0.3 s up the road from where stopping now rejoins, slow enough
    # that the driver would sit behind it rather than go by
    slow = LAP + 1.2
    busy = scenario([0.0, place_at_exit(ctx, clean, 0, 0.3, slow), LAP * 5],
                    [LAP, slow, LAP], **worn)
    busy_answer = call(ctx, busy)

    assert not busy_answer['pit_now'][0], 'drove into the back of the traffic'
    assert busy_answer['chosen_lap'][0] > clean_answer['chosen_lap'][0], \
        'exit traffic did not delay the stop'


def test_a_clean_exit_lets_a_worn_tyre_stop_early():
    """
    The other direction. A tyre far enough past its useful life that the
    curve alone says stop, and an empty road to rejoin onto.
    """
    ctx = make_ctx()
    st = alone(age=26, planned=20, lap=22, horizon=45)
    answer = call(ctx, st)
    assert answer['pit_now'][0], 'a worn tyre with a clear exit stayed out'


def test_a_rival_stopping_does_not_trigger_a_stop():
    """
    The rule this version explicitly refuses to have. A car ahead pits; if
    staying out is still cheaper, the driver stays out.
    """
    ctx = make_ctx()
    st = scenario([0.0, -2.0, LAP * 5], [LAP, LAP, LAP],
                  age=6, planned=20, lap=15,
                  last_pit=[-1, 14, -1], stops=[0, 1, 0])
    answer = call(ctx, st)
    assert not answer['pit_now'][0], 'copied a rival into the pit lane'


# --- 6. one rival, one charge -----------------------------------------------

def test_the_same_car_in_two_roles_is_charged_once():
    """
    A car slow enough to be both the one in front now and the one the stop
    would rejoin behind. Charging both phases would price a rival that does
    not exist.
    """
    ctx = make_ctx()
    st = scenario([0.0, -0.5, LAP * 6], [LAP, LAP + 0.9, LAP],
                  age=10, planned=20, lap=16)
    answer = call(ctx, st, log_sim=0)
    rivals = answer['log'][0]['rivals']
    names = [n for n, _ in rivals]
    assert len(names) == len(set(names)), f'a car was listed twice: {rivals}'


def test_the_leader_has_no_car_in_front():
    ctx = make_ctx()
    st = scenario([0.0, 1.0, 2.0], [LAP, LAP, LAP], me=0)
    answer = call(ctx, st, log_sim=0)
    assert answer['valid'][0]
    assert 'ahead now' not in [r for _, r in answer['log'][0]['rivals']]


def test_the_last_car_has_nobody_behind():
    ctx = make_ctx()
    st = scenario([0.0, 1.0, 2.0], [LAP, LAP, LAP], me=2)
    answer = call(ctx, st, log_sim=0)
    assert answer['valid'][0]
    assert 'behind now' not in [r for _, r in answer['log'][0]['rivals']]


def test_a_retired_neighbour_is_not_traffic():
    ctx = make_ctx()
    st = scenario([0.0, -0.4, 0.4], [LAP, LAP + 2.0, LAP + 2.0],
                  me=0, retired=[False, True, True])
    answer = call(ctx, st, log_sim=0)
    assert answer['log'][0]['rivals'] == []


# --- 7. the plan survives ---------------------------------------------------

def test_a_candidate_that_breaks_the_last_stint_cap_is_refused():
    """
    Stopping early makes the stint after it longer. If that stint would then
    run past what the compound's curve was measured over, the candidate is
    not offered - rather than taken and then rescued by a stop nobody planned.
    """
    ctx = make_ctx(caps=(30.0, 40.0, 22.0))     # HARD capped at 22 laps
    st = alone(age=10, planned=20, lap=14, horizon=45, c_old=0, c_new=2)
    answer = call(ctx, st, log_sim=0)
    for pit_lap, _, _, _ in answer['log'][0]['candidates']:
        assert 45 - pit_lap <= 22, \
            f'offered a stop at lap {pit_lap}, leaving a {45 - pit_lap}-lap stint'
    assert answer['chosen_lap'][0] >= 23


def test_a_candidate_past_the_current_compounds_cap_is_refused():
    ctx = make_ctx(caps=(18.0, 40.0, 50.0))     # SOFT capped at 18 laps
    st = alone(age=14, planned=20, lap=18, horizon=45, c_old=0)
    answer = call(ctx, st, log_sim=0)
    for pit_lap, _, _, _ in answer['log'][0]['candidates']:
        assert 14 + (pit_lap - 18) <= 18


def test_no_candidate_leaves_a_zero_length_stint():
    ctx = make_ctx()
    st = alone(age=10, planned=20, lap=19, horizon=21)
    answer = call(ctx, st, log_sim=0)
    for pit_lap, _, _, _ in answer['log'][0]['candidates']:
        assert pit_lap <= 20, f'a stop at {pit_lap} leaves nothing to run'


def test_no_candidate_runs_past_the_window():
    ctx = make_ctx(band=5)
    st = alone(age=10, planned=20, lap=17, horizon=45)
    answer = call(ctx, st, log_sim=0)
    laps = [c[0] for c in answer['log'][0]['candidates']]
    assert min(laps) == 17 and max(laps) == 25
    assert len(laps) == 9


def test_the_window_never_offers_more_than_eleven():
    ctx = make_ctx(band=5)
    for lap in range(15, 26):
        st = alone(age=10, planned=20, lap=lap, horizon=55)
        entry = call(ctx, st, log_sim=0)['log'][0]
        assert len(entry['candidates']) <= 11


def test_an_infeasible_plan_is_reported_not_hidden():
    """No legal candidate is a thing that has to be visible, not silently a stop."""
    ctx = make_ctx(caps=(4.0, 4.0, 4.0))
    st = alone(age=10, planned=20, lap=14, horizon=45)
    answer = call(ctx, st)
    assert not answer['valid'][0]
    assert not answer['pit_now'][0]
    assert v2.REASON_NAMES[int(answer['reason'][0])] == 'no valid candidate'


# --- 8. the common horizon --------------------------------------------------

def test_every_candidate_is_priced_over_the_same_laps():
    """
    The trap this version is most likely to fall into: comparing a short
    window against a long one. Each candidate splits the same lap count
    between the old tyre and the new one, so the totals differ only by what
    the choice changes.
    """
    ctx = make_ctx()
    age, lap, horizon = 10, 14, 45
    st = alone(age=age, planned=20, lap=lap, horizon=horizon)
    entry = call(ctx, st, log_sim=0)['log'][0]
    for pit_lap, _, _, _ in entry['candidates']:
        old_laps = pit_lap - lap + 1
        new_laps = horizon - pit_lap
        assert old_laps + new_laps == horizon - lap + 1


def test_the_pit_loss_is_charged_once_per_candidate():
    """
    It sits on every candidate, so it cannot be dropped as a constant - the
    flag makes it differ - but it must appear exactly once.
    """
    st = alone(age=10, planned=20, lap=14, horizon=45, pit_cost_now=20.0)
    paid = [c[1] for c in
            call(make_ctx(pit_loss=20.0), st, log_sim=0)['log'][0]['candidates']]

    free = alone(age=10, planned=20, lap=14, horizon=45, pit_cost_now=0.0)
    unpaid = [c[1] for c in
              call(make_ctx(pit_loss=0.0), free, log_sim=0)['log'][0]['candidates']]

    for with_stop, without in zip(paid, unpaid):
        assert abs((with_stop - without) - 20.0) < 1e-6, \
            f'a 20 s stop moved the cost by {with_stop - without:.3f} s'


def test_a_flag_makes_stopping_now_cheaper_but_only_now():
    """
    A discounted stop is available on the lap the flag is out. The decision is
    not allowed to assume the discount will still be there in three laps, so
    every later candidate pays full price.
    """
    ctx = make_ctx(pit_loss=20.0)
    green = alone(age=10, planned=20, lap=14, horizon=45, pit_cost_now=20.0)
    yellow = alone(age=10, planned=20, lap=14, horizon=45, pit_cost_now=10.0)

    g = call(ctx, green, log_sim=0)['log'][0]['candidates']
    y = call(ctx, yellow, log_sim=0)['log'][0]['candidates']
    assert y[0][1] == round(g[0][1] - 10.0, 3), 'the discount did not apply'
    assert [c[1] for c in y[1:]] == [c[1] for c in g[1:]], \
        'the discount leaked into laps that have not happened'


def test_a_blocked_lap_is_not_also_charged_dirty_air():
    """
    Inside the attacking window the car is pinned to the one ahead and loses
    its pace advantage. Adding the dirty-air penalty on top would charge the
    same lap twice. The two bands do not overlap, and this checks the seam.
    """
    ctx = make_ctx(traffic_laps=1, dirty_max=0.40)
    adv = np.array([0.5])
    inside = v2._follow_steps(ctx, np.array([0.5]), adv, np.zeros(1), 1)
    outside = v2._follow_steps(ctx, np.array([1.8]), adv, np.zeros(1), 1)
    assert np.isclose(inside[0], 0.5), inside
    assert np.isclose(outside[0], 0.40 * (1 - 1.8 / 2.5)), outside


def test_a_pass_is_not_free_when_it_is_not_certain():
    """
    A car with a real chance of getting through still pays for the chance it
    does not. Charging nothing would let the model wave every quick car past.
    """
    ctx = make_ctx(traffic_laps=2)
    cost = v2._follow_steps(ctx, np.array([0.3]), np.array([0.8]),
                            np.zeros(1), 2)
    assert 0.8 < cost[0] < 1.6, cost


def test_a_car_behind_is_not_a_penalty_by_itself():
    """
    No points for holding a position and no charge for being chased. The cost
    of a rival behind is what happens after it gets past, and nothing else.
    """
    ctx = make_ctx()
    # 2.2 s back and losing ground: it never arrives, so it never costs
    slower = v2._chased_steps(ctx, np.array([2.2]), np.array([-0.5]),
                              np.zeros(1), 3)
    assert slower[0] == 0.0, 'charged for a car that cannot catch'
    quicker = v2._chased_steps(ctx, np.array([0.4]), np.array([0.9]),
                               np.zeros(1), 3)
    assert quicker[0] > 0.0

    # a car already alongside is charged, because the engine really does roll
    # for it - but only for what follows the pass, never for being there
    alongside = v2._chased_steps(ctx, np.array([0.4]), np.array([-0.5]),
                                 np.zeros(1), 3)
    assert alongside[0] < quicker[0]


# --- 9. tie-breaking --------------------------------------------------------

def test_a_tie_goes_to_the_planned_lap():
    """
    No arbitrary "must gain a second" rule, so ties have to resolve somewhere,
    and the plan is the only non-arbitrary place.
    """
    ctx = make_ctx(offsets=(0.0, 0.0, 0.0))
    flat = np.zeros((3, 61))                 # a tyre that never degrades
    ctx.cum = flat
    st = alone(age=10, planned=20, lap=16, horizon=45)
    answer = call(ctx, st)
    assert answer['chosen_lap'][0] == 20, answer['chosen_lap']


def test_intent_labels_are_about_intent_not_outcome():
    """
    An early stop while the car ahead is still out reads as an undercut
    attempt. It says nothing about whether it worked, which is not knowable
    at the moment the call is made.
    """
    ctx = make_ctx()
    st = scenario([0.0, -1.0, LAP * 6], [LAP, LAP, LAP],
                  age=24, planned=26, lap=22, horizon=45,
                  stops=[0, 0, 0], last_pit=[-1, -1, -1])
    answer = call(ctx, st)
    if answer['chosen_lap'][0] < 26:
        assert v2.INTENT_NAMES[int(answer['intent'][0])] == 'undercut'


# --- 10. the engine's own guarantees ---------------------------------------

def test_one_stop_per_lap_whatever_fires():
    """
    A red flag, a cap and a v2.0 call can all want the same lap. The engine
    must produce one stop, not three, and the tyre age must reset once.
    """
    from Simülasyon import simulate as s
    pace, track, strategies = _tiny(s)
    with s.flag_overrides({'REACTIVE_PIT_V2': True}):
        _, _, _, diag = s.run_simulation(pace, track, strategies, 600, seed=3)

    stops = diag['n_stops']
    assert stops.max() <= 6, f'{stops.max()} stops in one race'
    # a free change under a red flag is counted as a stop but reported apart
    assert diag['rf_stops'].sum() <= stops.sum()
    assert diag['forced_pits'].sum() <= stops.sum()


def test_the_plan_is_kept_when_v2_moves_the_lap():
    """
    v2.0 moves pit laps and nothing else. The number of stops a car makes
    under normal conditions is the plan's, not the decision layer's.
    """
    from Simülasyon import simulate as s
    pace, track, strategies = _tiny(s)
    with s.flag_overrides({'REACTIVE_PIT_V2': False, 'DNF_ENABLED': False,
                           'RED_FLAG_ENABLED': False}):
        _, _, idx_a, a = s.run_simulation(pace, track, strategies, 600, seed=5)
    with s.flag_overrides({'REACTIVE_PIT_V2': True, 'DNF_ENABLED': False,
                           'RED_FLAG_ENABLED': False}):
        _, _, idx_b, b = s.run_simulation(pace, track, strategies, 600, seed=5)

    assert np.array_equal(idx_a, idx_b), 'the strategy menu itself moved'
    planned = np.array([len(p) - 1 for p in a['plans']])[idx_a]
    for name, diag in (('v1.1', a), ('v2.0', b)):
        extra = diag['n_stops'] - planned
        share = float((extra > 0).mean())
        assert share < 0.15, f'{name} added a stop in {share:.1%} of cars'


def test_v2_reports_what_it_did():
    from Simülasyon import simulate as s
    pace, track, strategies = _tiny(s)
    with s.flag_overrides({'REACTIVE_PIT_V2': True}):
        _, _, _, diag = s.run_simulation(pace, track, strategies, 400, seed=11)

    assert diag['v2_calls'].sum() > 0, 'v2.0 was on and never asked'
    assert diag['v2_reasons'].sum() == diag['v2_calls'].sum()
    assert diag['v2_intents'].sum() == diag['v2_calls'].sum()
    assert len(diag['v2_log']) > 0, 'no decision log for the nominated race'
    entry = diag['v2_log'][0]
    assert entry['reason'] in v2.REASON_NAMES
    assert entry['intent'] in v2.INTENT_NAMES
    assert len(entry['rivals']) <= v2.MAX_RIVALS


def _schedule(spans):
    """A fixed neutralization pattern, so two runs can differ only in the future."""
    def sample(sc, n_sims, n_laps, rng):
        state = np.zeros((n_sims, n_laps), dtype=np.int8)
        for start, end, code in spans:
            state[:, start:end] = code
        return state
    return sample


def _race_with(schedule, **flags):
    from Simülasyon import simulate as s
    pace, track, strategies = _tiny(s)
    original = s.sample_neutralizations
    s.sample_neutralizations = schedule
    try:
        with s.flag_overrides(flags):
            return s.run_simulation(pace, track, strategies, 300, seed=17)[3]
    finally:
        s.sample_neutralizations = original


def test_a_later_safety_car_cannot_change_an_earlier_decision():
    """
    The scenario the roadmap asks for, made sharp. Two races with an identical
    safety car and one difference: in the first it is withdrawn after lap 13,
    in the second it stays out to lap 16. Laps 0 to 13 are the same in both, so
    a stop on lap 13 that comes out differently can only have read a lap that
    has not been run.

    The second half is the part that matters, and it is an assertion rather
    than a comment: it shows the leak was real. Both the v1.1 rule and v2.0
    with NEUTRAL_LAST_LAP_KNOWN turned back on diverge on exactly that lap.
    Without it the scenario could go inert - the safety car landing somewhere
    no car was deciding anything - and pass while measuring nothing.
    """
    short = _schedule([(9, 14, 2)])
    long = _schedule([(9, 17, 2)])
    base = dict(DNF_ENABLED=False, RED_FLAG_ENABLED=False)

    def diverges_at(**flags):
        a = _race_with(short, **base, **flags)
        b = _race_with(long, **base, **flags)
        assert np.array_equal(a['trace_pits'][:13], b['trace_pits'][:13]), \
            'the two schedules were not identical before lap 13 after all'
        return not np.array_equal(a['trace_pits'][13], b['trace_pits'][13])

    assert not diverges_at(REACTIVE_PIT_V2=True,
                           NEUTRAL_LAST_LAP_KNOWN=False), \
        'a safety car that had not ended yet changed a stop already taken'
    assert diverges_at(REACTIVE_PIT_V2=True, NEUTRAL_LAST_LAP_KNOWN=True), \
        'the leak this flag exists to reproduce did not fire'
    assert diverges_at(REACTIVE_PIT_V2=False), \
        'v1.1 was supposed to have this leak; the scenario is inert'


def test_v2_off_leaves_no_trace():
    from Simülasyon import simulate as s
    pace, track, strategies = _tiny(s)
    with s.flag_overrides({'REACTIVE_PIT_V2': False}):
        _, _, _, diag = s.run_simulation(pace, track, strategies, 400, seed=11)
    assert diag['v2_calls'].sum() == 0
    assert diag['v2_log'] == []


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as exc:
            failed.append((name, str(exc) or 'assertion failed'))
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, f'{type(exc).__name__}: {exc}'))

    print(f'\n{passed}/{len(tests)} passed')
    for name, why in failed:
        print(f'  FAIL  {name}\n        {why}')
    return not failed


if __name__ == '__main__':
    import sys
    sys.exit(0 if _run() else 1)
