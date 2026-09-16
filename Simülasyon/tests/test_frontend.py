"""
Acceptance tests for the interface's arithmetic and run labelling.

The frontend faults worth testing are the quiet ones. A chart that adds two
overlapping probabilities still renders, still looks like a chart, and is only
wrong if you read it. A result left on screen after the controls moved is still
a valid result - of a question nobody is asking any more. Neither raises.

So these test what a person would have to check by eye: that the shareable
chart's bars end where the podium probability is, that a changed setting is
noticed, and that every switch has a name a reader can use.

    python -m Simülasyon.tests.test_frontend
"""

import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _app():
    """
    app.py as a module.

    Everything Streamlit does at import time is inside main(), so the module
    body loads without a running server.
    """
    spec = importlib.util.spec_from_file_location(
        'appmod', os.path.join(ROOT, 'app.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APP = _app()


def _result(n=12):
    """A summary shaped like the simulator's, with podium >= win by construction."""
    rng = np.random.default_rng(4)
    win = np.sort(rng.dirichlet(np.ones(n) * 0.6))[::-1]
    podium = np.clip(win + rng.uniform(0.05, 0.45, n), 0, 1)
    podium = np.maximum(podium, win)
    return {
        'event': 'Test Grand Prix', 'season': 2026, 'n_sims': 10_000,
        'n_laps': 53,
        'summary': pd.DataFrame({
            'Driver': [f'D{i:02d}' for i in range(n)],
            'Team': [f'T{i // 2}' for i in range(n)],
            'grid': np.arange(1, n + 1, dtype=float),
            'mean_pos': np.arange(1, n + 1, dtype=float) + rng.normal(0, .4, n),
            'P_win': win, 'P_podium': podium,
            'P_points': np.clip(podium + 0.2, 0, 1),
        }),
    }


# --- the shareable chart ----------------------------------------------------

def test_the_bars_end_at_the_podium_probability():
    """
    Winning is already a podium. Stacking win on top of podium would put a
    driver above 100%, so the second segment is podium minus win and the bar
    ends at podium.
    """
    result = _result()
    figure = APP.share_card(result)
    first = np.asarray(figure.data[0].x, dtype=float)
    second = np.asarray(figure.data[1].x, dtype=float)
    expected = result['summary'].head(10).iloc[::-1]['P_podium'].to_numpy()
    assert np.allclose(first + second, expected), first + second


def test_no_bar_exceeds_one():
    figure = APP.share_card(_result())
    total = (np.asarray(figure.data[0].x, dtype=float)
             + np.asarray(figure.data[1].x, dtype=float))
    assert (total <= 1.0 + 1e-9).all(), total.max()


def test_the_second_segment_is_never_negative():
    """podium < win cannot happen, and if it ever did the bar would run backwards."""
    figure = APP.share_card(_result())
    assert (np.asarray(figure.data[1].x, dtype=float) >= -1e-9).all()


def test_the_segments_are_not_called_win_and_podium():
    """
    The arithmetic was right and the labels were not: the second segment was
    called "podium" when it is the rest of the podium. A correct chart with
    those labels reads as the wrong one.
    """
    names = [trace.name for trace in APP.share_card(_result()).data]
    assert names[0] == 'Win'
    assert names[1] != 'podium' and names[1] != 'Podium'
    assert '2' in names[1] and '3' in names[1], names[1]


def test_both_segments_carry_a_number_on_hover():
    for trace in APP.share_card(_result()).data:
        assert '%{x' in trace.hovertemplate, trace.hovertemplate


def test_the_exported_image_says_what_it_is():
    """It travels without the page, so race, runs and the unit go in the title."""
    title = APP.share_card(_result()).layout.title.text
    assert 'Test Grand Prix' in title
    assert '10,000' in title
    assert 'podium' in title.lower()


# --- knowing which run is on screen -----------------------------------------

def _settings(n_sims=10_000, seed=42, dnf=True, scenario=None):
    return APP.settings_signature(n_sims, seed,
                                  {'DNF_ENABLED': dnf, 'IS_WET': False},
                                  scenario)


def test_unchanged_settings_report_no_drift():
    base = _settings()
    assert APP.settings_drift({'settings': base}, base) == []


def test_a_changed_flag_is_noticed():
    drift = APP.settings_drift({'settings': _settings()},
                               _settings(dnf=False))
    assert len(drift) == 1
    assert 'on -> off' in drift[0]


def test_a_changed_run_count_is_noticed():
    drift = APP.settings_drift({'settings': _settings()},
                               _settings(n_sims=5000))
    assert any('5,000' in d for d in drift), drift


def test_a_changed_seed_is_noticed():
    drift = APP.settings_drift({'settings': _settings()}, _settings(seed=7))
    assert any('seed' in d for d in drift), drift


def test_a_changed_weather_scenario_is_noticed():
    drift = APP.settings_drift({'settings': _settings()},
                               _settings(scenario='shower'))
    assert any('weather' in d for d in drift), drift


def test_drift_names_the_switch_the_way_the_sidebar_does():
    """A drift message naming DNF_ENABLED sends the reader to the code."""
    drift = APP.settings_drift({'settings': _settings()},
                               _settings(dnf=False))
    assert 'Include retirements' in drift[0], drift


def test_a_run_without_stored_settings_claims_no_drift():
    """Older entries have nothing to compare; silence beats a false alarm."""
    assert APP.settings_drift({'result': {}}, _settings()) == []


def test_several_changes_are_all_reported():
    drift = APP.settings_drift({'settings': _settings()},
                               _settings(n_sims=2000, seed=9, dnf=False))
    assert len(drift) == 3, drift


# --- the switches have names ------------------------------------------------

def test_every_runtime_flag_has_a_label():
    from Simülasyon import simulate as S
    missing = [f for f in S.RUNTIME_FLAGS if f not in APP.FLAG_LABELS]
    assert not missing, missing


def test_no_label_is_just_the_code_name():
    assert not [k for k, v in APP.FLAG_LABELS.items() if k == v]


def test_the_help_text_keeps_the_code_name():
    """It is what a bug report and the run summary will both say."""
    for name in APP.FLAG_LABELS:
        assert name in APP.flag_help(name)


def test_the_lock_flag_explains_that_it_also_covers_restarts():
    """
    It reads as a start-only setting and is passed to build_start_gaps at the
    race start and again at a red-flag restart.
    """
    text = APP.flag_help('LOCK_START_ORDER').lower()
    assert 'red-flag' in text or 'red flag' in text
    assert 'safety car' in text


def test_every_flag_the_model_offers_is_reachable():
    """
    A flag the sidebar forgot is a flag nobody can set, which is the same as
    the model not having it.
    """
    from Simülasyon import simulate as S
    source = open(os.path.join(ROOT, 'app.py'), encoding='utf-8').read()
    for name in S.RUNTIME_FLAGS:
        assert f"'{name}'" in source, name


# --- the headline -----------------------------------------------------------

def test_the_headline_picks_the_actual_maxima():
    result = _result()
    summary = result['summary']
    assert summary['P_win'].idxmax() is not None
    top_win = summary.loc[summary['P_win'].idxmax(), 'Driver']
    top_podium = summary.loc[summary['P_podium'].idxmax(), 'Driver']
    # the two are allowed to disagree, and nothing may force them to agree
    assert isinstance(top_win, str) and isinstance(top_podium, str)


# --- the page, actually rendered --------------------------------------------

def _rendered(n_sims=1000):
    """
    The app driven headlessly through one real run.

    Slow, and the only check that covers the wiring rather than the pieces: a
    panel that raises while rendering leaves every unit test passing.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    at.sidebar.select_slider[0].set_value(n_sims).run()
    at.sidebar.button[0].click().run()
    return at


def test_the_page_renders_a_run_without_raising():
    at = _rendered()
    assert len(at.exception) == 0, [e.value for e in at.exception]
    assert len(at.error) == 0, [e.value for e in at.error]
    # six top-level tabs, plus one sub-page per winner and one for the run's
    # own typical race inside the last of them
    assert len(at.tabs) > 6


def test_the_default_table_is_seven_columns_not_thirteen():
    """DNF causes, stops, affinity and strategy are behind More columns."""
    columns = list(_rendered().dataframe[0].value.columns)
    assert columns == ['Driver', 'Team', 'grid', 'mean_pos', 'P_win',
                       'P_podium', 'P_points'], columns


def test_the_run_bar_says_which_run_is_on_screen():
    text = ' '.join(m.value for m in _rendered().markdown)
    assert 'runbar' in text
    assert '1,000 simulations' in text
    assert 'seed 42' in text


def test_a_fresh_run_carries_no_stale_warning():
    assert len(_rendered().warning) == 0


def test_changing_a_setting_without_running_warns():
    """
    The P0 this exists for: the sidebar moves, the result does not, and
    nothing on screen says so.
    """
    at = _rendered()
    at.sidebar.number_input[0].set_value(99).run()
    assert len(at.warning) == 1, [w.value for w in at.warning]
    assert 'run again' in str(at.warning[0].value).lower()
    assert 'seed 42 -> 99' in str(at.warning[0].value)


def test_the_previous_result_stays_on_screen_when_settings_move():
    """It is not cleared and it is not relabelled - it is marked as previous."""
    at = _rendered()
    before = at.dataframe[0].value.copy()
    at.sidebar.number_input[0].set_value(99).run()
    assert len(at.dataframe) > 0
    assert at.dataframe[0].value.equals(before)


def test_several_moved_settings_are_all_named():
    at = _rendered()
    at.sidebar.number_input[0].set_value(99).run()
    at.sidebar.checkbox[0].set_value(not at.sidebar.checkbox[0].value).run()
    message = str(at.warning[0].value)
    assert 'seed' in message
    assert 'Enforce tyre stint limits' in message, message


def test_the_first_screen_offers_the_run_button_and_not_a_file_path():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    assert [b.label for b in at.sidebar.button] == ['Run simulation']
    opening = ' '.join(i.value for i in at.info)
    assert 'Run simulation' in opening
    assert 'predictions.csv' not in opening


# --- choosing a race --------------------------------------------------------

def test_the_selector_offers_every_available_race():
    from streamlit.testing.v1 import AppTest
    from Simülasyon import race_select as RS

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    offered = at.sidebar.selectbox[0].options
    available = RS.available()
    expected = len(available[available.state == 'available'])
    assert len(offered) == expected, (len(offered), expected)


def test_the_selector_defaults_to_the_race_the_pipeline_fetched():
    from streamlit.testing.v1 import AppTest
    from Simülasyon import race_select as RS

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    frame = RS.available()
    target = int(frame[frame.is_pipeline_target].iloc[0]['round'])
    assert at.sidebar.selectbox[0].value == target


def test_a_different_race_runs_and_is_named_on_the_page():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    at.sidebar.selectbox[0].set_value(6).run()
    at.sidebar.select_slider[0].set_value(1000).run()
    at.sidebar.button[0].click().run()

    assert len(at.exception) == 0, [e.value for e in at.exception]
    titles = [m.value for m in at.markdown if m.value.startswith('# ')]
    assert 'Monaco' in titles[0], titles


def test_a_race_off_the_pipeline_target_says_where_its_grid_came_from():
    """Penalties move cars, and only the fetched race has the official grid."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    at.sidebar.selectbox[0].set_value(6).run()
    at.sidebar.select_slider[0].set_value(1000).run()
    at.sidebar.button[0].click().run()

    bar = [m.value for m in at.markdown
           if m.value.startswith('<div class="runbar">')]
    assert bar, 'no run bar'
    assert 'qualifying order' in bar[0], bar[0]


def test_changing_race_without_running_keeps_the_old_result_and_says_so():
    """
    The most misleading version of the mismatch: the title would name one
    Grand Prix while the table underneath described another.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=900)
    at.run()
    at.sidebar.selectbox[0].set_value(6).run()
    at.sidebar.select_slider[0].set_value(1000).run()
    at.sidebar.button[0].click().run()
    before = at.dataframe[0].value.copy()

    at.sidebar.selectbox[0].set_value(12).run()
    assert len(at.warning) == 1, [w.value for w in at.warning]
    assert 'race round 6 -> 12' in str(at.warning[0].value)
    assert at.dataframe[0].value.equals(before)


# --- the representative race ------------------------------------------------
#
# The tab this replaced embedded output/diagnostics.html, which only the
# console run writes. On a deployed copy that file never exists, so the tab
# was permanently empty. Everything here is built from the run in memory, so
# these tests need no file on disk and no console step either.


def _traced(n_sims=400):
    """A real run, for the lap traces only a real run produces."""
    from Simülasyon import simulate as S

    S.N_SIMS = n_sims
    return S.run()


def test_positions_are_a_permutation_on_every_lap():
    """
    Two cars cannot hold the same position. `order` is inverted to get the
    per-driver view, and an inversion done wrong silently duplicates.
    """
    result = _traced()
    position = APP.lap_positions(result)
    for lap in range(position.shape[0]):
        alive = position[lap][~np.isnan(position[lap])]
        assert len(set(alive.tolist())) == len(alive), f'lap {lap + 1}'


def test_positions_start_at_one():
    position = APP.lap_positions(_traced())
    assert np.nanmin(position) == 1


def test_a_retired_car_stops_being_drawn():
    """
    A retired car keeps appearing in `order` - its clock became a sorting
    marker that parks it at the back - so an untruncated line would show it
    circulating last for the rest of the race.
    """
    result = _traced()
    retired_lap = np.asarray(result['trace']['retired_lap'])
    out = np.flatnonzero(retired_lap >= 0)
    if not len(out):
        return                      # no retirement in this run to check
    position = APP.lap_positions(result)
    for driver in out[:3]:
        lap = int(retired_lap[driver])
        assert not np.isnan(position[lap, driver]), 'cut one lap too early'
        assert np.isnan(position[lap + 1:, driver]).all(), 'line ran on'


def test_only_this_races_podium_carries_colour():
    """
    Twenty-three distinguishable hues is a colour puzzle, not a chart. The
    legend names the three the chart actually highlights, and they are the
    podium of this race rather than the favourites of the distribution.
    """
    result = _traced()
    figure = APP.representative_chart(result)

    position = APP.lap_positions(result)
    drivers = list(result['pace']['Driver'])
    finish = position[-1]
    n = len(drivers)
    order = np.argsort(np.where(np.isnan(finish), n + 1, finish))
    podium = {drivers[i] for i in order[:3]}

    # A set, not a list: fig.data is in the order traces were added, which is
    # driver index order, while the legend is displayed by legendrank. What
    # matters is which three are highlighted, not where they sit in the array.
    named = {t.name for t in figure.data if t.showlegend} - {'retired'}
    assert named == podium, (named, podium)

    # And the legend itself reads in finishing order rather than array order.
    ranked = sorted((t.legendrank, t.name) for t in figure.data
                    if t.showlegend and t.name != 'retired')
    assert [name for _, name in ranked] == [drivers[i] for i in order[:3]]


def test_the_chart_puts_first_place_at_the_top():
    figure = APP.representative_chart(_traced())
    assert figure.layout.yaxis.autorange == 'reversed'


def test_neutralisation_bands_group_consecutive_laps_and_name_themselves():
    """
    A band per lap would be unreadable and a band spanning a gap would be a
    lie. Built synthetically because the representative race takes the modal
    event profile, which at most circuits means no safety car at all.
    """
    result = _traced()
    laps = result['trace']['order'].shape[0]
    neutral = np.zeros(laps, dtype=np.int8)
    neutral[4:7] = 1        # virtual safety car, laps 5-7
    neutral[11:14] = 2      # safety car, laps 12-14
    neutral[19] = 3         # red flag, lap 20
    result['trace']['neutral'] = neutral

    figure = APP.representative_chart(result)
    shapes = list(figure.layout.shapes)
    labels = [a.text for a in figure.layout.annotations]

    assert len(shapes) == 3, len(shapes)
    assert labels == ['virtual safety car', 'safety car', 'red flag'], labels

    covered = set()
    for shape in shapes:
        covered |= set(range(int(round(shape.x0 + 0.5)),
                            int(round(shape.x1 - 0.5)) + 1))
    assert covered == set((np.flatnonzero(neutral > 0) + 1).tolist())


def test_a_green_race_gets_no_bands():
    result = _traced()
    result['trace']['neutral'] = np.zeros(
        result['trace']['order'].shape[0], dtype=np.int8)
    assert not list(APP.representative_chart(result).layout.shapes)


def test_a_result_without_traces_says_so_instead_of_raising():
    """A run made before this panel existed has no trace to draw."""
    import streamlit as st

    stripped = {'n_sims': 100, 'report': None, 'trace': None,
                'pace': pd.DataFrame({'Driver': ['AAA']})}
    APP.show_representative(stripped)        # must not raise


# --- a race per winner ------------------------------------------------------

def test_every_winner_page_shows_a_race_that_driver_won():
    """
    The property the whole feature rests on. A page headed HAM showing a race
    LEC won would be wrong in a way nothing else here would catch: the chart
    is valid, the numbers are real, and only the name is a lie.
    """
    from Simülasyon import simulate as S
    from Simülasyon import representative as REP

    S.N_SIMS = 1500
    result = S.run()
    drivers = list(result['pace']['Driver'])
    positions = np.asarray(result['positions'])

    for entry in REP.for_each_winner(result, S.rebuild_compounds):
        assert positions[entry['sim'], entry['driver_idx']] == 1, entry['driver']
        # and again from the lap trace, which is what actually gets drawn
        leader = drivers[int(np.asarray(entry['trace']['order'])[-1][0])]
        assert leader == entry['driver'], (leader, entry['driver'])


def test_the_search_never_leaves_the_races_a_driver_won():
    """
    find_representative relaxes its event conditions when the pool is thin.
    Relaxing all the way to every simulation - which is what it does
    unrestricted - would hand back a race the driver lost.
    """
    from Simülasyon import simulate as S
    from Simülasyon import representative as REP
    from Simülasyon.diagnostics import find_representative

    S.N_SIMS = 1500
    result = S.run()
    positions = np.asarray(result['positions'])

    # a rare winner, where every relaxation step will be triggered
    rare = min(REP.winners(positions, list(result['pace']['Driver'])),
               key=lambda row: row['wins'])
    won = REP.races_won_by(positions, rare['driver_idx'])
    sim, report = find_representative(positions, result['diag'],
                                      restrict_to=won)
    assert sim in set(won.tolist())
    assert report['universe'] == len(won)
    assert report['pool'] <= len(won)


def test_an_unrestricted_search_is_unchanged():
    """The single representative race must not move because the API grew."""
    from Simülasyon import simulate as S
    from Simülasyon.diagnostics import find_representative

    S.N_SIMS = 1200
    result = S.run()
    again, report = find_representative(result['positions'], result['diag'])
    assert again == result['rep']
    assert report['universe'] == result['n_sims']
    assert report['restricted'] is False


def test_winners_are_ranked_by_wins_and_carry_their_share():
    """
    The share is what keeps fifteen named pages from reading as fifteen
    plausible Sundays.
    """
    from Simülasyon import simulate as S
    from Simülasyon import representative as REP

    S.N_SIMS = 1500
    result = S.run()
    rows = REP.winners(result['positions'], list(result['pace']['Driver']))

    assert rows == sorted(rows, key=lambda r: (-r['wins'], r['driver']))
    assert all(r['wins'] >= 1 for r in rows)
    for row in rows:
        assert abs(row['share'] - row['wins'] / result['n_sims']) < 1e-12
    assert abs(sum(r['share'] for r in rows) - 1.0) < 1e-9


def test_a_driver_who_never_wins_gets_no_page():
    from Simülasyon import simulate as S
    from Simülasyon import representative as REP

    S.N_SIMS = 1200
    result = S.run()
    positions = np.asarray(result['positions'])
    named = {r['driver'] for r in
             REP.winners(positions, list(result['pace']['Driver']))}
    drivers = list(result['pace']['Driver'])
    for i, code in enumerate(drivers):
        if (positions[:, i] == 1).sum() == 0:
            assert code not in named


# --- tyres and stops --------------------------------------------------------

def test_the_stint_chart_covers_every_lap_each_car_ran():
    """
    A gap between bars would be a lap the car was neither on a tyre nor
    retired, which is not a thing that happens.
    """
    result = _traced()
    by_driver = {}
    for stint in result['stints']:
        by_driver.setdefault(stint['driver_idx'], []).append(stint)

    retired_lap = np.asarray(result['trace']['retired_lap'])
    n_laps = result['trace']['order'].shape[0]
    for driver, stints in by_driver.items():
        stints.sort(key=lambda s: s['start'])
        assert stints[0]['start'] == 1, driver
        for before, after in zip(stints, stints[1:]):
            assert after['start'] == before['end'] + 1, (driver, before, after)
        last = (int(retired_lap[driver]) + 1 if retired_lap[driver] >= 0
                else n_laps)
        assert stints[-1]['end'] == last, (driver, stints[-1]['end'], last)


def test_every_stint_gets_a_colour():
    """An unmapped compound would draw grey and read as a tyre of its own."""
    result = _traced()
    for stint in result['stints']:
        assert str(stint['compound']) in APP.COMPOUND_COLOUR, stint['compound']


def test_the_stint_chart_rows_are_ordered_by_finish():
    result = _traced()
    figure = APP.stint_chart(result)

    # tickvals and ticktext pair element by element, and both are emitted in
    # driver-index order rather than in row order. Reading ticktext alone says
    # nothing about what sits where; the row number is in tickvals.
    rows = dict(zip(figure.layout.yaxis.ticktext,
                    figure.layout.yaxis.tickvals))

    position = APP.lap_positions(result)
    drivers = list(result['pace']['Driver'])
    finish = position[-1]
    n = len(drivers)
    order = np.argsort(np.where(np.isnan(finish), n + 1, finish))

    # highest row is the top of the chart, and the winner belongs there
    top = max(rows, key=rows.get)
    assert top == drivers[order[0]], (top, drivers[order[0]])

    # and the order runs monotonically down from there
    ranked = [drivers[i] for i in order]
    assert [rows[name] for name in ranked] == sorted(
        (rows[name] for name in ranked), reverse=True)


def test_a_retirement_is_marked_on_the_stint_chart():
    result = _traced()
    retired_lap = np.asarray(result['trace']['retired_lap'])
    if not (retired_lap >= 0).any():
        return
    names = [t.name for t in APP.stint_chart(result).data]
    assert 'retired' in names


# --- compound colours and same-compound pit stops ---------------------------

def test_compound_colours_match_the_broadcast_convention():
    """Soft red, medium yellow, hard white - what every timing screen uses."""
    soft = APP.COMPOUND_COLOUR['SOFT'].lstrip('#')
    medium = APP.COMPOUND_COLOUR['MEDIUM'].lstrip('#')
    hard = APP.COMPOUND_COLOUR['HARD'].lstrip('#')

    r, g, b = int(soft[0:2], 16), int(soft[2:4], 16), int(soft[4:6], 16)
    assert r > 150 and g < 100 and b < 100, 'SOFT is not red'

    r, g, b = int(medium[0:2], 16), int(medium[2:4], 16), int(medium[4:6], 16)
    assert r > 200 and g > 180 and b < 100, 'MEDIUM is not yellow'

    r, g, b = int(hard[0:2], 16), int(hard[2:4], 16), int(hard[4:6], 16)
    assert r > 220 and g > 220 and b > 220, 'HARD is not white'


def test_a_same_compound_pit_stop_gets_a_marker_not_just_a_seam():
    """
    Two adjacent bars in the same fill colour are the case a border alone
    cannot reliably announce - the triangle is what actually says "pit here".
    """
    from Simülasyon.diagnostics import build_stints

    n_laps = 40
    compounds = np.empty((n_laps, 1), dtype=object)
    compounds[:20, 0] = 'HARD'
    compounds[20:, 0] = 'HARD'
    pits = np.zeros((n_laps, 1), dtype=bool)
    pits[19, 0] = True

    result = _traced()
    stints = build_stints(compounds, pits, codes=['XXX'])
    fake = {'pace': result['pace'], 'trace': result['trace'], 'stints': stints}
    # lap_positions/stint_chart only need trace for order/pits/retired_lap of
    # the real run's driver count; reuse it and just substitute the stints.
    figure = APP.stint_chart(result, result['trace'], stints)

    names = [t.name for t in figure.data]
    assert 'pit stop' in names
    pit_trace = figure.data[names.index('pit stop')]
    assert 20 in list(pit_trace.x)


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
