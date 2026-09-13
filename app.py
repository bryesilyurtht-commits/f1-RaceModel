"""
F1 Prediction Simulation - v1.6 - app.py
The interface.

    streamlit run app.py

What it is for
--------------
The question this project keeps asking itself is "where did that number come
from". deg_overrides, shape_source, cap_source, sc_source, offset_source -
every one of those exists to answer it, and every one of them currently
scrolls past in a console and is gone.

So the interface is built around that answer rather than around the result.
Every parameter carries a badge saying whether it was measured here, derived
from another cell, borrowed from a donor circuit, shrunk toward the calendar,
or simply chosen by a person and still owed a measurement. A reader who
cannot tell those apart will trust the wrong one.

Visual language
---------------
Analytical. Light ground, flat type, one accent. Colour is never decoration:
it encodes the heat map's density, the direction of a gain or loss against
the grid, and the two badge states that want attention - hand-set and
fallback. Team colours appear nowhere, because a team colour carries no
value. Numbers are right-aligned with fixed decimals so columns compare by
eye.

Layers, in the order the roadmap set them
-----------------------------------------
    K0  results, table and heat map
    K1  running, from the sidebar, only on the button
    K2  flags and the parameter panel
    K3  two runs side by side, with the changed flags named
    K5  the existing diagnostics report, embedded unchanged

Runs live in session state and die with the tab. That was asked for.
"""

import io
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Simülasyon import simulate as S
from Simülasyon import target_race as TR
from Simülasyon import dnf as DNF
from Simülasyon import pipeline as PIPE
from Simülasyon import weather as WX

OUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

MAX_RUNS = 3          # kept in memory for comparison; the roadmap's number

# --- visual language --------------------------------------------------------

INK = '#16181d'
MUTED = '#6b7280'
HAIRLINE = '#e3e5e9'
GROUND = '#ffffff'
ACCENT = '#2f5d8a'          # the one accent
ATTENTION = '#b4541f'       # reserved for hand-set and fallback only

# single hue, light to dark. Probability has no natural midpoint, so a
# diverging red-to-green ramp would invent one.
HEAT = [[0.0, '#f7f8fa'], [0.15, '#dce4ed'], [0.35, '#a9bed4'],
        [0.6, '#6e90b4'], [0.8, '#43678f'], [1.0, ACCENT]]

BADGE_STYLE = {
    'measured':  (MUTED, 'normal', 'normal'),
    'derived':   (MUTED, 'normal', 'italic'),
    'borrowed':  (MUTED, 'normal', 'italic'),
    'shrunk':    (MUTED, 'normal', 'italic'),
    'hand-set':  (ATTENTION, '600', 'normal'),
    'fallback':  (ATTENTION, '600', 'normal'),
}

BADGE_HELP = {
    'measured': 'read off this circuit\'s own data',
    'derived': 'computed from another cell rather than measured here',
    'borrowed': 'taken from a donor circuit',
    'shrunk': 'measured, then pulled toward the calendar because the sample '
              'is thin',
    'hand-set': 'chosen by a person, and still owed a measurement',
    'fallback': 'nothing was available',
}

CSS = f"""
<style>
  .stApp {{ background: {GROUND}; }}
  html, body, [class*="css"] {{
      font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: {INK};
  }}
  .block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
  h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.012em; color: {INK}; }}
  h1 {{ font-size: 1.55rem; margin-bottom: 0.1rem; }}
  h2 {{ font-size: 1.05rem; margin-top: 1.6rem; }}
  .eyebrow {{
      font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.09em;
      color: {MUTED}; font-weight: 600;
  }}
  .lede {{ color: {MUTED}; font-size: 0.88rem; margin: 0.15rem 0 0.9rem 0; }}
  .rule {{ border-top: 1px solid {HAIRLINE}; margin: 1.1rem 0 0.9rem 0; }}
  /* Which run is on screen. First thing under the title, so that reading a
     number and knowing where it came from are the same glance. */
  .runbar {{
      font-size: 0.80rem; color: {MUTED}; padding: 0.4rem 0.65rem;
      border: 1px solid {HAIRLINE}; border-radius: 3px;
      background: #fff; margin: 0.2rem 0 0.9rem 0;
      font-variant-numeric: tabular-nums;
  }}
  /* The one answer the page exists to give, before any chart. */
  .headline {{ display: flex; gap: 2.6rem; flex-wrap: wrap;
               margin: 0.1rem 0 0.2rem 0; }}
  .headline .k {{ font-size: 0.68rem; text-transform: uppercase;
                  letter-spacing: 0.08em; color: {MUTED}; font-weight: 600; }}
  .headline .v {{ font-size: 1.75rem; font-weight: 600; color: {INK};
                  font-variant-numeric: tabular-nums; line-height: 1.15; }}
  .headline .sub {{ font-size: 0.78rem; color: {MUTED}; }}
  /* Narrow screens: Streamlit keeps columns side by side well past the point
     where a chart is readable, so the two-up row is unwound by hand. Nothing
     is hidden - a cramped chart is still the data. */
  @media (max-width: 900px) {{
      .block-container {{ padding-left: 0.8rem; padding-right: 0.8rem; }}
      div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
          min-width: 100% !important; flex: 1 1 100% !important;
      }}
      .headline .v {{ font-size: 1.4rem; }}
      .statline {{ gap: 1.2rem; }}
  }}
  .badge {{
      display: inline-block; font-size: 0.68rem; letter-spacing: 0.02em;
      padding: 0.05rem 0.4rem; border: 1px solid {HAIRLINE}; border-radius: 2px;
  }}
  .paramrow {{
      display: grid; grid-template-columns: 180px 1fr 120px;
      gap: 0.6rem; padding: 0.28rem 0; border-bottom: 1px solid {HAIRLINE};
      font-size: 0.83rem; align-items: baseline;
  }}
  .paramrow .name {{ font-weight: 500; }}
  .paramrow .val {{
      font-variant-numeric: tabular-nums; color: {MUTED}; text-align: right;
  }}
  .statline {{ display: flex; gap: 2.2rem; flex-wrap: wrap; margin: 0.2rem 0 0.4rem 0; }}
  .stat .k {{ font-size: 0.68rem; text-transform: uppercase;
              letter-spacing: 0.08em; color: {MUTED}; }}
  .stat .v {{ font-size: 1.25rem; font-variant-numeric: tabular-nums;
              font-weight: 600; }}
  code, pre {{ font-size: 0.78rem; }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid {HAIRLINE}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
  [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {HAIRLINE}; }}
</style>
"""


def badge(kind):
    colour, weight, style = BADGE_STYLE.get(kind, (MUTED, 'normal', 'normal'))
    return (f'<span class="badge" style="color:{colour};font-weight:{weight};'
            f'font-style:{style}">{kind}</span>')


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


def statline(pairs):
    cells = ''.join(
        f'<div class="stat"><div class="k">{k}</div>'
        f'<div class="v">{v}</div></div>' for k, v in pairs)
    st.markdown(f'<div class="statline">{cells}</div>', unsafe_allow_html=True)


# --- naming -----------------------------------------------------------------

# What each switch is called on screen. The code name stays in the help text,
# because it is what a bug report and the run summary will both say - but it
# should not be the first thing a reader has to decode.
#
# Each label was checked against what the flag does in simulate.py rather than
# against its name. LOCK_START_ORDER is the one that repays that: it reads as
# a start-only setting and is passed to build_start_gaps at the race start and
# again at a red-flag restart, so the help text says both.
FLAG_LABELS = {
    'ENFORCE_STINT_CAP': 'Enforce tyre stint limits',
    'RECOST_STRATEGIES': 'Recalculate strategy costs',
    'REACTIVE_PIT_V2': 'Decide pit laps against rivals',
    'REACTIVE_PIT': 'Decide pit laps on tyre wear',
    'PER_DRIVER_STRATEGY': 'Let strategy vary by driver',
    'NEUTRAL_FREEZES_GAPS': 'Freeze gaps under neutralisation',
    'SC_BUNCHING': 'Bunch the field behind a safety car',
    'RED_FLAG_ENABLED': 'Allow red flags',
    'NEUTRAL_LAST_LAP_KNOWN': 'Let strategy see the safety car schedule',
    'START_MODEL': 'Race the start',
    'LOCK_START_ORDER': 'Keep opening order fixed',
    'DNF_ENABLED': 'Include retirements',
    'TWO_CAUSE_DNF': 'Separate crashes from failures',
    'ACCIDENT_NEUTRALIZATION': 'Let crashes bring out flags',
    'EVOLUTION_ENABLED': 'Include track evolution',
    'AFFINITY_ENABLED': 'Include circuit suitability',
    'AFFINITY_TEAM_2026': 'Use 2026 team suitability',
    'GAP_IN_PASS_MODEL': 'Use gap in overtaking model',
    'TEAM_PASS_ENABLED': 'Include team overtaking differences',
    'IS_WET': 'Start the race on a wet track',
    'WEATHER_ENABLED': 'Include weather',
    'WEATHER_MAY_BREAK_PLAN': 'Let weather break the strategy',
}

# Extra scope the flag's own description does not carry.
FLAG_NOTES = {
    'LOCK_START_ORDER': 'Applies to the race start and to a red-flag '
                        'restart. It does not touch the real prohibitions on '
                        'passing under a safety car or virtual safety car, '
                        'which are always in force.',
    'START_MODEL': 'Calibrated on 1313 measured starts. Off, the opening lap '
                   'keeps the grid order, which is what this project did '
                   'before v2.7.',
}


def flag_label(name):
    return FLAG_LABELS.get(name, name)


def flag_help(name):
    """The flag's own description, its code name, and any extra scope."""
    parts = [S.RUNTIME_FLAGS.get(name, '')]
    if name in FLAG_NOTES:
        parts.append(FLAG_NOTES[name])
    parts.append(f'Code name: {name}')
    return '\n\n'.join(p for p in parts if p)


# --- which run am I looking at ----------------------------------------------


def settings_signature(n_sims, seed, flags, scenario):
    """
    Everything that changes a result, as one comparable value.

    Streamlit re-runs the script on every widget change, so the sidebar can
    say one thing while the result on screen came from another. Keeping the
    settings that produced a result beside it is what lets the page notice.
    """
    return {'n_sims': int(n_sims), 'seed': int(seed),
            'flags': dict(sorted(flags.items())), 'scenario': scenario}


def settings_drift(run, live):
    """
    What the sidebar now says that the shown result did not use.

    Returns a list of human-readable differences, empty when the result still
    matches the controls. A silent mismatch is the failure this exists for:
    the result would keep its old label while looking like an answer to the
    new question.
    """
    was = run.get('settings')
    if not was:
        return []

    drift = []
    if was['n_sims'] != live['n_sims']:
        drift.append(f'simulations {was["n_sims"]:,} -> {live["n_sims"]:,}')
    if was['seed'] != live['seed']:
        drift.append(f'seed {was["seed"]} -> {live["seed"]}')
    if was['scenario'] != live['scenario']:
        drift.append(f'weather {was["scenario"] or "off"} -> '
                     f'{live["scenario"] or "off"}')
    for name in sorted(set(was['flags']) | set(live['flags'])):
        before, after = was['flags'].get(name), live['flags'].get(name)
        if before != after:
            drift.append(f'{flag_label(name)} {"on" if before else "off"} -> '
                         f'{"on" if after else "off"}')
    return drift


def run_header(run, live=None):
    """
    One line at the top of a result saying what it is, before anything else.

    The race, where the numbers came from, how many runs finished, and - when
    the controls have moved since - that this is no longer an answer to what
    the sidebar is asking.
    """
    result = run['result']
    summary = run.get('run_summary') or {}
    completed = summary.get('completed_runs', result.get('n_sims'))
    requested = summary.get('requested_runs', completed)

    bits = [f'{result["event"]} {result["season"]}',
            f'{completed:,} simulations']
    if requested and completed != requested:
        bits[-1] = f'{completed:,} of {requested:,} simulations'
    bits.append(f'seed {run["settings"]["seed"]}'
                if run.get('settings') else 'seed unknown')
    bits.append(run.get('source', 'run in this session'))

    st.markdown(
        '<div class="runbar">' + ' &nbsp;·&nbsp; '.join(bits) + '</div>',
        unsafe_allow_html=True)

    drift = settings_drift(run, live) if live else []
    if drift:
        st.warning('Settings changed - run again to update results. '
                   'Showing the previous run, which used: '
                   + '; '.join(drift[:6])
                   + ('; and more' if len(drift) > 6 else '') + '.')
    return drift


# --- data on disk -----------------------------------------------------------


def data_target():
    """Which race the CSVs on disk were built for, from the poles table."""
    path = os.path.join(DATA_DIR, f'f1_{S.SEASON}_poles.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'is_target' not in df.columns:
        return None
    hit = df[df['is_target'].astype(bool)]
    return str(hit.iloc[0]['Race']) if not hit.empty else None


def target_matches():
    """
    Whether the pipeline outputs belong to the race the module is pointed at.

    fetch.py withholds one race and downloads its grid; clean.py weights pace
    around that split. Pointing simulate.py somewhere else without re-running
    them produces a confident answer built on another circuit's grid, which
    is the sort of wrong that looks right.
    """
    on_disk = data_target()
    if on_disk is None:
        return None, None
    wanted = S.TARGET_EVENT
    ok = any(a in on_disk.lower() for a in S.TRACK_ALIASES)
    return ok, on_disk


# --- figures ----------------------------------------------------------------


def heatmap(dist, order):
    import plotly.graph_objects as go

    dist = dist.loc[order]
    n = dist.shape[1]
    fig = go.Figure(go.Heatmap(
        z=dist.to_numpy(), x=list(range(1, n + 1)), y=list(dist.index),
        colorscale=HEAT, zmin=0, zmax=float(dist.to_numpy().max()),
        xgap=1, ygap=1,
        hovertemplate='%{y} finishes P%{x} in %{z:.1%} of races<extra></extra>',
        colorbar=dict(title=dict(text='share of races', side='right',
                                 font=dict(size=10, color=MUTED)),
                      thickness=9, len=0.55, outlinewidth=0,
                      tickfont=dict(size=9, color=MUTED), tickformat='.0%'),
    ))
    fig.update_layout(
        height=26 * len(dist) + 90,
        margin=dict(l=0, r=0, t=8, b=28),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        font=dict(color=INK, size=11),
        xaxis=dict(title=dict(text='finishing position', font=dict(size=10,
                                                                  color=MUTED)),
                   side='bottom', tickfont=dict(size=9, color=MUTED),
                   showgrid=False, zeroline=False, dtick=1),
        yaxis=dict(autorange='reversed', tickfont=dict(size=10),
                   showgrid=False, zeroline=False),
    )
    return fig


def grid_delta_chart(summary, pace):
    """Positions gained or lost against the grid. Colour carries the sign."""
    import plotly.graph_objects as go

    grid = pace.set_index('Driver')['grid']
    d = summary.copy()
    d['grid'] = d['Driver'].map(grid)
    d['delta'] = d['grid'] - d['mean_pos']
    d = d.sort_values('delta')

    fig = go.Figure(go.Bar(
        x=d['delta'], y=d['Driver'], orientation='h',
        marker=dict(color=[ACCENT if v >= 0 else ATTENTION for v in d['delta']]),
        hovertemplate='%{y}: %{x:+.2f} positions<extra></extra>',
    ))
    fig.update_layout(
        height=22 * len(d) + 70, margin=dict(l=0, r=0, t=8, b=28),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        font=dict(color=INK, size=11), showlegend=False,
        # Colour repeats the sign, it does not carry it alone: the axis names
        # both directions, the hover is signed, and the table above holds the
        # same numbers for anyone the colours do not separate.
        xaxis=dict(title=dict(text='← lost   ·   expected positions vs grid   '
                                   '·   gained →',
                              font=dict(size=10, color=MUTED)),
                   zeroline=True, zerolinecolor=HAIRLINE, zerolinewidth=1,
                   gridcolor=HAIRLINE, tickfont=dict(size=9, color=MUTED)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
    )
    return fig


def share_card(result):
    """
    One figure, vertical, readable on a phone. Plotly's own menu saves it.

    The stack is win and podium-minus-win, so the bar ends at the podium
    probability rather than at win plus podium. Winning is already a podium,
    and adding the two would put a driver on 102%.

    That arithmetic was always right and the labels were not: the second
    segment was called "podium" when it is the rest of the podium, and its
    hover showed no number at all. A reader has only the legend to go on, so
    a correct chart with those labels reads as the wrong one. The segments
    now say what they are and both hovers carry their value.
    """
    import plotly.graph_objects as go

    d = result['summary'].head(10).copy()
    d = d.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(x=d['P_win'], y=d['Driver'], orientation='h', name='Win',
                marker_color=ACCENT,
                customdata=d['P_podium'],
                hovertemplate='%{y}<br>win %{x:.1%}<br>'
                              'podium %{customdata:.1%}<extra></extra>')
    fig.add_bar(x=d['P_podium'] - d['P_win'], y=d['Driver'], orientation='h',
                name='P2-P3', marker_color='#b9c8d8',
                customdata=d['P_podium'],
                hovertemplate='%{y}<br>second or third %{x:.1%}<br>'
                              'podium %{customdata:.1%}<extra></extra>')
    fig.update_layout(
        barmode='stack', height=460, width=520,
        margin=dict(l=0, r=12, t=58, b=34),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        # The image travels without the page around it, so it has to say what
        # it is on its own: which race, how many runs, and that the bar ends
        # at the podium probability rather than at a sum of two.
        title=dict(text=f'{result["event"]} {result["season"]}<br>'
                        f'<span style="font-size:11px;color:{MUTED}">'
                        f'{result["n_sims"]:,} simulations &nbsp;·&nbsp; '
                        f'bar length is the podium probability</span>',
                   font=dict(size=15, color=INK), x=0, xanchor='left'),
        legend=dict(orientation='h', y=-0.10, x=0, font=dict(size=10,
                                                             color=MUTED),
                    bgcolor='rgba(0,0,0,0)'),
        font=dict(color=INK, size=11),
        xaxis=dict(tickformat='.0%', gridcolor=HAIRLINE,
                   tickfont=dict(size=9, color=MUTED), zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    return fig


# --- panels -----------------------------------------------------------------


def show_parameters(result):
    counts = {}
    for b in result['badges']:
        counts[b['kind']] = counts.get(b['kind'], 0) + 1

    eyebrow('where every number came from')
    st.markdown(
        f'<div class="lede" style="font-size:0.82rem">Parameter set '
        f'<b>{S.PARAM_SET_VERSION}</b>. Every constant in the model is '
        f'inventoried in <code>Simülasyon/parameters.py</code> with what kind '
        f'of number it is, what it was measured against, and how much the '
        f'result moves when it moves.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lede">These say where a number came from, not how good '
        'it is. A measured value can rest on a thin sample and a hand-set one '
        'can be close; what the labels give you is the ability to ask. The '
        'two that most often want a second look are marked in colour.</div>',
        unsafe_allow_html=True)

    legend = '  '.join(
        f'{badge(k)} <span style="font-size:0.74rem;color:{MUTED}">'
        f'{BADGE_HELP[k]}{f" &middot; {counts[k]}" if k in counts else ""}</span>'
        for k in ('measured', 'derived', 'borrowed', 'shrunk', 'hand-set',
                  'fallback'))
    st.markdown(f'<div style="line-height:2.0">{legend}</div>',
                unsafe_allow_html=True)
    rule()

    rows = []
    for b in result['badges']:
        # a source that only repeats its own badge adds nothing to the row
        detail = '' if b['source'].strip().lower() == b['kind'] else b['source']
        rows.append(
            f'<div class="paramrow"><div class="name">{b["name"]}</div>'
            f'<div>{badge(b["kind"])} '
            f'<span style="color:{MUTED};font-size:0.78rem">{detail}</span>'
            f'</div><div class="val">{b["value"]}</div></div>')
    st.markdown(''.join(rows), unsafe_allow_html=True)

    # The fit descriptions and the full switch list are a step down, not gone:
    # they are why someone technical opens this tab, and they are also what
    # makes the first screen unreadable for everyone else.
    st.markdown('')
    with st.expander('Technical details - every setting this run used'):
        st.code('\n'.join(result['param_lines']), language=None)


def headline(result):
    """
    The answer, before any chart.

    Who is most likely to win, who is most likely to reach the podium, and
    how many races that was read off. The two can be different drivers and
    are not forced to agree - a car that wins often and retires often is a
    real thing, and hiding it would be the point of the page going missing.
    """
    summary = result['summary']
    win = summary.loc[summary['P_win'].idxmax()]
    podium = summary.loc[summary['P_podium'].idxmax()]

    cells = [('most likely winner',
              f'{win["Driver"]} {win["P_win"]:.1%}',
              str(win['Team'])),
             ('most likely podium',
              f'{podium["Driver"]} {podium["P_podium"]:.1%}',
              str(podium['Team'])),
             ('simulations', f'{result["n_sims"]:,}',
              f'{result["n_laps"]} laps')]
    html = ''.join(f'<div><div class="k">{k}</div>'
                   f'<div class="v">{v}</div>'
                   f'<div class="sub">{s}</div></div>' for k, v, s in cells)
    st.markdown(f'<div class="headline">{html}</div>', unsafe_allow_html=True)


def results_table(result):
    """
    The predicted order, as a table a person can read.

    Probabilities as percentages rather than as 0.317, sorted by expected
    finish, and the columns most readers want first. The rest - causes of
    retirement, stop count, circuit suitability, the strategy string - are a
    checkbox away rather than gone, because they are the reason someone
    technical opens this at all.

    Sorting is done on the numbers and the formatting is display only, so a
    percentage column still orders numerically rather than alphabetically.
    """
    summary = result['summary']
    pace = result['pace']

    show = summary.copy()
    if 'grid' not in show.columns:
        show['grid'] = show['Driver'].map(pace.set_index('Driver')['grid'])

    # why a car did not finish, not only whether. The two columns and the
    # finishing probability are the whole of it - there is no third cause the
    # simulation can produce.
    causes = result.get('dnf_by_driver')
    if causes is not None:
        show = show.merge(causes, on='Driver', how='left')

    show = show.sort_values('mean_pos').reset_index(drop=True)

    base = ['Driver', 'Team', 'grid', 'mean_pos', 'P_win', 'P_podium',
            'P_points']
    extra = ['P_dnf', 'P_accident', 'P_mechanical', 'stops', 'affinity',
             'strategy']

    more = st.checkbox('More columns', value=False, key='more_columns',
                       help='Retirement causes, stop count, circuit '
                            'suitability and the chosen strategy.')
    cols = base + (extra if more else [])
    cols = [c for c in cols if c in show.columns]

    percent = st.column_config.NumberColumn
    st.dataframe(
        show[cols], width='stretch', hide_index=True,
        column_config={
            'grid': st.column_config.NumberColumn('Grid', format='%d'),
            'mean_pos': percent(
                'Expected finish', format='%.1f',
                help='The average finishing position across every simulation. '
                     'It is a mean, not a position anyone finishes in: a car '
                     'that either wins or retires averages out in the middle.'),
            'P_win': percent('Win', format='%.1f%%'),
            'P_podium': percent(
                'Podium', format='%.1f%%',
                help='Finishing in the top three. Winning is already included '
                     'in this, so podium is never smaller than win.'),
            'P_points': percent('Points', format='%.1f%%',
                                help='Finishing in the top ten.'),
            'P_dnf': percent('Retires', format='%.1f%%'),
            'P_accident': percent('Crash', format='%.1f%%'),
            'P_mechanical': percent('Failure', format='%.1f%%'),
            'stops': percent('Stops', format='%.2f'),
        })


def show_results(result):
    summary = result['summary']
    pace = result['pace']

    headline(result)
    rule()

    eyebrow('predicted results')
    st.markdown(
        '<div class="lede">Sorted by expected finish. Percentages are read '
        'off the simulations, not off a single race.</div>',
        unsafe_allow_html=True)
    results_table(result)

    rule()
    left, right = st.columns([1.02, 1])

    with left:
        eyebrow('finishing position probabilities')
        st.markdown(
            '<div class="lede">Each row is one car, each column a finishing '
            'position. A tight band is a predictable car; a smear is one whose '
            'race depends on how the day goes.</div>',
            unsafe_allow_html=True)
        dist = S.position_distribution(result['positions'], pace['Driver'])
        st.plotly_chart(heatmap(dist, summary['Driver'].tolist()),
                        width='stretch',
                        config={'displaylogo': False})

    with right:
        eyebrow('against the grid')
        st.markdown(
            '<div class="lede">Expected positions gained or lost relative to '
            'the starting grid. Positive is a gain. This is what the model '
            'expects to happen, not a score against what did - the race has '
            'not been run.</div>',
            unsafe_allow_html=True)
        st.plotly_chart(grid_delta_chart(summary, pace),
                        width='stretch',
                        config={'displaylogo': False})

    show_retirements(result)

    rule()
    eyebrow('race context')
    winner = float(result['total'].min(axis=1).mean())
    grid_arr = pace['grid'].to_numpy()
    # Absolute, not signed: this is how far the model moves cars, which is a
    # description of the simulation and not a measure of whether it is right.
    shift = float(np.abs(result['positions'] - grid_arr[None, :]).mean())
    statline([
        ('laps', f'{result["n_laps"]}'),
        ('pole', f'{S.POLE_DRIVER} {S.POLE_TIME:.3f}s'),
        ('mean winning time', f'{winner / 60:.1f} min'),
        ('mean move from grid', f'{shift:.2f} pos'),
    ])
    st.markdown(
        '<div class="lede">Mean move from grid is the average distance a car '
        'ends up from where it started, ignoring direction. It says how much '
        'the model shuffles the field, not how often it is right. Accuracy is '
        'not measured here - see docs/BACKTEST.md.</div>',
        unsafe_allow_html=True)

    rule()
    eyebrow('shareable')
    st.markdown(
        '<div class="lede">One image with the race, the run count and nothing '
        'else. Bars end at the podium probability.</div>',
        unsafe_allow_html=True)
    with st.expander('Show chart to download', expanded=False):
        st.plotly_chart(share_card(result), width='content',
                        config={'displaylogo': False,
                                'toImageButtonOptions': {
                                    'scale': 2,
                                    'filename': f'{result["event"]}_'
                                                f'{result["season"]}'}})
        st.markdown(
            '<div class="lede">Hover the chart and use the camera icon to '
            'save a PNG.</div>', unsafe_allow_html=True)


def retirement_chart(summary, n_laps):
    """
    When each kind of retirement happens, as a share of the race.

    Two series, because the interesting thing is whether they land in
    different places. They do not, as it turns out - which is itself the
    finding that killed the bell-shaped hazard this version was going to use.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for label, key, colour in (('accident', 'accident_laps', ATTENTION),
                               ('mechanical', 'mechanical_laps', ACCENT)):
        laps = summary.get(key)
        if laps is None or len(laps) == 0:
            continue
        counts, edges = np.histogram(laps, bins=12, range=(0, n_laps))
        centres = (edges[:-1] + edges[1:]) / 2
        fig.add_scatter(x=centres, y=counts / max(counts.sum(), 1),
                        mode='lines', name=label,
                        line=dict(color=colour, width=2),
                        hovertemplate=label + ' lap %{x:.0f}: %{y:.1%}'
                                              '<extra></extra>')

    fig.update_layout(
        height=190, margin=dict(l=0, r=0, t=6, b=26),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        font=dict(color=INK, size=11),
        legend=dict(orientation='h', y=1.18, x=0, font=dict(size=10)),
        xaxis=dict(title=dict(text='lap', font=dict(size=10, color=MUTED)),
                   gridcolor=HAIRLINE, tickfont=dict(size=9, color=MUTED)),
        yaxis=dict(title=dict(text='share of that cause',
                              font=dict(size=10, color=MUTED)),
                   tickformat='.0%', gridcolor=HAIRLINE,
                   tickfont=dict(size=9, color=MUTED)))
    return fig


def show_retirements(result):
    """
    Why cars stopped, and how much of the real thing that covers.

    The coverage line is the one that matters. The model produces two causes
    and they add to the retirement rate it reports - but a third of the
    retirements in eight seasons of results say only "Retired", and that gap
    belongs next to the number rather than buried in a module.
    """
    d = result.get('dnf')
    if d is None:
        return

    rule()
    eyebrow('why cars stop')
    st.markdown(
        '<div class="lede">Two causes, measured per lap at risk rather than '
        'per race: accident risk follows the driver, mechanical risk follows '
        'the car. Finishing and the two causes are the whole of it.</div>',
        unsafe_allow_html=True)

    statline([
        ('finish', f'{d["p_finish"]:.1%}'),
        ('accident', f'{d["p_accident"]:.1%}'),
        ('mechanical', f'{d["p_mechanical"]:.1%}'),
        ('flags from crashes', f'{d["accident_neutralizations"]:.2f}'),
    ])

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(retirement_chart(d, result['n_laps']),
                        width='stretch', config={'displayModeBar': False})
    with right:
        rows = [
            ('Failure timing',
             'flat through the race' if not d['bell_supported']
             else 'bell-shaped',
             'the proposed bell lost to flat on AIC'),
            ('After a crash', '  '.join(f'{k} {v:.0%}' for k, v
                                        in d['outcome_mix'].items()),
             'measured over 123 incidents, not the 90% assumed'),
            ('Flags', f'{d["accident_neutralizations"]:.2f} from crashes, '
                      f'{d["background_neutralizations"]:.2f} background',
             'one manager - the background rate is scaled so neither is '
             'counted twice'),
            ('Cause known for', f'{d["coverage"]:.0%} of real retirements',
             'the rest say only "Retired" and are not redistributed'),
        ]
        st.markdown(''.join(
            '<div class="paramrow"><div class="name">' + name + '</div>'
            '<div style="color:' + MUTED + ';font-size:0.78rem">' + note +
            '</div><div class="val">' + value + '</div></div>'
            for name, value, note in rows), unsafe_allow_html=True)

    st.markdown(
        '<div class="lede">The model produces ' +
        f'{d["p_accident"] + d["p_mechanical"]:.1%}' +
        ' of cars retiring. The results file says about 14% really do, and '
        'the difference is the third of retirements whose cause is not '
        'recorded. That gap is a limit on what was validated, not a third '
        'kind of failure.</div>', unsafe_allow_html=True)

    with st.expander('measured, or chosen'):
        as_badge = {'measured': 'measured', 'hand-set': 'hand-set',
                    'assumption': 'hand-set'}
        st.markdown(''.join(
            '<div class="paramrow"><div class="name">' + name + '</div>'
            '<div>' + badge(as_badge.get(kind, kind)) +
            ' <span style="color:' + MUTED + ';font-size:0.78rem">' + note +
            '</span></div><div class="val">' + value + '</div></div>'
            for name, kind, value, note in d['provenance']),
            unsafe_allow_html=True)


def show_comparison(runs):
    if len(runs) < 2:
        st.info('Two runs are needed. Change a flag in the sidebar and run '
                'again - both stay in memory until the tab is closed.')
        return

    labels = [r['label'] for r in runs]
    c1, c2 = st.columns(2)
    a = c1.selectbox('baseline', labels, index=0)
    b = c2.selectbox('against', labels, index=min(1, len(labels) - 1))
    if a == b:
        st.info('Pick two different runs.')
        return

    ra = next(r for r in runs if r['label'] == a)
    rb = next(r for r in runs if r['label'] == b)

    changed = {k: (ra['result']['flags'][k], rb['result']['flags'][k])
               for k in ra['result']['flags']
               if ra['result']['flags'][k] != rb['result']['flags'][k]}

    rule()
    eyebrow('what is different')
    if not changed:
        st.markdown(
            f'<div class="lede">No flag differs. Any change below is the seed '
            f'or the simulation count, not the model.</div>',
            unsafe_allow_html=True)
    else:
        rows = ''.join(
            f'<div class="paramrow"><div class="name">{flag_label(k)}</div>'
            f'<div style="color:{MUTED};font-size:0.80rem">'
            f'{"on" if v[0] else "off"} &rarr; '
            f'<b style="color:{ATTENTION}">{"on" if v[1] else "off"}</b></div>'
            f'<div class="val"></div></div>' for k, v in changed.items())
        st.markdown(rows, unsafe_allow_html=True)

    if ra['result']['seed'] != rb['result']['seed']:
        st.warning('The seeds differ, so part of what you see below is noise '
                   'rather than the flags.')

    # Two runs of different races are two different questions. Putting them
    # side by side is allowed, but the difference between them is not the
    # effect of anything.
    if ra['result']['event'] != rb['result']['event']:
        st.warning(f'These are different races - {ra["result"]["event"]} '
                   f'against {rb["result"]["event"]}. The columns can be read '
                   f'side by side, but the difference between them is not a '
                   f'comparison of settings.')

    rule()
    eyebrow('side by side')
    sa = ra['result']['summary'].set_index('Driver')
    sb = rb['result']['summary'].set_index('Driver')
    # Probability differences are in percentage points, not in percent: 20%
    # going to 25% is five points, and calling it "+25%" is a different and
    # wrong claim. The column says pp so the unit is never in doubt.
    joined = pd.DataFrame({
        'Team': sa['Team'],
        'Expected finish A': sa['mean_pos'],
        'Expected finish B': sb['mean_pos'].reindex(sa.index),
        'Finish diff': (sb['mean_pos'].reindex(sa.index) - sa['mean_pos']),
        'Win A': sa['P_win'],
        'Win B': sb['P_win'].reindex(sa.index),
        'Win diff (pp)': (sb['P_win'].reindex(sa.index) - sa['P_win']) * 100,
        'Podium A': sa['P_podium'],
        'Podium B': sb['P_podium'].reindex(sa.index),
        'Podium diff (pp)': (sb['P_podium'].reindex(sa.index)
                             - sa['P_podium']) * 100,
    }).reset_index()

    st.caption(f'A = {a}    ·    B = {b}')
    st.dataframe(
        joined.sort_values('Win diff (pp)', ascending=False),
        width='stretch', hide_index=True,
        column_config={
            'Expected finish A': st.column_config.NumberColumn(format='%.1f'),
            'Expected finish B': st.column_config.NumberColumn(format='%.1f'),
            'Finish diff': st.column_config.NumberColumn(
                format='%.2f', help='B minus A, in positions. Negative is a '
                                    'better expected finish under B.'),
            'Win A': st.column_config.NumberColumn(format='%.1f%%'),
            'Win B': st.column_config.NumberColumn(format='%.1f%%'),
            'Win diff (pp)': st.column_config.NumberColumn(
                format='%+.1f', help='Percentage points, not percent: 20% to '
                                     '25% is +5.0.'),
            'Podium A': st.column_config.NumberColumn(format='%.1f%%'),
            'Podium B': st.column_config.NumberColumn(format='%.1f%%'),
            'Podium diff (pp)': st.column_config.NumberColumn(format='%+.1f'),
        })

    biggest = joined.reindex(joined['Finish diff'].abs()
                             .sort_values(ascending=False).index).head(3)
    moved = ', '.join(f'{row["Driver"]} {row["Finish diff"]:+.2f}'
                      for _, row in biggest.iterrows())
    st.markdown(f'<div class="lede">Largest movement in expected finishing '
                f'position: {moved}.</div>', unsafe_allow_html=True)


def decision_chart(summary):
    """
    Where the stops actually went, against where the plan put them.

    Three bars, because there are three answers and the interesting one is the
    middle: a decision layer that confirms the plan most of the time is
    working, not idle. Colour carries the direction, the same way the grid
    chart does.
    """
    import plotly.graph_objects as go

    r, calls = summary['reasons'], summary['calls']
    parts = [
        ('earlier', r['tyre, earlier'] + r['traffic, earlier'], ATTENTION),
        ('on plan', r['on plan'], ACCENT),
        ('later', r['tyre, later'] + r['traffic, later'], ATTENTION),
    ]

    fig = go.Figure()
    for label, count, colour in parts:
        fig.add_bar(x=[count / calls], y=['calls'], orientation='h',
                    name=label, marker=dict(color=colour, opacity=
                                            1.0 if label == 'on plan' else 0.55),
                    hovertemplate=f'{label}: %{{x:.1%}}<extra></extra>')
    fig.update_layout(
        barmode='stack', height=86, margin=dict(l=0, r=0, t=4, b=18),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND, bargap=0.45,
        font=dict(color=INK, size=11),
        legend=dict(orientation='h', y=-0.55, x=0, font=dict(size=10)),
        xaxis=dict(tickformat='.0%', gridcolor=HAIRLINE, zeroline=False,
                   tickfont=dict(size=9, color=MUTED), range=[0, 1]),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


def show_strategy(result):
    """
    What the pit wall decided, and why.

    The counts are decisions, not stops: a car sits in its window for up to
    eleven laps and answers the question on every one of them. That is the
    right denominator for "how often did traffic change the answer" and the
    wrong one for "how many stops were made", so it is named on the page
    rather than left for the reader to assume.
    """
    v2 = result.get('v2')
    if v2 is None:
        st.markdown('<div class="lede">The v2.0 decision was off for this '
                    'run, so the pit lap came from the tyre curve alone. '
                    'Turn on REACTIVE_PIT_V2 and run again to compare.</div>',
                    unsafe_allow_html=True)
        return

    eyebrow('when to stop')
    st.markdown(
        '<div class="lede">Inside a window around the planned stop the car '
        'prices every pit lap still open to it - tyre, pit loss and up to four '
        'rivals - and takes the cheapest. The plan itself does not move: same '
        'number of stops, same compounds, only the lap.</div>',
        unsafe_allow_html=True)

    statline([
        ('decisions', f'{v2["calls"]:,}'),
        ('per driver', f'{v2["per_driver"]:.1f}'),
        ('mean shift', f'{v2["mean_shift"]:+.2f} laps'),
        ('traffic decided', f'{v2["traffic_share"]:.0%}'),
    ])
    st.plotly_chart(decision_chart(v2), width='stretch',
                    config={'displayModeBar': False})
    rule()

    r, calls = v2['reasons'], v2['calls']
    rows = [
        ('Confirmed the plan', 'the window opened and the plan still won',
         f'{r["on plan"] / calls:.0%}'),
        ('Tyre moved it', 'the curve alone chose a different lap',
         f'{(r["tyre, earlier"] + r["tyre, later"]) / calls:.0%}'),
        ('Traffic moved it', 'rivals changed the answer the tyre gave',
         f'{v2["traffic_share"]:.0%}'),
        ('Undercut attempted', 'stopped early with the car ahead still out',
         f'{v2["intents"]["undercut"] / calls:.0%}'),
        ('Overcut attempted', 'stayed out after a nearby car had stopped',
         f'{v2["intents"]["overcut"] / calls:.0%}'),
        ('Forced by the tyre cap', 'per driver, and not a v2.0 choice',
         f'{v2["forced_cap_stop"]:.3f}'),
        ('Free change, red flag', 'per driver, counted apart from normal stops',
         f'{v2["free_rf_change"]:.3f}'),
        ('No legal pit lap', 'times per race the plan ran out of candidates',
         f'{v2["infeasible"]:.2f}'),
    ]
    st.markdown(''.join(
        f'<div class="paramrow"><div class="name">{name}</div>'
        f'<div style="color:{MUTED};font-size:0.78rem">{note}</div>'
        f'<div class="val">{val}</div></div>' for name, note, val in rows),
        unsafe_allow_html=True)

    st.markdown('')
    st.markdown(
        f'<div class="lede">Undercut and overcut are what the car was '
        f'trying, read off what it could see at the time. Neither says the '
        f'move worked - that needs the rival\'s stop to have happened, and it '
        f'had not.</div>', unsafe_allow_html=True)

    log = v2.get('log') or []
    if log:
        rule()
        eyebrow(f'one race, every call: simulation #{S.V2_LOG_SIM}')
        st.markdown('<div class="lede">Ten thousand races of decision '
                    'arithmetic is tens of megabytes nobody reads. One race '
                    'is a thing you can check the model against.</div>',
                    unsafe_allow_html=True)

        table = pd.DataFrame([{
            'lap': e['lap'], 'driver': e['driver'], 'planned': e['planned'],
            'chose': e['chosen'], 'stopped': 'yes' if e['pit_now'] else '',
            'why': e['reason'], 'intent': e['intent'],
            'rivals': ', '.join(f'{n} ({role})' for n, role in e['rivals'])
                      or 'clear road',
        } for e in log])
        st.dataframe(table, width='stretch', hide_index=True, height=340)

        picked = st.selectbox(
            'the arithmetic behind one call', range(len(log)),
            format_func=lambda i: f'lap {log[i]["lap"]} - {log[i]["driver"]} '
                                  f'- {log[i]["reason"]}')
        entry = log[picked]
        detail = pd.DataFrame(entry['candidates'],
                              columns=['pit lap', 'own time', 'traffic',
                                       'total'])
        detail['chosen'] = np.where(detail['pit lap'] == entry['chosen'],
                                    '<-', '')
        st.dataframe(detail, width='stretch', hide_index=True)
        st.markdown(
            f'<div class="lede">Own time is this car\'s seconds from lap '
            f'{entry["lap"]} to the lap before its next planned stop, tyre and '
            f'pit loss together. Every candidate covers exactly those laps, '
            f'which is what makes the column comparable down the page.</div>',
            unsafe_allow_html=True)


def weather_chart(summary):
    """
    The track through the race: where the water came, where it went, and the
    two levels at which one tyre stops being the right one.
    """
    import plotly.graph_objects as go

    track = summary['track']
    laps = list(range(1, len(track) + 1))

    fig = go.Figure()
    fig.add_scatter(x=laps, y=track, mode='lines',
                    line=dict(color=ACCENT, width=2), fill='tozeroy',
                    fillcolor='rgba(47,93,138,0.10)',
                    hovertemplate='lap %{x}: %{y:.2f}<extra></extra>')

    for level, label in ((0.30, 'slicks give way'), (0.70, 'wets take over')):
        fig.add_hline(y=level, line=dict(color=HAIRLINE, width=1, dash='dot'),
                      annotation=dict(text=label, x=1, xanchor='right',
                                      font=dict(size=9, color=MUTED)))

    fig.update_layout(
        height=210, margin=dict(l=0, r=0, t=6, b=28),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND, showlegend=False,
        font=dict(color=INK, size=11),
        xaxis=dict(title=dict(text='lap', font=dict(size=10, color=MUTED)),
                   gridcolor=HAIRLINE, tickfont=dict(size=9, color=MUTED)),
        yaxis=dict(title=dict(text='wetness index',
                              font=dict(size=10, color=MUTED)),
                   range=[0, 1.02], gridcolor=HAIRLINE,
                   tickfont=dict(size=9, color=MUTED)))
    return fig


def show_weather(result):
    """
    What the rain did, what it cost, and which half of this model was ever
    measured.

    The provenance table is the point of the tab. Half of v2.1 rests on
    thousands of laps and half on a scenario pinned to two thin anchors, and a
    reader who cannot tell them apart will believe the wrong half.
    """
    w = result.get('weather')
    if w is None:
        st.markdown('<div class="lede">This run stayed dry. Turn the weather '
                    'on in the sidebar and pick a scenario to see a '
                    'changeable race.</div>', unsafe_allow_html=True)
        return

    eyebrow('scenario: ' + w['scenario'])
    st.markdown('<div class="lede">' + w['description'] + ' <b>This is a '
                'scenario, not a forecast.</b> Nobody has measured how likely '
                'it is, so every number here is conditional on the rain doing '
                'exactly this.</div>', unsafe_allow_html=True)

    statline([
        ('laps wet', f'{w["wet_lap_share"]:.0%}'),
        ('peak wetness', f'{w["peak_wetness"]:.2f}'),
        ('weather stops', f'{w["weather_stops"]:.2f}'),
        ('laps on wets', f'{w["laps_on_wet"]:.0f}'),
    ])
    st.plotly_chart(weather_chart(w), width='stretch',
                    config={'displayModeBar': False})
    rule()

    eyebrow('what the pit wall did')
    reasons, calls = w['reasons'], max(w['calls'], 1)
    rows = [
        ('Changed category', 'the conditions paid for the stop',
         f'{reasons["weather_switch"] / calls:.0%}'),
        ('Stayed out', 'the advantage did not cover the pit loss',
         f'{reasons["stay"] / calls:.0%}'),
        ('Waited a lap', 'close enough to look again',
         f'{reasons["wait one lap"] / calls:.0%}'),
        ('Forced off an unraceable tyre', 'not a strategy call at all',
         f'{reasons["current tyre unraceable"] / calls:.0%}'),
        ('Pit-lane stops', 'per driver, all causes', f'{w["pit_stops"]:.2f}'),
        ('of which for weather', 'per driver', f'{w["weather_stops"]:.2f}'),
        ('Free changes, red flag', 'per driver, not counted as pit stops',
         f'{w["rf_changes"]:.2f}'),
        ('Forced by a tyre cap', 'per driver, not a v2.1 choice',
         f'{w["forced_cap"]:.2f}'),
        ('5/10/15-lap windows agree', 'how much the answer depends on where '
         'the window was drawn', f'{w["horizon_agreement"]:.0%}'),
    ]
    st.markdown(''.join(
        '<div class="paramrow"><div class="name">' + name + '</div>'
        '<div style="color:' + MUTED + ';font-size:0.78rem">' + note + '</div>'
        '<div class="val">' + val + '</div></div>'
        for name, note, val in rows), unsafe_allow_html=True)

    if w['horizon_agreement'] < 0.80:
        st.markdown(
            '<div class="lede" style="color:' + ATTENTION + '">The three '
            'horizons disagree on ' + f'{1 - w["horizon_agreement"]:.0%}' +
            ' of calls, so on those the window is making the decision rather '
            'than the weather. Ten laps is a starting preference, not a '
            'measured optimum.</div>', unsafe_allow_html=True)

    rule()
    eyebrow('measured, or chosen')
    st.markdown('<div class="lede">The wet has been filtered out of every '
                'measurement in this project until now, so half of this rests '
                'on thousands of laps and half on almost nothing. Which is '
                'which:</div>', unsafe_allow_html=True)

    as_badge = {'measured': 'measured', 'thin': 'shrunk',
                'scenario': 'hand-set', 'assumption': 'hand-set'}
    st.markdown(''.join(
        '<div class="paramrow"><div class="name">' + name + '</div>'
        '<div>' + badge(as_badge.get(kind, kind)) +
        ' <span style="color:' + MUTED + ';font-size:0.78rem">' + note +
        '</span></div><div class="val">' + value + '</div></div>'
        for name, kind, value, note in w['provenance']),
        unsafe_allow_html=True)

    log = w.get('log') or []
    if log:
        rule()
        eyebrow('one race, every change: simulation #'
                + str(S.WEATHER_LOG_SIM))
        st.dataframe(pd.DataFrame(log), width='stretch', hide_index=True,
                     height=300)


def show_run(result):
    """
    What this result is, so two of them can be compared without guessing.

    It lives in memory beside the result and is written nowhere. The data
    column is the content digest of each file the prediction read, which is
    what says whether two runs saw the same inputs - a filename and a date
    cannot.
    """
    run = result.get('run')
    if run is None:
        return

    rule()
    eyebrow('this run')
    status = run['status']
    if run['completed_runs'] != run['requested_runs']:
        status += (f' - {run["completed_runs"]:,} of '
                   f'{run["requested_runs"]:,} finished, and the '
                   f'probabilities are out of the completed ones')
    statline([
        ('simulations', f'{run["completed_runs"]:,}'),
        ('seed', f'{run["seed"]}'),
        ('parameter set', run['parameter_set']),
        ('elapsed', f'{run["seconds"]:.1f} s'),
    ])

    rows = [
        ('Status', status, ''),
        ('Race', run['race'], f'data on disk: {run["cutoff"]["data_belongs_to"]}'),
        ('Historical replay', 'not available',
         'one copy of each file is kept and overwritten, so an earlier '
         'state of the data cannot be reconstructed'),
        ('Versions',
         ' '.join(f'{k} {v}' for k, v in run['versions'].items()), ''),
        ('Data files read', f'{len(run["data"])} tracked',
         'compared by content, not by date'),
    ]
    if run.get('stale_inputs'):
        rows.append(('Stale inputs', ', '.join(run['stale_inputs']),
                     'the pipeline would rebuild these'))

    st.markdown(''.join(
        '<div class="paramrow"><div class="name">' + name + '</div>'
        '<div style="color:' + MUTED + ';font-size:0.78rem">' + note + '</div>'
        '<div class="val" style="text-align:left">' + str(value) + '</div>'
        '</div>' for name, value, note in rows), unsafe_allow_html=True)


def show_diagnostics():
    """
    The per-lap report, which this interface cannot build.

    Diagnostics are written by the console run, not by the app, so the honest
    state here is "none for this run" rather than an instruction to go and
    open a terminal. What is on disk belongs to some other run and is offered
    as that, clearly labelled - the failure worth preventing is a reader
    taking a report of one race for a report of the one on screen.
    """
    path = os.path.join(OUT_DIR, 'diagnostics.html')
    if not os.path.exists(path):
        st.info('No diagnostics available for this run.')
        st.markdown(
            '<div class="lede">The per-lap report is produced by the console '
            'run rather than by this page, and none has been written '
            'yet.</div>', unsafe_allow_html=True)
        with st.expander('Building one (developers)'):
            st.caption('`python -m Simülasyon.simulate` writes '
                       'output/diagnostics.html. This page shows it unchanged '
                       'and never rebuilds it.')
        return

    stamp = pd.Timestamp(os.path.getmtime(path), unit='s')
    st.info('No diagnostics available for this run.')
    rule()
    eyebrow('report from an earlier console run')
    st.markdown(
        f'<div class="lede">Built {stamp:%Y-%m-%d %H:%M}. This is a different '
        f'run from the one shown on the Result tab - it was not produced by '
        f'this session, and its laps, retirements and safety cars are its '
        f'own.</div>', unsafe_allow_html=True)
    with open(path, encoding='utf-8') as f:
        html = f.read()
    components.html(html, height=900, scrolling=True)
    with st.expander('Where this came from'):
        st.caption(f'{path}, written by `python -m Simülasyon.simulate`.')


# --- main -------------------------------------------------------------------


def main():
    st.set_page_config(page_title='F1 Race Model', layout='wide',
                       initial_sidebar_state='expanded')
    st.markdown(CSS, unsafe_allow_html=True)

    if 'runs' not in st.session_state:
        st.session_state.runs = []

    # ---- sidebar ----------------------------------------------------------
    with st.sidebar:
        eyebrow('race')
        ok, on_disk = target_matches()
        st.markdown(f'**{S.TARGET_EVENT}** &nbsp; {S.SEASON}',
                    unsafe_allow_html=True)
        if ok is False:
            st.error(f'The data on disk was built for {on_disk}. Change '
                     f'TARGET_RACE in target_race.py, then re-run fetch.py, '
                     f'clean.py and team_affinity.py before trusting this.')
        elif ok:
            st.markdown(f'<div class="lede">grid, pace and affinity on disk '
                        f'all belong to this race</div>',
                        unsafe_allow_html=True)

        st.markdown('')
        n_sims = st.select_slider('Simulations',
                                  options=[1000, 2000, 5000, 10000, 20000],
                                  value=10000,
                                  help='More runs narrow the sampling noise. '
                                       'They do not make the model more '
                                       'correct.')

        # Run sits with the two controls almost everyone touches, not below
        # eight groups of model flags. Reaching it should not require reading
        # anything.
        go_now = st.button('Run simulation', type='primary', width='stretch')
        if st.session_state.runs:
            if st.button(f'Clear {len(st.session_state.runs)} run(s)',
                         width='stretch'):
                st.session_state.runs = []
                st.rerun()

        st.markdown('')
        eyebrow('advanced model settings')
        st.markdown('<div class="lede">Each switch selects between two '
                    'modelled behaviours. The constants around them are '
                    'calibrations and stay out of reach.</div>',
                    unsafe_allow_html=True)
        seed = st.number_input('Seed', value=int(S.RANDOM_SEED), step=1,
                               help='The same seed and the same settings give '
                                    'the same result, to the digit. Changing '
                                    'it resamples; it does not change the '
                                    'model.')

        groups = {
            'Tyres': ['ENFORCE_STINT_CAP', 'RECOST_STRATEGIES'],
            'Strategy': ['REACTIVE_PIT_V2', 'REACTIVE_PIT',
                         'PER_DRIVER_STRATEGY'],
            'Neutralisation': ['NEUTRAL_FREEZES_GAPS', 'SC_BUNCHING',
                               'RED_FLAG_ENABLED', 'NEUTRAL_LAST_LAP_KNOWN'],
            'Racing': ['START_MODEL', 'LOCK_START_ORDER', 'EVOLUTION_ENABLED',
                       'GAP_IN_PASS_MODEL', 'TEAM_PASS_ENABLED'],
            'Retirements': ['TWO_CAUSE_DNF', 'ACCIDENT_NEUTRALIZATION',
                            'DNF_ENABLED'],
            'Affinity': ['AFFINITY_ENABLED', 'AFFINITY_TEAM_2026'],
            'Weather': ['WEATHER_ENABLED', 'WEATHER_MAY_BREAK_PLAN'],
            'Conditions': ['IS_WET'],
        }
        # a flag the model offers and this panel forgot is a flag nobody can
        # reach, which is the same as not having it
        hidden = set(S.RUNTIME_FLAGS) - {n for ns in groups.values() for n in ns}
        if hidden:
            st.warning(f'not reachable from here: {", ".join(sorted(hidden))}')

        flags = {}
        for title, names in groups.items():
            with st.expander(title, expanded=False):
                for name in names:
                    flags[name] = st.checkbox(
                        flag_label(name), value=bool(getattr(S, name)),
                        help=flag_help(name), key=f'flag_{name}')

        scenario = S.WEATHER_SCENARIO
        if flags.get('WEATHER_ENABLED'):
            names = list(WX.SCENARIOS)
            scenario = st.selectbox(
                'Weather scenario', names,
                index=names.index(S.WEATHER_SCENARIO)
                if S.WEATHER_SCENARIO in names else 0,
                help='A scenario, not a forecast. Nothing here was measured '
                     'from a weather service and the probabilities of each '
                     'are unknown, so read a wet result as conditional on '
                     'the rain doing this.')
            st.markdown(f'<div class="lede">{WX.SCENARIOS[scenario]}</div>',
                        unsafe_allow_html=True)

    # ---- header -----------------------------------------------------------
    st.markdown(f'# {S.TARGET_EVENT} {S.SEASON}')
    st.markdown(
        '<div class="lede">A lap-by-lap Monte Carlo of the race. Every '
        'parameter below is labelled with where it came from.</div>',
        unsafe_allow_html=True)

    # ---- run --------------------------------------------------------------
    if go_now:
        # Everything the run needs is read once, here. Streamlit re-executes
        # the script from the top on every interaction, so the widget values
        # above are already a snapshot of the moment the button was pressed -
        # but the run is built from named locals rather than re-read from the
        # widgets, so that stays true if the layout ever changes.
        settings = {'n_sims': int(n_sims), 'seed': int(seed),
                    'flags': dict(flags),
                    'scenario': scenario if flags.get('WEATHER_ENABLED')
                    else None}

        stage = st.empty()
        bar = st.progress(0.0, text='checking data')

        # Stage one: is the data on disk the right data. This is the failure
        # that used to produce a confident answer built on another circuit's
        # grid, and it costs a fraction of a second to rule out.
        stage.markdown('<div class="lede">checking data</div>',
                       unsafe_allow_html=True)
        try:
            entries, manifest = PIPE.scan()
            blocked = PIPE.blocking(entries)
        except Exception as exc:                       # noqa: BLE001
            entries, blocked = [], []
            st.warning(f'the data check could not run: {exc}')

        if blocked:
            bar.empty()
            stage.empty()
            st.error('Required data is not ready: '
                     + ', '.join(f'{e["artefact"]} ({e["state"]})'
                                 for e in blocked)
                     + '. Run `python -m Simülasyon.pipeline --update` '
                       'before predicting.')
            st.stop()

        stale = [e for e in entries if e['state'] == PIPE.STALE]
        if stale:
            st.warning('Built from data the pipeline calls stale: '
                       + ', '.join(e['artefact'] for e in stale)
                       + '. The result is still shown, labelled as this.')

        overrides = dict(settings['flags'])
        if settings['scenario']:
            overrides['WEATHER_SCENARIO'] = settings['scenario']

        started = time.perf_counter()
        stage.markdown('<div class="lede">simulating</div>',
                       unsafe_allow_html=True)
        try:
            # N_SIMS and seed are not flags, so they move the same way by
            # hand and go back the same way in the finally below
            old_n, old_seed = S.N_SIMS, S.RANDOM_SEED
            S.N_SIMS, S.RANDOM_SEED = settings['n_sims'], settings['seed']
            try:
                result = S.run(
                    overrides=overrides,
                    progress_callback=lambda i, n: bar.progress(
                        min((i + 1) / n, 1.0),
                        text=f'lap {i + 1} of {n}  -  '
                             f'{settings["n_sims"]:,} races'))
            finally:
                S.N_SIMS, S.RANDOM_SEED = old_n, old_seed
        except Exception as exc:                       # surface, do not hide
            bar.empty()
            stage.empty()
            st.error(f'{type(exc).__name__}: {exc}')
            st.stop()

        stage.markdown('<div class="lede">collecting results</div>',
                       unsafe_allow_html=True)
        elapsed = time.perf_counter() - started
        try:
            result['run'] = PIPE.run_summary(
                S, result, elapsed, settings['n_sims'], result['n_sims'],
                'complete')
            result['run']['stale_inputs'] = [e['artefact'] for e in stale]
        except Exception:                              # noqa: BLE001
            result['run'] = None
        bar.empty()
        stage.empty()

        changed = [flag_label(k) for k, v in settings['flags'].items()
                   if v != getattr(S, k)]
        label = (f'#{len(st.session_state.runs) + 1} '
                 + (', '.join(changed) if changed else 'defaults')
                 + f' | {settings["n_sims"]:,} | seed {settings["seed"]}')
        # The settings are stored beside the result, not only rendered into a
        # label: that is what lets the page notice later that the sidebar has
        # moved on and the result on screen is answering the old question.
        st.session_state.runs.append({
            'label': label[:80], 'result': result,
            'settings': settings_signature(
                settings['n_sims'], settings['seed'], settings['flags'],
                settings['scenario']),
            'run_summary': result.get('run'),
            'source': 'run in this session',
            'changed': changed})
        st.session_state.runs = st.session_state.runs[-MAX_RUNS:]

    live = settings_signature(n_sims, seed, flags,
                              scenario if flags.get('WEATHER_ENABLED')
                              else None)

    runs = st.session_state.runs
    if not runs:
        st.markdown('')
        st.info('Press **Run simulation** on the left. Ten thousand races are '
                'simulated lap by lap and the finishing order is read off the '
                'distribution. Nothing is written to disk, and runs are kept '
                'only while this tab is open.')

        # A result from a console run is still a result, but it is not this
        # session's and must not be able to pass for it.
        csv = os.path.join(OUT_DIR, 'predictions.csv')
        if os.path.exists(csv):
            rule()
            eyebrow('loaded result - not generated in this session')
            st.markdown(
                '<div class="lede">The prediction committed with the '
                'repository, shown so the page is not empty before the first '
                'run. Press Run to replace it with one of your own.</div>',
                unsafe_allow_html=True)
            st.dataframe(pd.read_csv(csv), width='stretch', hide_index=True)
            with st.expander('Where this came from'):
                st.caption(f'{csv}, written by '
                           f'`python -m Simülasyon.simulate`.')
        return

    current = runs[-1]['result']
    drift = run_header(runs[-1], live)

    tabs = st.tabs(['Result', 'Strategy', 'Weather', 'Parameters', 'Compare',
                    'Diagnostics'])
    with tabs[0]:
        show_results(current)
    with tabs[1]:
        show_strategy(current)
    with tabs[2]:
        show_weather(current)
    with tabs[3]:
        show_parameters(current)
        show_run(current)
    with tabs[4]:
        show_comparison(runs)
    with tabs[5]:
        show_diagnostics()


if __name__ == '__main__':
    main()
