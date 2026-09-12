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

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Simülasyon import simulate as S
from Simülasyon import target_race as TR

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
        xaxis=dict(title=dict(text='positions gained against the grid',
                              font=dict(size=10, color=MUTED)),
                   zeroline=True, zerolinecolor=HAIRLINE, zerolinewidth=1,
                   gridcolor=HAIRLINE, tickfont=dict(size=9, color=MUTED)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
    )
    return fig


def share_card(result):
    """K4: one figure, vertical, readable on a phone. Plotly's own menu saves it."""
    import plotly.graph_objects as go

    d = result['summary'].head(10).copy()
    d = d.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(x=d['P_win'], y=d['Driver'], orientation='h', name='win',
                marker_color=ACCENT,
                hovertemplate='%{y} wins %{x:.1%}<extra></extra>')
    fig.add_bar(x=d['P_podium'] - d['P_win'], y=d['Driver'], orientation='h',
                name='podium', marker_color='#b9c8d8',
                hovertemplate='%{y} podium<extra></extra>')
    fig.update_layout(
        barmode='stack', height=460, width=520,
        margin=dict(l=0, r=12, t=58, b=34),
        paper_bgcolor=GROUND, plot_bgcolor=GROUND,
        title=dict(text=f'{result["event"]} {result["season"]}<br>'
                        f'<span style="font-size:11px;color:{MUTED}">'
                        f'{result["n_sims"]:,} simulations</span>',
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
        '<div class="lede">A measured value and a hand-set one are not the '
        'same evidence. The two that want attention are marked in colour; '
        'the rest separate by type.</div>', unsafe_allow_html=True)

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

    st.markdown('')
    with st.expander('every setting this run used'):
        st.code('\n'.join(result['param_lines']), language=None)


def show_results(result):
    summary = result['summary']
    pace = result['pace']

    winner = float(result['total'].min(axis=1).mean())
    grid_arr = pace['grid'].to_numpy()
    shift = float(np.abs(result['positions'] - grid_arr[None, :]).mean())

    statline([
        ('laps', f'{result["n_laps"]}'),
        ('simulations', f'{result["n_sims"]:,}'),
        ('pole', f'{S.POLE_DRIVER} {S.POLE_TIME:.3f}s'),
        ('mean winning time', f'{winner / 60:.1f} min'),
        ('mean shift from grid', f'{shift:.2f} pos'),
    ])
    rule()

    left, right = st.columns([1.02, 1])

    with left:
        eyebrow('finishing probability')
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
            '<div class="lede">Mean finish against starting slot. This is the '
            'only thing the model can be scored on before the race.</div>',
            unsafe_allow_html=True)
        st.plotly_chart(grid_delta_chart(summary, pace),
                        width='stretch',
                        config={'displaylogo': False})

    rule()
    eyebrow('classification')
    # summarize() already carries grid, so this only picks an order
    show = summary.copy()
    if 'grid' not in show.columns:
        show['grid'] = show['Driver'].map(pace.set_index('Driver')['grid'])
    cols = ['Driver', 'Team', 'grid', 'mean_pos', 'P_win', 'P_podium',
            'P_points', 'P_dnf', 'stops', 'affinity', 'strategy']
    cols = [c for c in cols if c in show.columns]
    st.dataframe(
        show[cols], width='stretch', hide_index=True,
        column_config={
            'mean_pos': st.column_config.NumberColumn('mean pos', format='%.2f'),
            'P_win': st.column_config.NumberColumn('win', format='%.3f'),
            'P_podium': st.column_config.NumberColumn('podium', format='%.3f'),
            'P_points': st.column_config.NumberColumn('points', format='%.3f'),
            'P_dnf': st.column_config.NumberColumn('dnf', format='%.3f'),
            'stops': st.column_config.NumberColumn('stops', format='%.2f'),
        })

    rule()
    eyebrow('shareable')
    st.markdown(
        '<div class="lede">No parameters, no diagnostics. Use the camera icon '
        'on the chart to save it.</div>', unsafe_allow_html=True)
    st.plotly_chart(share_card(result), width='content',
                    config={'displaylogo': False,
                            'toImageButtonOptions': {'scale': 2}})


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
            f'<div class="paramrow"><div class="name">{k}</div>'
            f'<div style="color:{MUTED};font-size:0.80rem">'
            f'{"on" if v[0] else "off"} &rarr; '
            f'<b style="color:{ATTENTION}">{"on" if v[1] else "off"}</b></div>'
            f'<div class="val"></div></div>' for k, v in changed.items())
        st.markdown(rows, unsafe_allow_html=True)

    if ra['result']['seed'] != rb['result']['seed']:
        st.warning('The seeds differ, so part of what you see below is noise '
                   'rather than the flags.')

    rule()
    eyebrow('side by side')
    sa = ra['result']['summary'].set_index('Driver')
    sb = rb['result']['summary'].set_index('Driver')
    joined = pd.DataFrame({
        'Team': sa['Team'],
        f'mean_pos {a}': sa['mean_pos'],
        f'mean_pos {b}': sb['mean_pos'].reindex(sa.index),
        'mean_pos diff': (sb['mean_pos'].reindex(sa.index) - sa['mean_pos']),
        f'P_win {a}': sa['P_win'],
        f'P_win {b}': sb['P_win'].reindex(sa.index),
        'P_win diff': (sb['P_win'].reindex(sa.index) - sa['P_win']),
    }).reset_index()

    st.dataframe(
        joined.sort_values('P_win diff', ascending=False),
        width='stretch', hide_index=True,
        column_config={c: st.column_config.NumberColumn(
            c, format='%.3f') for c in joined.columns if c != 'Driver'
            and c != 'Team'})

    biggest = joined.reindex(joined['mean_pos diff'].abs()
                             .sort_values(ascending=False).index).head(3)
    moved = ', '.join(f'{row["Driver"]} {row["mean_pos diff"]:+.2f}'
                      for _, row in biggest.iterrows())
    st.markdown(f'<div class="lede">Largest movement in mean finishing '
                f'position: {moved}.</div>', unsafe_allow_html=True)


def show_diagnostics():
    path = os.path.join(OUT_DIR, 'diagnostics.html')
    if not os.path.exists(path):
        st.info('Run `python -m Simülasyon.simulate` once to build '
                'output/diagnostics.html. The interface does not rebuild it, '
                'on purpose - it is the console run\'s artefact and is shown '
                'here unchanged.')
        return
    stamp = pd.Timestamp(os.path.getmtime(path), unit='s')
    st.markdown(f'<div class="lede">output/diagnostics.html, built '
                f'{stamp:%Y-%m-%d %H:%M}. Shown exactly as it is written.</div>',
                unsafe_allow_html=True)
    with open(path, encoding='utf-8') as f:
        html = f.read()
    components.html(html, height=900, scrolling=True)


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
        n_sims = st.select_slider('simulations',
                                  options=[1000, 2000, 5000, 10000, 20000],
                                  value=10000)
        seed = st.number_input('seed', value=int(S.RANDOM_SEED), step=1)

        st.markdown('')
        eyebrow('model')
        st.markdown('<div class="lede">Each switch selects between two '
                    'modelled behaviours. The constants around them are '
                    'calibrations and stay out of reach.</div>',
                    unsafe_allow_html=True)

        groups = {
            'Tyres': ['ENFORCE_STINT_CAP', 'RECOST_STRATEGIES'],
            'Strategy': ['REACTIVE_PIT', 'PER_DRIVER_STRATEGY'],
            'Neutralisation': ['NEUTRAL_FREEZES_GAPS', 'SC_BUNCHING',
                               'RED_FLAG_ENABLED'],
            'Racing': ['LOCK_START_ORDER', 'DNF_ENABLED', 'EVOLUTION_ENABLED',
                       'GAP_IN_PASS_MODEL', 'TEAM_PASS_ENABLED'],
            'Affinity': ['AFFINITY_ENABLED', 'AFFINITY_TEAM_2026'],
            'Conditions': ['IS_WET'],
        }
        flags = {}
        for title, names in groups.items():
            with st.expander(title, expanded=title in ('Racing', 'Tyres')):
                for name in names:
                    flags[name] = st.checkbox(
                        name, value=bool(getattr(S, name)),
                        help=S.RUNTIME_FLAGS[name], key=f'flag_{name}')

        st.markdown('')
        go_now = st.button('Run', type='primary', width='stretch')
        if st.session_state.runs:
            if st.button(f'Clear {len(st.session_state.runs)} run(s)',
                         width='stretch'):
                st.session_state.runs = []
                st.rerun()

    # ---- header -----------------------------------------------------------
    st.markdown(f'# {S.TARGET_EVENT} {S.SEASON}')
    st.markdown(
        '<div class="lede">A lap-by-lap Monte Carlo of the race. Every '
        'parameter below is labelled with where it came from.</div>',
        unsafe_allow_html=True)

    # ---- run --------------------------------------------------------------
    if go_now:
        bar = st.progress(0.0, text='building the field')
        overrides = dict(flags)
        try:
            # N_SIMS and seed are not flags, so they move the same way by
            # hand and go back the same way in the finally below
            old_n, old_seed = S.N_SIMS, S.RANDOM_SEED
            S.N_SIMS, S.RANDOM_SEED = int(n_sims), int(seed)
            try:
                result = S.run(
                    overrides=overrides,
                    progress_callback=lambda i, n: bar.progress(
                        min((i + 1) / n, 1.0), text=f'lap {i + 1} of {n}'))
            finally:
                S.N_SIMS, S.RANDOM_SEED = old_n, old_seed
        except Exception as exc:                       # surface, do not hide
            bar.empty()
            st.error(f'{type(exc).__name__}: {exc}')
            st.stop()
        bar.empty()

        changed = [k for k, v in flags.items() if v != getattr(S, k)]
        label = (f'#{len(st.session_state.runs) + 1} '
                 + (', '.join(changed) if changed else 'defaults')
                 + f' | {int(n_sims):,} | seed {int(seed)}')
        st.session_state.runs.append({'label': label[:80], 'result': result})
        st.session_state.runs = st.session_state.runs[-MAX_RUNS:]

    runs = st.session_state.runs
    if not runs:
        st.markdown('')
        st.info('Set the flags on the left and press Run. Nothing is written '
                'to disk and runs are kept only while this tab is open.')
        rule()
        eyebrow('last console run')
        csv = os.path.join(OUT_DIR, 'predictions.csv')
        if os.path.exists(csv):
            st.markdown('<div class="lede">output/predictions.csv, from the '
                        'last `python -m Simülasyon.simulate`.</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.read_csv(csv), width='stretch',
                         hide_index=True)
        return

    current = runs[-1]['result']
    tabs = st.tabs(['Result', 'Parameters', 'Compare', 'Diagnostics'])
    with tabs[0]:
        st.markdown(f'<div class="lede">{runs[-1]["label"]}</div>',
                    unsafe_allow_html=True)
        show_results(current)
    with tabs[1]:
        show_parameters(current)
    with tabs[2]:
        show_comparison(runs)
    with tabs[3]:
        show_diagnostics()


if __name__ == '__main__':
    main()
