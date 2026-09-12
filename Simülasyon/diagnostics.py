"""
F1 Prediction Simulation - diagnostics.py
Interactive diagnostic report, written as a single self-contained HTML file.

Four panels, each aimed at a failure this project has already hit once.

Grid to finish
    A line per driver from their grid slot to their mean finishing position.
    All stubs means the model is reproducing the grid, which is the naive
    benchmark wearing a Monte Carlo costume. One line crossing fifteen places
    means the overtaking model is broken, which is how the 23rd-to-7th problem
    surfaced.

Lap by lap
    Running order through the most representative race, safety car periods
    shaded. Lines that never cross mean nobody can pass. That panel is what
    exposed the frozen field caused by the attack range equalling the hold
    distance.

Tyre strategy
    A bar per driver split into stints, coloured by compound, with the pit laps
    marked. Shows what the field actually did rather than what was offered, so
    a strategy set that collapses onto one option is obvious at a glance.

Parameters
    Every number the run used. Without it there is no telling, a week later,
    which settings produced a given picture.

Plotly rather than matplotlib: the panels carry per-driver detail that only
works on hover, and stacked matplotlib titles kept colliding.
"""

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- styling ----------------------------------------------------------------

COMPOUND_COLORS = {
    'SOFT': '#e53935',
    'MEDIUM': '#fdd835',
    'HARD': '#eceff1',
    'INTERMEDIATE': '#43a047',
    'WET': '#1e88e5',
}
COMPOUND_TEXT = {'SOFT': 'white', 'MEDIUM': '#333', 'HARD': '#333',
                 'INTERMEDIATE': 'white', 'WET': 'white'}

DRIVER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
    '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31',
]

GAIN, LOSS, FLAT = '#2e7d32', '#c62828', '#9e9e9e'


def _colors(n):
    return [DRIVER_PALETTE[i % len(DRIVER_PALETTE)] for i in range(n)]


# --- representative race ----------------------------------------------------


def find_representative(positions, diag, top_fraction=0.10):
    """
    Picks the single simulation that best stands for the whole set.

    The obvious approach - rank every race by the joint likelihood of its
    finishing order and take the best - quietly guarantees a dull race. A
    retirement drops a quick driver to the back, which is a low-probability
    result for that driver, so any race with one scores badly and never reaches
    the shortlist. Safety cars scramble the order and get filtered out the same
    way. The "most likely race" ends up being the one where nothing happened,
    which is the opposite of representative when retirements and safety cars
    are common.

    So the events are conditioned on rather than scored. First the modal number
    of retirements and safety cars is found - if most races have one stoppage,
    the chosen race has one. Only races matching that profile are considered.
    Likelihood then decides between them, computed over the drivers who
    actually finished so that the retirement itself is not penalised, and a
    median-distance tiebreak keeps the pace of the race typical too.
    """
    n_sims, n_drivers = positions.shape

    retired = diag.get('retired')
    neutral = diag.get('neutral')

    n_retired = (retired.sum(axis=1) if retired is not None
                 else np.zeros(n_sims, dtype=int))
    n_sc = ((neutral == 2).any(axis=1).astype(int) if neutral is not None
            else np.zeros(n_sims, dtype=int))

    # the profile most races actually have, not the tidiest one
    modal_retired = int(np.bincount(n_retired).argmax())
    modal_sc = int(np.bincount(n_sc).argmax())

    pool = np.flatnonzero((n_retired == modal_retired) & (n_sc == modal_sc))
    if len(pool) < 50:                      # too strict, drop the SC condition
        pool = np.flatnonzero(n_retired == modal_retired)
    if len(pool) < 50:
        pool = np.arange(n_sims)

    counts = np.zeros((n_drivers, n_drivers))
    for i in range(n_drivers):
        counts[i] = np.bincount(positions[:, i] - 1, minlength=n_drivers)
    probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + n_drivers)

    idx = positions - 1
    per_driver = np.log(probs[np.arange(n_drivers)[None, :], idx])
    if retired is not None:
        # judge the order of the cars still running, not the luck of the ones out
        per_driver = np.where(retired, 0.0, per_driver)
    loglik = per_driver.sum(axis=1)

    features = []
    if 'overtakes' in diag:
        features.append(np.asarray(diag['overtakes'], dtype=float))
    if 'n_stops' in diag:
        features.append(diag['n_stops'].mean(axis=1))
    if 'laps_stuck' in diag:
        features.append(diag['laps_stuck'].mean(axis=1))
    if neutral is not None:
        features.append((neutral > 0).sum(axis=1).astype(float))

    pool_loglik = loglik[pool]
    cutoff = np.quantile(pool_loglik, 1.0 - top_fraction)
    candidates = pool[pool_loglik >= cutoff]

    if features:
        stack = np.column_stack(features)
        med = np.median(stack[pool], axis=0)
        spread = np.std(stack[pool], axis=0)
        spread[spread == 0] = 1.0
        distance = np.abs((stack[candidates] - med) / spread).sum(axis=1)
        best = int(candidates[int(np.argmin(distance))])
    else:
        best = int(candidates[int(np.argmax(loglik[candidates]))])

    report = {
        'sim': best,
        'loglik': float(loglik[best]),
        'loglik_percentile': float((loglik[pool] < loglik[best]).mean()),
        'pool': len(pool),
        'candidates': len(candidates),
        'modal_retired': modal_retired,
        'modal_sc': modal_sc,
        'retired': int(n_retired[best]),
        'overtakes': int(diag['overtakes'][best]) if 'overtakes' in diag else None,
        'stops': float(diag['n_stops'][best].mean()) if 'n_stops' in diag else None,
        'neutral_laps': int((neutral[best] > 0).sum()) if neutral is not None else None,
        'had_sc': bool((neutral[best] == 2).any()) if neutral is not None else None,
    }
    return best, report


def build_stints(compound_by_lap, pits_by_lap, codes, retired_lap=None):
    """
    Turns per-lap compound history into stint blocks.

    retired_lap ends a driver's race early. Without it a car that parked on lap
    30 still showed a bar running to the flag, which read as a quiet run to the
    end rather than a retirement.
    """
    n_laps, n_drivers = compound_by_lap.shape
    stints = []

    for d in range(n_drivers):
        last = n_laps
        out_lap = None
        if retired_lap is not None and retired_lap[d] >= 0:
            out_lap = int(retired_lap[d]) + 1      # lap index to lap number
            last = min(last, out_lap)

        start = 0
        for lap in range(1, last + 1):
            ends = lap == last or compound_by_lap[lap, d] != compound_by_lap[start, d]
            if ends:
                stints.append({
                    'driver': codes[d],
                    'driver_idx': d,
                    'compound': compound_by_lap[start, d],
                    'start': start + 1,
                    'length': lap - start,
                    'end': lap,
                    'pit_lap': lap if lap < last else None,
                    'retired': out_lap is not None and lap == last,
                })
                start = lap
    return stints


# --- panels -----------------------------------------------------------------


def _add_grid_to_finish(fig, pace, positions, row, col):
    order = np.argsort(pace['grid'].to_numpy())
    codes = pace['Driver'].to_numpy()[order]
    grid = pace['grid'].to_numpy()[order].astype(float)
    finish = positions.mean(axis=0)[order]

    for code, g, f in zip(codes, grid, finish):
        gain = g - f
        colour = GAIN if gain > 0.5 else (LOSS if gain < -0.5 else FLAT)
        fig.add_trace(go.Scatter(
            x=[g, f], y=[code, code], mode='lines',
            line=dict(color=colour, width=3),
            hovertemplate=f'{code}<br>grid {g:.0f} to {f:.1f} ({gain:+.1f})<extra></extra>',
            showlegend=False,
        ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=grid, y=codes, mode='markers', name='grid',
        marker=dict(color='#37474f', size=8), showlegend=True,
        hovertemplate='grid %{x:.0f}<extra></extra>',
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=finish, y=codes, mode='markers', name='mean finish',
        marker=dict(color='#ff9800', size=8), showlegend=True,
        hovertemplate='finish %{x:.2f}<extra></extra>',
    ), row=row, col=col)

    # plotly thins category ticks by default and silently drops every other
    # driver, so the tick values are pinned explicitly
    fig.update_yaxes(autorange='reversed', tickmode='array',
                     tickvals=list(codes), ticktext=list(codes),
                     tickfont=dict(size=10), row=row, col=col)
    fig.update_xaxes(title_text='Position', row=row, col=col)


def neutralization_summary(neutral):
    """Plain description of when the race was neutralised, or that it was not."""
    spans = []
    for code, label in [(3, 'RED FLAG'), (2, 'SC'), (1, 'VSC')]:
        active = neutral == code
        start = None
        for i, on in enumerate(active):
            if on and start is None:
                start = i
            elif not on and start is not None:
                spans.append((start + 1, i, label))
                start = None
        if start is not None:
            spans.append((start + 1, len(active), label))

    if not spans:
        return 'green flag throughout, no SC or VSC'
    spans.sort()
    return '  '.join(f'{label} laps {a}-{b}' for a, b, label in spans)


def _add_lap_positions(fig, pace, trace, row, col, n_label=6):
    order_by_lap = trace['order']
    neutral = trace['neutral']
    retired_lap = trace.get('retired_lap')
    codes = pace['Driver'].to_numpy()
    n_laps, n_drivers = order_by_lap.shape

    pos = np.empty((n_laps, n_drivers), dtype=int)
    ranks = np.broadcast_to(np.arange(1, n_drivers + 1), order_by_lap.shape)
    np.put_along_axis(pos, order_by_lap, ranks, axis=1)

    laps = np.arange(1, n_laps + 1)
    colours = _colors(n_drivers)
    still_running = np.ones(n_drivers, dtype=bool)
    if retired_lap is not None:
        still_running = retired_lap < 0
    ranked = np.argsort(np.where(still_running, pos[-1], n_drivers + 1))
    front = ranked[:n_label]

    for code, colour in [(3, 'rgba(33,33,33,0.35)'),
                         (2, 'rgba(229,57,53,0.18)'),
                         (1, 'rgba(253,216,53,0.22)')]:
        active = neutral == code
        start = None
        for i, on in enumerate(active):
            if on and start is None:
                start = i
            elif not on and start is not None:
                fig.add_vrect(x0=start + 1, x1=i + 1, fillcolor=colour,
                              line_width=0, layer='below', row=row, col=col)
                start = None
        if start is not None:
            fig.add_vrect(x0=start + 1, x1=n_laps, fillcolor=colour,
                          line_width=0, layer='below', row=row, col=col)

    out_x, out_y, out_text = [], [], []

    for d in range(n_drivers):
        lead = d in front
        stop = n_laps
        if retired_lap is not None and retired_lap[d] >= 0:
            stop = int(retired_lap[d]) + 1
            out_x.append(stop)
            out_y.append(pos[stop - 1, d])
            out_text.append(f'{codes[d]} retired on lap {stop}')

        fig.add_trace(go.Scatter(
            x=laps[:stop], y=pos[:stop, d], mode='lines', name=codes[d],
            line=dict(color=colours[d] if lead else '#d0d0d0',
                      width=2.4 if lead else 1.0),
            opacity=1.0 if lead else 0.55,
            showlegend=False,
            hovertemplate=f'{codes[d]}<br>lap %{{x}}, P%{{y}}<extra></extra>',
        ), row=row, col=col)

    if out_x:
        fig.add_trace(go.Scatter(
            x=out_x, y=out_y, mode='markers', name='retirement',
            marker=dict(symbol='x', size=11, color='#212121',
                        line=dict(width=2)),
            hovertext=out_text, hoverinfo='text', showlegend=True,
        ), row=row, col=col)

    fig.update_yaxes(autorange='reversed', title_text='Position',
                     tickmode='linear', tick0=1, dtick=1,
                     tickfont=dict(size=9), row=row, col=col)
    fig.update_xaxes(title_text='Lap', row=row, col=col)

    # label the leaders on the right so the lines can be read without hovering
    for d in front:
        fig.add_annotation(x=n_laps, y=pos[-1, d], text=f' {codes[d]}',
                           showarrow=False, xanchor='left', yanchor='middle',
                           font=dict(size=10, color=colours[d]),
                           row=row, col=col)


def _add_tyre_strategy(fig, pace, stints, final_positions, row, col,
                       retired_lap=None):
    codes = pace['Driver'].to_numpy()
    order = np.argsort(final_positions)

    def label_for(d):
        if retired_lap is not None and retired_lap[d] >= 0:
            return f'DNF  {codes[d]}'
        return f'P{final_positions[d]:.0f}  {codes[d]}'

    labels = [label_for(d) for d in order]
    row_of = {d: labels[i] for i, d in enumerate(order)}

    seen = set()
    for st in stints:
        compound = st['compound']
        colour = COMPOUND_COLORS.get(compound, '#90a4ae')
        show = compound not in seen
        seen.add(compound)
        fig.add_trace(go.Bar(
            x=[st['length']], y=[row_of[st['driver_idx']]],
            base=[st['start'] - 1], orientation='h',
            marker=dict(color=colour, line=dict(color='white', width=1)),
            name=compound, legendgroup=compound, showlegend=show,
            text=[str(st['length'])], textposition='inside',
            insidetextfont=dict(color=COMPOUND_TEXT.get(compound, '#333'), size=9),
            hovertemplate=(f"{st['driver']}<br>{compound}<br>"
                           f"laps {st['start']}-{st['end']} "
                           f"({st['length']})<extra></extra>"),
        ), row=row, col=col)

    out_x, out_y, out_text = [], [], []
    for st in stints:
        if st.get('retired'):
            out_x.append(st['end'])
            out_y.append(row_of[st['driver_idx']])
            out_text.append(f"{st['driver']} retired on lap {st['end']}")
    if out_x:
        fig.add_trace(go.Scatter(
            x=out_x, y=out_y, mode='markers', name='retirement',
            marker=dict(symbol='x', size=11, color='#212121',
                        line=dict(width=2)),
            hovertext=out_text, hoverinfo='text', showlegend=False,
        ), row=row, col=col)

    pit_x, pit_y, pit_text = [], [], []
    for st in stints:
        if st['pit_lap'] is not None:
            pit_x.append(st['pit_lap'])
            pit_y.append(row_of[st['driver_idx']])
            pit_text.append(f"{st['driver']} pits lap {st['pit_lap']}")
    if pit_x:
        fig.add_trace(go.Scatter(
            x=pit_x, y=pit_y, mode='markers', name='pit stop',
            marker=dict(symbol='triangle-down', size=9, color='#212121'),
            hovertext=pit_text, hoverinfo='text', showlegend=True,
        ), row=row, col=col)

    fig.update_yaxes(autorange='reversed', categoryorder='array',
                     categoryarray=labels, tickmode='array',
                     tickvals=labels, ticktext=labels,
                     tickfont=dict(size=10), row=row, col=col)
    fig.update_xaxes(title_text='Lap', row=row, col=col)


def _add_parameters(fig, param_lines, row, col):
    pairs = []
    for line in param_lines:
        if not line.strip():
            continue
        parts = line.split(None, 1)
        pairs.append((parts[0], parts[1].strip() if len(parts) > 1 else ''))

    half = (len(pairs) + 1) // 2
    left, right = pairs[:half], pairs[half:]
    while len(right) < len(left):
        right.append(('', ''))

    fig.add_trace(go.Table(
        columnwidth=[70, 210, 70, 210],
        header=dict(values=['', '', '', ''], height=1,
                    fill_color='white', line_color='white'),
        cells=dict(
            values=[[p[0] for p in left], [p[1] for p in left],
                    [p[0] for p in right], [p[1] for p in right]],
            align='left', height=19,
            font=dict(family='monospace', size=11),
            fill_color=[['#f5f5f5'] * len(left), ['white'] * len(left),
                        ['#f5f5f5'] * len(right), ['white'] * len(right)],
            line_color='#e0e0e0',
        ),
    ), row=row, col=col)


# --- plain-text dump ---------------------------------------------------------
# The same three panels' underlying arrays, restated as monospace text at the
# bottom of the page. The charts are for a person; this is for reading the
# exact numbers back out - by Claude or anyone else - without decoding a plot.
# No formatting effort beyond fixed-width columns: ugly is fine here.


def _dump_grid_to_finish(pace, positions):
    order = np.argsort(pace['grid'].to_numpy())
    codes = pace['Driver'].to_numpy()[order]
    grid = pace['grid'].to_numpy()[order].astype(float)
    finish = positions.mean(axis=0)[order]

    lines = ['-- grid to finish (mean over all simulations) --',
            f'{"driver":8s}{"grid":>6s}{"mean_finish":>13s}{"gain":>8s}']
    for code, g, f in zip(codes, grid, finish):
        lines.append(f'{code:8s}{g:6.0f}{f:13.2f}{g - f:+8.2f}')
    return lines


def _dump_lap_positions(pace, trace):
    order_by_lap = trace['order']
    codes = pace['Driver'].to_numpy()
    n_laps, n_drivers = order_by_lap.shape

    pos = np.empty((n_laps, n_drivers), dtype=int)
    ranks = np.broadcast_to(np.arange(1, n_drivers + 1), order_by_lap.shape)
    np.put_along_axis(pos, order_by_lap, ranks, axis=1)

    retired_lap = trace.get('retired_lap')

    lines = ['-- lap-by-lap position, representative race '
            f'(driver order = starting grid) --',
            'lap  ' + ''.join(f'{c:>5s}' for c in codes)]
    for lap in range(n_laps):
        cells = []
        for d in range(n_drivers):
            out = retired_lap is not None and 0 <= retired_lap[d] < lap
            cells.append(f'{"--":>5s}' if out else f'{pos[lap, d]:>5d}')
        lines.append(f'{lap + 1:3d}  ' + ''.join(cells))
    return lines


def _dump_pit_and_neutral(pace, trace):
    codes = pace['Driver'].to_numpy()
    pits = trace['pits']
    neutral = trace['neutral']
    n_laps, n_drivers = pits.shape

    lines = ['-- pit stops, representative race --']
    any_pit = False
    for lap in range(n_laps):
        stopped = [codes[d] for d in range(n_drivers) if pits[lap, d]]
        if stopped:
            any_pit = True
            lines.append(f'lap {lap + 1:3d}: ' + ', '.join(stopped))
    if not any_pit:
        lines.append('(none)')

    lines.append('')
    lines.append('-- neutralization by lap (0 green, 1 VSC, 2 SC, 3 red flag) --')
    codes_line = ''.join(str(int(neutral[lap])) for lap in range(n_laps))
    lines.append(codes_line)

    retired_lap = trace.get('retired_lap')
    if retired_lap is not None and (retired_lap >= 0).any():
        lines.append('')
        lines.append('-- retirements --')
        for d in np.flatnonzero(retired_lap >= 0):
            lines.append(f'{codes[d]}: retired lap {int(retired_lap[d]) + 1}')
    return lines


def _dump_stints(stints, pace, final_positions):
    codes = pace['Driver'].to_numpy()
    by_driver = {}
    for st in stints:
        by_driver.setdefault(st['driver_idx'], []).append(st)

    lines = ['-- tyre strategy, representative race --',
            f'{"driver":8s}{"final":>6s}  stints (compound laps_start-laps_end)']
    for d in sorted(by_driver, key=lambda d: final_positions[d]):
        parts = ' -> '.join(
            f'{s["compound"]}({s["start"]}-{s["end"]})' for s in by_driver[d])
        lines.append(f'{codes[d]:8s}{final_positions[d]:6.0f}  {parts}')
    return lines


def build_text_dump(pace, positions, trace, stints, param_lines):
    """
    Everything the three chart panels are built from, as plain text.

    Deliberately not pretty: fixed-width columns, no colour, no interaction.
    The point is that every number the charts show can be read back exactly,
    including the full lap-by-lap position matrix, which a plot only shows on
    hover one point at a time.
    """
    final = np.empty(trace['order'].shape[1], dtype=int)
    final[trace['order'][-1]] = np.arange(1, trace['order'].shape[1] + 1)
    retired_lap = trace.get('retired_lap')
    if retired_lap is not None and (retired_lap >= 0).any():
        out = np.flatnonzero(retired_lap >= 0)
        running = np.flatnonzero(retired_lap < 0)
        final[running[np.argsort(final[running])]] = np.arange(1, len(running) + 1)
        for rank, d in enumerate(out[np.argsort(-retired_lap[out])], start=1):
            final[d] = len(running) + rank

    blocks = [
        ['=' * 78, 'RAW DATA BEHIND THE CHARTS ABOVE (unvisualised)', '=' * 78, ''],
        _dump_grid_to_finish(pace, positions), [''],
        _dump_lap_positions(pace, trace), [''],
        _dump_pit_and_neutral(pace, trace), [''],
        _dump_stints(stints, pace, final), [''],
        ['-- run parameters --'], list(param_lines),
    ]
    return '\n'.join(line for block in blocks for line in block)


# --- report -----------------------------------------------------------------


def build_report(pace, positions, trace, stints, param_lines, title,
                 subtitle, out_path, text_dump=True):
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy', 'colspan': 2}, None],
               [{'type': 'table', 'colspan': 2}, None]],
        row_heights=[0.30, 0.36, 0.34],
        vertical_spacing=0.085, horizontal_spacing=0.09,
        subplot_titles=('Grid to finish',
                        f'Most representative race - {subtitle}',
                        'Tyre strategy in that race',
                        ''),
    )

    retired_lap = trace.get('retired_lap')
    has_retirements = retired_lap is not None and bool((retired_lap >= 0).any())

    neutral_text = neutralization_summary(trace['neutral'])
    if has_retirements:
        codes = pace['Driver'].to_numpy()
        outs = ', '.join(f'{codes[d]} lap {int(retired_lap[d]) + 1}'
                         for d in np.argsort(retired_lap)[::-1]
                         if retired_lap[d] >= 0)
        neutral_text += f'  |  retired: {outs}'
    else:
        neutral_text += '  |  no retirements'

    order_by_lap = trace['order']
    n_drivers = order_by_lap.shape[1]
    final = np.empty(n_drivers, dtype=int)
    final[order_by_lap[-1]] = np.arange(1, n_drivers + 1)

    # a retired car is classified behind everyone who finished, and ahead of
    # anyone who stopped earlier than it did
    if has_retirements:
        out = np.flatnonzero(retired_lap >= 0)
        running = np.flatnonzero(retired_lap < 0)
        final[running[np.argsort(final[running])]] = np.arange(1, len(running) + 1)
        for rank, d in enumerate(out[np.argsort(-retired_lap[out])], start=1):
            final[d] = len(running) + rank

    _add_grid_to_finish(fig, pace, positions, 1, 1)
    _add_lap_positions(fig, pace, trace, 1, 2)
    _add_tyre_strategy(fig, pace, stints, final, 2, 1,
                       retired_lap=retired_lap)
    _add_parameters(fig, param_lines, 3, 1)

    fig.add_annotation(
        text=f'<b>{neutral_text}</b>', xref='paper', yref='paper',
        x=0.56, y=1.005, showarrow=False, xanchor='left', yanchor='bottom',
        font=dict(size=11, color='#c62828' if 'SC' in neutral_text else '#2e7d32'),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center',
                   font=dict(size=19)),
        height=1500, width=1560,
        barmode='stack', bargap=0.30,
        template='plotly_white',
        margin=dict(t=95, b=45, l=70, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.045,
                    xanchor='center', x=0.5),
        hovermode='closest',
        font=dict(size=12),
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13)
        annotation['xanchor'] = 'left'
        annotation['x'] = annotation['x'] - 0.04

    if text_dump:
        # appended below the chart in the same file, not a separate one - so
        # there is exactly one report to open, scroll, and read to the bottom
        dump = build_text_dump(pace, positions, trace, stints, param_lines)
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        pre = ('<pre style="font-size:11px; line-height:1.25; padding:16px; '
              'white-space:pre; overflow-x:auto;">'
              + dump.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
              + '</pre>')
        html = html.replace('</body>', pre + '</body>')
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write(html)
    else:
        fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'Diagnostics saved: {out_path}')

    png_path = str(out_path).replace('.html', '.png')
    try:
        fig.write_image(png_path, scale=2)
        print(f'Diagnostics saved: {png_path}')
    except Exception:
        pass                       # kaleido not installed, HTML is enough

    return fig