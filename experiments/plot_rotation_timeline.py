import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7,
                     "hatch.linewidth": 0.5, "pdf.fonttype": 42})
src, out, T0, T1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rows = [r for r in csv.DictReader(open(src))]
for r in rows:
    r['t'] = float(r['t_ns']); r['d'] = float(r['dur_ns']); r['v'] = int(r['value'])
BC = ['#4477AA', '#EE6677', '#228833', '#CCBB44']        # bank colours
lanes = ['C/A', 'Bank 0', 'Bank 1', 'Bank 2', 'Bank 3', 'BG line', 'I/O']
Y = {n: len(lanes) - 1 - i for i, n in enumerate(lanes)}
H = 0.62
fig, ax = plt.subplots(figsize=(3.45, 1.38))
def rect(x0, x1, lane, **kw):
    a, b = max(x0, T0), min(x1, T1)
    if b > a: ax.add_patch(Rectangle((a, Y[lane] - H / 2), b - a, H, **kw))
for k in range(4):
    lane = f'Bank {k}'
    ev = sorted([r for r in rows if r['unit'] == f'bg0.bank{k}'], key=lambda r: r['t'])
    def spans(p):
        s, cur = [], None
        for r in ev:
            if r['kind'] == p + '_ACT': cur = r['t']
            elif r['kind'] == p + '_PRE':
                s.append((cur if cur is not None else T0 - 100, r['t'])); cur = None
        if cur is not None: s.append((cur, T1 + 100))
        return s
    for a, b in spans('PIM'):
        rect(a, b, lane, facecolor=BC[k], alpha=0.30, lw=0)
    rd = [r['t'] for r in ev if r['kind'] == 'PIM_RD' and T0 <= r['t'] <= T1]
    ax.vlines(rd, Y[lane] - H / 2, Y[lane] + H / 2, color=BC[k], lw=0.35)
    for a, b in spans('HOST'):
        rect(a, b, lane, facecolor='white', edgecolor='black', lw=0.6, hatch='////')
    hr = [r['t'] for r in ev if r['kind'] == 'HOST_RD' and T0 <= r['t'] <= T1]
    ax.vlines(hr, Y[lane] - H / 2, Y[lane] + H / 2, color='black', lw=0.9)
for r in rows:
    if r['unit'] == 'bg0.bus' and T0 <= r['t'] < T1:
        rect(r['t'], r['t'] + r['d'], 'BG line', facecolor=BC[r['v']], lw=0)
    if r['unit'] == 'io' and T0 <= r['t'] < T1:
        rect(r['t'], r['t'] + r['d'], 'I/O',
             facecolor='#555555' if r['kind'] == 'HOST_DATA' else '#EE7733', lw=0)
    if r['unit'] == 'ca' and T0 <= r['t'] < T1:
        rect(r['t'], r['t'] + r['d'], 'C/A',
             facecolor='#888888' if r['kind'] == 'HOST_CMD' else '#AA3377', lw=0)
# annotate one bank cycle on bank 0
acts = [r['t'] for r in rows if r['unit'] == 'bg0.bank0' and r['kind'] == 'PIM_ACT' and T0 <= r['t'] <= T1]
ax.set_xlim(T0, T1); ax.set_ylim(-0.55, len(lanes) - 0.45)
ax.set_yticks([Y[n] for n in lanes]); ax.set_yticklabels(lanes)
ticks = list(range(int(T0), int(T1) + 1, 50))
ax.set_xticks(ticks); ax.set_xticklabels([str(int(t - T0)) for t in ticks])
ax.set_xlabel('Time (ns)', labelpad=1)
ax.tick_params(axis='both', length=2, pad=1)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
leg = [Patch(facecolor=BC[0], alpha=0.30, label='PIM row stream'),
       Patch(facecolor='white', edgecolor='black', hatch='////', lw=0.6, label='xPU row'),
       Line2D([0], [0], color='black', lw=0.9, label='xPU read'),
       Patch(facecolor='#555555', label='xPU data'),
       Patch(facecolor='#EE7733', label='PIM I/O'),
       Patch(facecolor='#AA3377', label='PIM queue cmd')]
ax.legend(handles=leg, ncol=3, loc='lower center', bbox_to_anchor=(0.46, 1.0),
          frameon=False, fontsize=6.3, handlelength=1.2, columnspacing=0.8,
          handletextpad=0.4, borderaxespad=0.1)
plt.tight_layout(pad=0.2)
plt.savefig(out, bbox_inches='tight', pad_inches=0.02)
plt.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
