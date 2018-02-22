import sys

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import colorcet as cc
mpl.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')


try:
    only_1_and_2 = sys.argv[2] != 'full'
except IndexError:
    only_1_and_2 = True
overlay = False
single = False

logs = torch.load(sys.argv[1])
target_var = sys.argv[1].split('/')[-1].split('-')[0]

if 'weights' in logs:
    single = True
    lookup_plins = range(8)
    plins = [[[logs['weights']['module.counter.f.{}.weight'.format(i)].cpu()]*200] for i in lookup_plins]
    only_1_and_2 = False
    configs = [0]
    other_configs = [0]
    try:
        logs2 = torch.load(sys.argv[3])
        plins2 = [x['plins'] for x in logs2]
        configs2 = [x['config']['noise'] for x in logs2]
        eps = 0.0001
        plins2, configs2 = zip(*[
            (p, c) for (p, c) in zip(plins2, configs2) \
#            if -eps < c - 0.5 < eps or -eps < c - 0.75 < eps
            if 0.35 - eps < c < 0.45 + eps or 0.65 - eps < c < 0.75 + eps or c > 0.9 - eps
        ])
        plins2 = list(zip(*plins2))

        plins = [x + list(y) for x, y in zip(plins, plins2)]
        configs = configs + list(configs2)
        overlay = True
    except IndexError:
        pass
else:
    plins = [x['plins'] for x in logs]
    configs = [x['config'][target_var] for x in logs]
    other_configs = [x['config']['noise' if target_var == 'coord' else 'coord'] for x in logs]
    plins = list(zip(*plins))

cm = cc.m_rainbow


axes = []
if not only_1_and_2:
    fig = plt.figure(figsize=(9, 4.8), dpi=200)
else:
    fig = plt.figure(figsize=(4.5, 2.1), dpi=200)

for plin_number, plin in enumerate(plins):
    plin = list(zip(*plin))
    cnorm = colors.Normalize(vmin=min(configs), vmax=max(configs) if not overlay else 1)
    scalar_map = cmx.ScalarMappable(norm=cnorm, cmap=cm)
    if only_1_and_2 and not plin_number < 2:
        break
    for i, (p, param) in enumerate(zip(plin[-1], configs)):
        if p.size(0) <= 16:
            p = torch.cat([torch.zeros(1), p], dim=0)
        p = p.abs()
        p = p.cumsum(dim=0) / p.sum()

        if not only_1_and_2:
            ax = plt.subplot(2, 4, plin_number + 1)
        else:
            ax = plt.subplot(1, 2, plin_number + 1)
        axes.append(ax)
        col = scalar_map.to_rgba(param) if not (single and i == 0) else 'k'
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        ax.yaxis.set_ticks(np.linspace(0, 1, 5, endpoint=True))
        ax.xaxis.set_ticks(np.linspace(0, 1, 5, endpoint=True))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(r'${%g}$'))
        x = np.linspace(0, 1, 16+1, endpoint=True)
        plt.plot(x, p.numpy(), '-', markersize=3, color=col, alpha=0.35 if not single or i > 0 else 1, linewidth=1 if not single or i > 0 else 2)
    if plin_number % 4 != 0:
        plt.tick_params(axis='y', which='both', labelleft='off')
    else:
        plt.ylabel('$f(x)$')
    first = False
    if not single:
        plt.title('$f_{}$, ${}={}$'.format(plin_number + 1, 'q' if target_var == 'coord' else 'l', other_configs[0]))
    else:
        plt.title('$f_{}$'.format(plin_number + 1))
    plt.xlabel('$x$')

color_label = '$l$' if target_var == 'coord' else '$q$'
plt.tight_layout()
fig.subplots_adjust(right=0.91 - only_1_and_2 * 0.1 + 0.075 * single, left=0.125, hspace=0.18 if only_1_and_2 else 0.5, wspace=0.1 if only_1_and_2 else 0.08, bottom=0.18 if only_1_and_2 else 0.1)

if not single:
    cbar_ax = fig.add_axes([0.93 - only_1_and_2 * 0.1, 0.2 if only_1_and_2 else 0.15, 0.03, 0.6])
    cbar_ax.set_title(color_label)
    ticks = np.linspace(min(configs), max(configs), 5, endpoint=True)
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm, norm=cnorm, orientation='vertical', ticks=ticks)
plt.savefig('{}{}.pdf'.format(target_var, '' if only_1_and_2 else '-full'))
