import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')


name_lookup = {
    'coord': 'l',
    'noise': 'q',
}


base = sys.argv[1]

plt.figure(figsize=(8, 2.05), dpi=200)
paths = list(os.listdir(base))
i = 0
lines = []
for path in paths:
    full_path = os.path.join(base, path)
    if not (path.startswith('noise') or path.startswith('coord')):
        continue
    target_var = path.split('-')[0]
    logs = torch.load(full_path)
    accs = [x['accs'] for x in logs]
    configs = [x['config'][target_var] for x in logs]
    other_configs = [x['config']['noise' if target_var == 'coord' else 'coord'] for x in logs]

    ax = plt.subplot(1, 4, i + 1)
    plt.ylim(0, 1)
    plt.xlim(min(configs), max(configs))
    plt.xticks(np.linspace(min(configs), max(configs), 5, endpoint=True))
    name = name_lookup[target_var]
    plt.xlabel('${}$'.format(name))
    val = str(other_configs[0]) if name == 'l' or other_configs[0] != 0.0 else '10^{-6}'
    plt.title('${}={}$'.format('q' if name == 'l' else 'l', val))
    plt.grid()
    ax.set_prop_cycle(cycler('color', ['#004CDB', '#FF7800']))
    l = plt.plot(configs, accs, '-')
    lines.append(l)

    i += 1

plt.legend(
    list(zip(*lines)),
    ['Counting module', 'Baseline'],
    ncol=2,
    bbox_to_anchor=(-0.5, -0.6),
    loc='lower right',
    frameon=False,
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.30, top=0.89)
plt.savefig('acc.pdf')
