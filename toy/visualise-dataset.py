import sys
import random

import data

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')


q = float(sys.argv[1])

# guessing seeds for nice looking datasets
torch.manual_seed(int(2 * q) + 10)
random.seed(int(2 * q) + 16)

cm = plt.cm.coolwarm
params = [
    (0.05, q),
    (0.1, q),
    (0.2, q),
    (0.3, q),
    (0.4, q),
    (0.5, q),
]

n = 0
plt.figure(figsize=(4, 11.5), dpi=200)
for coord, noise in params:
    dataset = data.ToyTask(10, coord, noise)

    a, b, c = next(iter(dataset))

    ax_true = plt.subplot(len(params), 2, n + 1, aspect='equal')
    ax_data = plt.subplot(len(params), 2, n + 2, aspect='equal')
    for i, (weight, box) in enumerate(zip(a, b)):
        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        config = {
            'alpha': 0.3,
            'linewidth': 0,
        }
        ax_true.add_patch(patches.Rectangle(
            (x, y), w, h,
            **config,
            color=cm(1 - float(i < c))
        ))
        ax_data.add_patch(patches.Rectangle(
            (x, y), w, h,
            **config,
            color=cm(1 - weight)
        ))
        ax_true.axes.get_xaxis().set_visible(False)
        ax_data.axes.get_xaxis().set_visible(False)
        ax_true.axes.get_yaxis().set_major_locator(plt.NullLocator())
        ax_data.axes.get_yaxis().set_visible(False)
        ax_true.set_title('Ground truth: {}'.format(c))
        ax_data.set_title('Data')
        ax_true.set_ylabel('$l = {}$'.format(coord))
    n += 2
    plt.suptitle('\Large$q = {}$'.format(noise))

plt.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.0, hspace=0)
plt.savefig('dataset-{}.pdf'.format(int(round(10 * q))))
