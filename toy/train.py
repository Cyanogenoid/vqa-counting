import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

import data
import model


def extract_plins(net):
    plins = [net.counter.f[i] for i in range(8)]
    return [p.weight.data.cpu().clone() for p in plins]


def get_loader(*args, **kwargs):
    dataset = data.ToyTask(*args, **kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=8,
        pin_memory=True,
    )
    return loader


def run(nets, loader, iterations, train):
    if train:
        [net.train() for net in nets]
    else:
        [net.eval() for net in nets]
    optimizers = [torch.optim.Adam(net.parameters(), lr=0.01) for net in nets]

    loss_function = nn.CrossEntropyLoss()
    data = []
    tq = tqdm(loader, total=iterations, ncols=0, position=2, desc='train' if train else 'val')
    for i, (a, b, c) in enumerate(tq):
        if i >= iterations:
            break

        a = Variable(a.cuda(async=True), requires_grad=False)
        b = Variable(b.cuda(async=True).transpose(1, 2).contiguous(), requires_grad=False)
        c = Variable(c.cuda(async=True), requires_grad=False)

        pred_cs = [net(a, b) for net in nets]
        losses = [loss_function(pred_c, c) for pred_c in pred_cs]

        if train:
            [optimizer.zero_grad() for optimizer in optimizers]
            [loss.backward() for loss in losses]
            [optimizer.step() for optimizer in optimizers]
            data.append(extract_plins(nets[0]))
        else:
            acc = [(pred_c.data.max(dim=1)[1] == c.data).float().mean() for pred_c in pred_cs]
            data.append(acc)
    data = list(zip(*data))
    if not train:
        data = [np.mean(d) for d in data]
    return data


def main(objects, **kwargs):
    nets = [
        model.Net(objects).cuda(),
        model.Baseline(objects).cuda(),
    ]
    loader = get_loader(objects, **kwargs)
    plins = run(nets, loader, 1000, train=True)
    accs = run(nets, loader, 200, train=False)
    return {'plins': plins, 'accs': accs}


resolution = 100 + 1
configuration = sys.argv[1]

params = {
    'easy': {
        'objects': 10,
        'coord': 0.0,
        'noise': 0.0,
    },
    'hard': {
        'objects': 10,
        'coord': 0.5,
        'noise': 0.5,
    },
}[configuration]

param_ranges = {
    'coord': torch.linspace(0, 1, resolution),
    'noise': torch.linspace(0, 1, resolution),
}


for name, ran in tqdm(param_ranges.items(), ncols=0, desc='all', position=0):
    logs = []
    for x in tqdm(ran, ncols=0, desc=name, position=1):
        p = dict(params)
        p[name] = x
        log = main(**p)
        log['config'] = p
        logs.append(log)
    filename = '{}-{}.pth'.format(name, configuration)
    torch.save(logs, filename)
