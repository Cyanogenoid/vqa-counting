import sys
import os.path
import argparse
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils


def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for v, q, a, b, idx, q_len in loader:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        b = Variable(b.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        out = net(v, b, q, q_len)
        if has_answers:
            nll = -F.log_softmax(out, dim=1)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.data[0])
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', nargs='*')
    args = parser.parse_args()

    if args.test:
        args.eval_only = True
    src = open('model.py').read()
    if len(sys.argv) > 1:
        name = ' '.join(args.name)
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    if not args.test:
        # target_name won't be used in test mode
        print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    if not args.eval_only:
        train_loader = data.get_loader(train=True, val=True)
    if args.test:
        val_loader = data.get_loader(test=True)
        train_loader = val_loader

    net = model.Net(train_loader.dataset.num_tokens).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / config.lr_halflife))
    if args.resume:
        net.load_state_dict(torch.load(' '.join(args.resume))['weights'])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        if not args.eval_only:
            run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)
        else:
            r = run(net, val_loader, optimizer, scheduler, tracker, train=False, has_answers=False, prefix='test', epoch=i)

        if not args.test:
            results = {
                'name': name,
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.state_dict(),
                'vocab': train_loader.dataset.vocab,
                'src': src,
            }
            torch.save(results, target_name)
        else:
            # in test mode, save a results file in the format accepted by the submission server
            answer_index_to_string = {a:  s for s, a in val_loader.dataset.answer_to_index.items()}
            results = []
            for answer, index in zip(r[0], r[2]):
                answer = answer_index_to_string[answer]
                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer,
                }
                results.append(entry)
            with open('results.json', 'w') as fd:
                json.dump(results, fd)

        if args.eval_only:
            break


if __name__ == '__main__':
    main()
