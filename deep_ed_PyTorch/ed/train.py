import os
import time
import argparse
from itertools import chain
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.ed.args import arg_parse
from deep_ed_PyTorch.ed.model import ModelLocal, ModelGlobal
from deep_ed_PyTorch.ed.minibatch import DataLoader

from deep_ed_PyTorch.utils import utils

from deep_ed_PyTorch.ed.test.test import Test

''' **YD** TODO list 11/1/2020
1. add learning rate decay (when aida-testa > 0.9, set lr to 0.4???)
2. allow device set ups in the global model
3. record performances figures by time in training and testing?
'''

class ED(object):
    def __init__(self, args):
        self.args = args

        self.data_loader = DataLoader(args)
        self.build_minibatch = self.data_loader.build_minibatch

        if args.model_type == 'local':
            self.model = ModelLocal(args)
        else:
            self.model = ModelGlobal(args)


def train(args):

    ed = ED(args)

    # Define model
    model = ed.model
    args.model = model

    test = Test(args)
    print(model)

    # Define criterion
    if args.loss == 'maxm':
        # **YD** detailed parameters not valid
        criterion = nn.MultiMarginLoss(p=1, weight=torch.ones(args.max_num_cand), margin=0.01)
    else:
        raise ValueError('unknown criterion')

    # define test class here, after defining the model.

    if 'cuda' in args.type:
        model.cuda()
        criterion.cuda()

    model.train()
    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(model.parameters()),
        )
    )

    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Define optimizer
    if args.optimization == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(params, lr=args.lr)
    elif args.optimization == 'ADAM':
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        raise ValueError('unknown optimizer' + args.optimization)

    epoch = 0
    if not hasattr(args, 'max_epoch'):
        args.max_epoch = 400

    while epoch < args.max_epoch:
        start_time = time.time()

        print('One epoch = ' + str(args.num_batches_per_epoch / 1000) + ' full passes over AIDA-TRAIN in our case.')
        print('==> TRAINING EPOCH #' + str(epoch) + ' <==')

        # **YD** "print_net_weights" not implemented
        # print_net_weights()

        processed_mentions = 0

        if args.type == 'cuda':
            model.cuda()
            criterion.cuda()

        model.train()
        # **YD** "args.num_batches_per_epoch" has been implemented
        for batch_index in tqdm(range(args.num_batches_per_epoch)):

            # -- Read one mini-batch from one data_thread:
            # **YD** "get_minibatch" has been implemented
            inputs, targets = ed.data_loader.get_minibatch()

            if args.type == 'cuda':
                inputs = utils.move_to_cuda(inputs)
                targets = utils.move_to_cuda(targets)

            num_mentions = targets.size(0)
            processed_mentions += num_mentions

            # **YD** "get_model" not implemented, structure is wired
            # model, _ = get_model(num_mentions)

            optimizer.zero_grad()
            outputs, beta, entity_context_sim_scores = model(inputs)

            # **YD** core debugging
            # print("outputs.size()", outputs.size(), "targets.size()", targets.size())
            # print("outputs", outputs)
            # print("targets", targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # -- Regularize the f_network with projected SGD.
            # *YD** "regularize_f_network" has been implemented
            regularize_f_network(model)

        # -- Measure time taken
        total_time = time.time() - start_time

        single_time = total_time / processed_mentions

        print("epoch: {} takes {}s".format(epoch, total_time))
        print("==> time to learn 1 sample = " + str(single_time * 1000) + 'ms')

        # -- Test:
        # **YD** "test" has been implemented
        # args.type = 'double'
        # test.test(epoch)
        # args.type = 'cuda'

        # **YD** "args.save_interval" has been implemented
        if epoch % args.save_interval == 0:
            file_dir = os.path.join(args.root_data_dir, 'generated/ed_models/')
            if not os.path.exists(file_dir):
                os.makedirs(file_dir,exist_ok=True)

            file_name = os.path.join(args.root_data_dir, 'generated/ed_models/' +
                                     args.model_type + '_' + str(epoch) + '.pt')
            print('==> saving model to ', file_name)

            # **YD** model save logic has rewritten
            torch.save(model.state_dict(), file_name)

        epoch += 1


def regularize_f_network(model):
    with torch.no_grad():
        for p in [model.linear1.weight, model.linear1.bias, model.linear2.weight, model.linear2.bias]:
            if p.norm() > 1:
                p.mul_(1/p.norm())


def main():
    args = arg_parse()
    train(args)


if __name__ == '__main__':
    main()