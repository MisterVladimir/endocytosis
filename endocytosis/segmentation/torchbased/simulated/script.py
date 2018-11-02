# -*- coding: utf-8 -*-
"""
@author: Vladimir Shteyn
@email: vladimir.shteyn@googlemail.com

Copyright Vladimir Shteyn, 2018

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from .dataset import SimulatedDataset
from .model import SimulatedModel
from .loss import CombinedLoss
from ....config import (CONFIG, reset_global_config)


def eval_model(mod):
    pass


def get_optim_params(optimizer):
    return CONFIG.OPTIMIZER[optimizer]


# not tested
def train(datapath, outpath, nepochs, cuda=False):
    reset_global_config()
    dset = SimulatedDataset(datapath, True)
    batch_size = CONFIG.SIMULATED.TRAIN.BATCH_SIZE
    loader = DataLoader(dset, batch_size)

    model = SimulatedModel('resnet50')
    loss_function = CombinedLoss(model.outplanes)
    optim_params = get_optim_params('Adam')
    optimizer = optim.Adam(model.parameters(), **optim_params)

    for i, data in enumerate(loader):
        if i % 10 == 0:
            print("Training epoch {}.".format(i))
        if i == nepochs:
            break
        if cuda:
            device = torch.device('cuda')
            model.to(device)
            loss_function.to(device)
            for tensor in data:
                tensor.to(device)

        im, mask, dx, dy = data
        out = model(im)
        loss = loss_function(out, mask, dx, dy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dset.cleanup()


# incomplete
def infer(datapath, modelpath, cuda=False):
    model = SimulatedModel('resnet50')
    dset = InferenceDataset(datapath)
    batch_size = CONFIG.INFER.BATCH_SIZE
    loader = DataLoader(dset, batch_size)
    # placeholder: load model here
    model = SimulatedModel('resnet50')

    for i, im in enumerate(loader):
        if i % 10 == 0:
            print("Inference epoch {}.".format(i))

        if cuda:
            device = torch.device('cuda')
            model.to(device)
            im.to(device)

        mask, deltas = model(im)
        start, stop = np.arange([i, i+1]) * batch_size
        indices = np.arange(start, stop)


# not tested
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--training", action='store_true',
        help="Set this flag if training. Otherwise, default to inference.")
    parser.add_argument(
        "-d", "--data", type=str,
        help="Set path to training or inference data.")
    parser.add_argument(
        "-m", "--model", type=str,
        help="Path to model. If training, this is where the model will be "
        "saved. If inference, this is where the model will be loaded from.")
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of epochs to train for.")
    parser.add_argument("-cu", "--usecuda", type=bool, default=True,
                        help="Should we use CUDA?")

    train = parser.train
    datapath = parser.data
    modelpath = parser.model
    epochs = parser.epochs
    use_cuda = parser.usecuda

    has_cuda = torch.cuda.is_available()
    if use_cuda and has_cuda:
        print("Using CUDA.")
    else:
        use_cuda = False
        if use_cuda and not has_cuda:
            print("No CUDA available. Defaulting to CPU.")
        else:
            print("Using CPU.")

    if train:
        train(datapath, modelpath, epochs, use_cuda)
    else:
        infer(datapath, modelpath, use_cuda)
