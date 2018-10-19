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
import torch
from torch import optim
from torch.utils.data import DataLoader

from .dataset import SimulatedDataset
from .model import SimulatedModel
from .loss import CombinedLoss
from ....config import CONFIG


def eval_model(mod):
    pass


def get_optim_params(optimizer):
    return CONFIG.TRAIN.OPTIMIZER[optimizer]


def main(datapath, outpath, train, nepochs, cuda=False):
    dset = SimulatedDataset(datapath, train)
    batch_size = CONFIG.TRAIN.BATCH_SIZE
    loader = DataLoader(dset, batch_size)
    if train:
        model = SimulatedModel('resnet50')
        loss_function = CombinedLoss(model.outplanes)
        optim_params = get_optim_params('Adam')
        optimizer = optim.Adam(model.parameters(), **optim_params)
        for i, data in enumerate(loader):
            if i % 50 == 0:
                print("Training epoch {}.".format(i))
            if i == nepochs:
                break

            im, mask, dx, dy = data
            out = model(im)
            loss = loss_function(out, mask, dx, dy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", type=str,
                        help="Path to training data.")
    parser.add_argument("-i", "--inference",
                        type=str, help="Path to the test data.")
    parser.add_argument("-o", "--out", type=str,
                        help="Path to write inference result.")
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of epochs to train for.")
    # parser.add_argument("-bs", "--batch-size", type=int,
    #                     help="Number of images per batch.")

    train = parser.train
    datapath = parser.data
    outpath = parser.out
    epochs = parser.epochs

    has_cuda = torch.cuda.is_available()

    if has_cuda:
        print("Using CUDA.")
    else:
        print("Using CPU.")

    main(datapath, outpath, train, epochs, has_cuda)
