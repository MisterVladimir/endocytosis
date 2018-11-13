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
import os
import numpy as np
import torch
import sys
import h5py
from torch import optim
from torch.utils.data import DataLoader
from vladutils.iteration import isiterable

from endocytosis.config import (CONFIG, reset_global_config)
from endocytosis.segmentation.torchbased.enums import ModelTask
from endocytosis.segmentation.torchbased.simulated.dataset import SimulationDataset
from endocytosis.segmentation.torchbased.simulated.model import SimulationModel, save
from endocytosis.segmentation.torchbased.simulated.loss import CombinedLoss


def eval_model(mod):
    pass


def get_optim_params(optimizer):
    return CONFIG.OPTIMIZER[optimizer]


def check_paths(*args):
    def test_file(filename):
        message = ''
        if not filename:
            message += "{} not a valid data path.\n".format(filename)
        elif not os.path.isfile(filename):
            message += "{} does not exist.\n".format(filename)
        return message

    error_message = ''
    for arg in args:
        error_message += test_file(arg)

    return error_message


def check_cuda(selection):
    """
    If the user elects to use the GPU, i.e. 'selection' argument is True,
    check whether a GPU is available. If it's not available, return True.
    (In this case, the 'cuda' flag should be switched to False). Otherwise, if
    GPU is available or user explicitly elects to use the CPU, return False.
    (In this case, 'cuda' flag is not changed).
    """
    cuda_available = torch.cuda.is_available()
    if selection and cuda_available:
        print("Using CUDA.")
    elif selection:
        # cuda_available is False
        print("No CUDA available. Defaulting to CPU.")
        return True
    else:
        print("Using CPU.")

    return False


def write_results(filename, attrs=None, **kwargs):
    # XXX: super clunky, but good enough for now
    with h5py.File(filename) as f:
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                arrays = [isiterable(item) for item in v]
                if all(arrays):
                    grp = f.create_group(k)
                    for i, arr in enumerate(v):
                        i = "{:0>5}".format(i) 
                        _ = grp.create_dataset(str(i), data=arr)
                else:
                    _ = f.create_dataset(k, data=v)
        if attrs is not None:
            for k, v in attrs.items():
                f.attrs.create(k, v)

    return True


def test(model, testpath, cuda=False, **kwargs):
    batch_size = CONFIG.SIMULATED.TRAIN.BATCH_SIZE
    model.eval()
    with SimulationDataset(testpath, ModelTask.TESTING) as dset:
        with torch.no_grad():
            loader = DataLoader(dset, batch_size)
            loss_function = CombinedLoss()
            nsamples = len(dset) // batch_size
            loss_list = [''] * nsamples
            deltas_list = [''] * nsamples
            if cuda:
                # training on multi-gpu not implemented
                device = torch.device('cuda')
                model = model.to(device)
                loss_function = loss_function.to(device)

            print('Testing started...', end='\n')
            for i, data in enumerate(loader):
                if cuda:
                    data = [d.to(device) for d in data]

                im, mask, dx, dy = data
                probabilities, trained_deltas = model(im)
                loss = loss_function(probabilities, trained_deltas, mask, dx, dy)
                loss_list[i] = loss.item()

                deltas_list[i] = model.apply_mask(probabilities, trained_deltas).cpu().numpy()

                print("Testing progress: {:>3.1f}%.".format(i / nsamples * 100.),
                      end='\r')

            return deltas_list, loss_list


def train(trainpath, epochs, cuda=False, **kwargs):
    batch_size = CONFIG.SIMULATED.TRAIN.BATCH_SIZE
    with SimulationDataset(trainpath, ModelTask.TRAINING) as dset:
        loader = DataLoader(dset, batch_size)
        model = SimulationModel('resnet50', ModelTask.TRAINING)
        loss_function = CombinedLoss()
        optim_params = get_optim_params('Adam')
        optimizer = optim.Adam(model.parameters(), **optim_params)
        loss_list = [''] * epochs
        if cuda:
            # training on multi-gpu not implemented
            device = torch.device('cuda')
            model = model.to(device)
            loss_function = loss_function.to(device)
        print('Training started...', end='\n')
        for i, data in enumerate(loader):
            if i == epochs:
                break
            if cuda:
                data = [d.to(device) for d in data]

            im, mask, dx, dy = data
            probabilities, trained_deltas = model(im)
            loss = loss_function(probabilities, trained_deltas, mask, dx, dy)
            loss_list[i] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Training progress: {:>4.1f}%.".format(i / epochs * 100.),
                  end='\r')

        return model, loss_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trainpath", type=str,
        help="Set path to training data.")
    parser.add_argument(
        "--testpath", type=str, default='',
        help="Set path to test data.")
    parser.add_argument(
        "-w", "--writefolder", type=str,
        help="Folder to save neural net and testing results. If not entered, "
        "model and losses will be saved to the training data's folder.")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="Number of epochs to train for.")
    parser.add_argument("-c", "--cuda", type=bool, default=True,
                        help="Should we use CUDA?")

    kwargs = vars(parser.parse_args())

    task_enum = ModelTask.TRAINING
    error_message = check_paths(kwargs['trainpath'])
    if kwargs['testpath']:
        task_enum = task_enum | ModelTask.TESTING
        error_message += check_paths(kwargs['testpath'])
    if error_message:
        raise IOError(error_message)

    if not kwargs['epochs'] > 0:
        raise Exception('Number of epochs must be greater than 0.')

    if check_cuda(kwargs['cuda']):
        kwargs.update({'cuda': False})

    model, training_losses = train(**kwargs)
    if kwargs['writefolder'] is None:
        # no folder to save model and losses was entered
        # save in same folder as training data
        writefolder = os.path.dirname(kwargs['trainpath'])
    else:
        writefolder = os.path.abspath(kwargs['writefolder'])

    modelfilename = os.path.join(writefolder, 'model.torch')
    success = save(modelfilename, model)
    if success:
        print('Model saved as {}'.format(modelfilename))
    else:
        print('Something went wrong while saving the model.')

    write_filename = os.path.join(writefolder, 'results.h5')
    savekwargs = {'train_losses': training_losses}
    success = write_results(write_filename, **savekwargs)
    if success:
        print('Training losses saved in {}'.format(write_filename))

    if task_enum | ModelTask.TESTING:
        deltas, losses = test(model, **kwargs)
        savekwargs = {'deltas': deltas, 'test_losses': losses}
        success = write_results(write_filename, **savekwargs)
        if success:
            print('Test results saved in {}'.format(write_filename))

    sys.exit(0)
