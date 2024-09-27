# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import time

import torch
from torch.utils.data import Dataset


def load_ae(base_encoder, args):
    """ Loads the finetuned SEAD model parameters.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the AE query_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir, map_location="cpu")

    # Load the encoder parameters
    state_dict = checkpoint['base_encoder']
    base_encoder.load_state_dict(state_dict, strict=True)

    return base_encoder

def load_sclm(base_encoder, args):
    """ Loads the pre-trained SCLM model parameters.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the SCLM query_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir, map_location="cpu")

    # rename SCLM pre-trained keys
    state_dict = checkpoint['sclm']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        del state_dict[k]

    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=False)

    for name, param in base_encoder.named_parameters():
        param.requires_grad = True

    return base_encoder


def init_weights(m):
    '''Initialize weights with fc
    '''

    # init the fc layer
    m.fc.weight.data.normal_(mean=0.0, std=0.01)
    m.fc.bias.data.zero_()

class CustomDataset(Dataset):
    """ Creates a custom PyTorch dataset for one-dimensional sequences.

    Args:
        data (array): Array/List of data samples.
        labels (array, optional): Array/List of labels corresponding to the data samples. (Default: None)
        transform (callable, optional): A function/transform to be applied to each data sample. (Default: None)

    Returns:
        sample (Tensor): Data sample to feed to the model.
        label (Tensor, optional): Corresponding label to the data sample.
    """

    def __init__(self, data, labels=None, transform=None, two_crop=False):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.two_crop = two_crop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform is not None:
            sample_cov = self.transform(sample)
            sample_n = sample_cov.unsqueeze(0)
            if self.two_crop:
                # Augments the images again to create a second view of the data
                sample_cov2 = self.transform(sample)
                sample_cov2 = sample_cov2.unsqueeze(0)
                # Combine the views to pass to the model
                sample_n = torch.cat([sample_n, sample_cov2], dim=0)

        # Simple deviation network
        if self.transform is None and self.two_crop is False:
            sample_n = sample
            sample_n = sample_n.unsqueeze(0)

        if self.transform is None and self.two_crop is True:
            sample = sample.unsqueeze(0)
            sample_n = torch.cat([sample, sample], dim=0)

        if self.labels is not None:
            label = self.labels[index]
            return sample_n, label
        else:
            return sample_n


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def experiment_config(parser, args):
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    run_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')

    os.makedirs(run_dir, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create all save dirs
    model_dir = os.path.join(run_dir, run_name)

    os.makedirs(model_dir, exist_ok=True)

    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')

    if not args.finetune:
        args.load_checkpoint_dir = args.checkpoint_dir

    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} \n'.format(str(key), str(value)))

    # save config file used in .txt file
    with open(os.path.join(model_dir, 'config.txt'), 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args
