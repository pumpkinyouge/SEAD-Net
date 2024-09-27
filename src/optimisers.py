# -*- coding: utf-8 -*-
import torch.optim as optim


def get_optimiser(models, mode, args):
    '''Get the desired optimiser

    - Selects and initialises an optimiser with model params.

    Args:
        models (tuple): models which we want to optmise, (e.g. encoder)

        mode (string): the mode of training, (i.e. 'pretrain', 'finetune')

        args (Dictionary): Program Arguments

    Returns:
        optimiser (torch.optim.optimizer):
    '''
    params_models = []
    reduced_params = []

    removed_params = []

    skip_lists = ['bn', 'bias']

    for m in models:

        m_skip = []
        m_noskip = []

        params_models += list(m.parameters())

        for name, param in m.named_parameters():
            if (any(skip_name in name for skip_name in skip_lists)):
                m_skip.append(param)
            else:
                m_noskip.append(param)
        reduced_params += list(m_noskip)
        removed_params += list(m_skip)
    # Set hyperparams depending on mode
    if mode == 'pretrain':
        lr = args.learning_rate
        wd = args.weight_decay
    else:
        lr = args.learning_rate_finetune
        wd = args.finetune_weight_decay

    # Select Optimiser
    if args.optimiser == 'adam':

        optimiser = optim.Adam(params_models, lr=lr,
                               weight_decay=wd)

    elif args.optimiser == 'adamw':

        optimiser = optim.AdamW(params_models, lr=lr,
                              weight_decay=wd)

    elif args.optimiser == 'amsgrad':

        optimiser = optim.Adam(params_models, lr=lr,
                              weight_decay=wd, amsgrad=True)

    elif args.optimiser == 'sgd':

        optimiser = optim.SGD(params_models, lr=lr,
                              weight_decay=wd, momentum=0.9)

    else:

        raise NotImplementedError('{} not setup.'.format(args.optimiser))

    return optimiser

