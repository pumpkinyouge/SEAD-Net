#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import configargparse
import warnings

from train import evaluate, pretrain, score_finetune
from datasets import get_dataloaders
from utils import *
import model.network as models
from model.SEAD import SEAE


warnings.filterwarnings("ignore")


default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

parser = configargparse.ArgumentParser(
    description='Pytorch SEAE', default_config_files=[default_config])
parser.add_argument('--dataset', default='imagenet',
                    help='Dataset, (Options: cifar10, cifar100, stl10, imagenet, tinyimagenet).')
parser.add_argument('--model', default='autoencoder',
                    help='Model, (Options: autoencoder).')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--learning_rate_finetune', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--base_lr_finetune', type=float, default=0.0001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--c_lr', type=float, default=1,
                    help='Contrastive Learning Rate for Model Training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='sgd',
                    help='Optimiser, (Options: sgd, adam).')
parser.add_argument('--patience', default=50, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--queue_size', type=int, default=65536,
                    help='Size of Memory Queue, Must be Divisible by batch_size.')
parser.add_argument('--queue_momentum', type=float, default=0.999,
                    help='Momentum for the Key Encoder Update.')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='InfoNCE Temperature Factor')
parser.set_defaults(twocrop=True)
parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
# parser.set_defaults(finetune=False)
parser.set_defaults(finetune=True)
parser.add_argument('--supervised', dest='supervised', action='store_true',
                    help='Perform Supervised Pre-Training. (Default: False)')
parser.set_defaults(supervised=False)
parser.add_argument('--input_dim', type=int, default=256,
                    help='Input dimension of the autoencoder')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='Dimension of the hidden layer in the autoencoder')
parser.add_argument('--crop_size', type=int, default=29,
                    help='Dimension of the hidden layer in the autoencoder')
parser.add_argument('--autoencoder_crop_size', type=int, default=15,
                    help='Dimension of the hidden layer in the autoencoder')
parser.add_argument('--mode', type=str, default=pretrain,
                    help='Dimension of the hidden layer in the autoencoder')


def setup():
    """
    returns:
        local_rank (int): rank of local machine / host to perform distributed training.

        device (string): Device and rank of device to perform training on.
    """
    local_rank = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, local_rank


def main():
    """ Main """
    runs = 1
    finetune_epochs = 10

    rauc = np.zeros((runs, finetune_epochs + 1))
    ap = np.zeros((runs, finetune_epochs + 1))

    for t in range(runs):

        # Arguments
        args = parser.parse_args()

        # Setup Training
        device, local_rank = setup()

        # Get Dataloaders for Dataset of choice
        dataloaders, args = get_dataloaders(args)

        # Setup logging, saving models, summaries
        args = experiment_config(parser, args)

        args.finetune_epochs = finetune_epochs
        # args.n_epochs = 150 + 1 * (t + 1)
        args.n_epochs = 6
        args.runs = runs

        ''' Base Encoder '''

        # Get available models from /model/network.py
        model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

        # If model exists
        if any(args.model in model_name for model_name in model_names):
            # Load model
            base_encoder = getattr(models, args.model)(
                args, args.mode ,num_classes=args.n_classes)  # Encoder

        else:
            raise NotImplementedError("Model Not Implemented: {}".format(args.model))

        if not args.supervised:
            # freeze all layers
            for name, param in base_encoder.named_parameters():
                if name not in ['fc.weight', 'fc.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
                    param.requires_grad = False
            # init base_encoder
            init_weights(base_encoder)

        ''' SEAE Model '''
        ae = SEAE(args, queue_size=args.queue_size,
                          momentum=args.queue_momentum, temperature=args.temperature)

        gpu_id = torch.cuda.current_device()
        print('\nUsing', gpu_id, ':', torch.cuda.get_device_name(gpu_id), 'GPU(s).\n')
        logging.info('\nn_epochs: {} \t finetune_epochs: {}:\n'.format(args.n_epochs, args.finetune_epochs))

        ae.to(device)
        base_encoder.to(device)

        args.print_progress = True

        # Print Network Structure and Params
        if args.print_progress:
            # print_network(ae, args)  # prints out the network architecture etc
            logging.info('\npretrain/train: {} - valid: {} - test: {}'.format(
                len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
                len(dataloaders['test'].dataset)))

        # launch model training or inference
        max_model_save_sum = float('-inf')
        if not args.finetune:

            ''' Pretraining / Finetuning / Evaluate '''
            if not args.supervised:
                # Pretrain the autoencoder
                pretrain(ae, dataloaders, args)

                # Load the state_dict from query encoder and load it
                base_encoder = load_sclm(base_encoder, args)

            score_finetune(base_encoder, dataloaders, args, rauc, ap, t)

        else:

            ''' Evaluate '''

            # Do not Pretrain, just inference
            # Evaluate the model
            args.load_checkpoint_dir = '../experiments/2023-11-08_09-10-32/checkpoint_finetune.pt'
            base_encoder = load_ae(base_encoder, args)
            # test_loss, test_acc, test_acc_top5, model_save_index, max_model_save_sum = evaluate(
            model_save_index, max_model_save_sum = evaluate(base_encoder, dataloaders, 'test', args.finetune_epochs, args, rauc, ap, t, max_model_save_sum)


if __name__ == '__main__':
    main()
