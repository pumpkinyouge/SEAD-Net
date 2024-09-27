# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader

from torchvision import transforms

from toolsdev import *
from sklearn.model_selection import train_test_split

from utils import *

from torch import Tensor

from typing import Iterator, Sequence

from scipy.signal import savgol_filter

import random
from torch.utils.data import Sampler

def get_dataloaders(args):
    '''
    Retrives the dataloaders for the dataset of choice.

    Initalise variables that correspond to the dataset of choice.

    args:
        args (dict): Program arguments/commandline arguments.

    returns:
        dataloaders (dict): pretrain,train,test set split dataloaders.

        args (dict): Updated and Additional program/commandline arguments dependent on dataset.

    '''
    if args.dataset == 'nslkdd_normalization':
        dataset = 'NSLKDD'

        args.class_names = None

        args.crop_dim = 64
        args.data_size = 122
        args.crop_size = 122
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    elif args.dataset == 'arrhythmia_normalization':
        dataset = 'ARRHY'

        args.class_names = None

        args.crop_dim = 128
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    elif args.dataset == 'fraud_normalization':
        dataset = 'FRAUD'

        args.class_names = None

        args.crop_dim = 16
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    elif args.dataset == 'shuttle_normalization':
        dataset = 'SHUTTLE'

        args.class_names = None

        args.crop_dim = 5
        args.data_size = 9
        args.crop_size = 9
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    elif args.dataset == 'celeba_baldvsnonbald_normalised':
        dataset = 'CELEBA'

        args.class_names = None

        args.crop_dim = 20
        args.data_size = 39
        args.crop_size = 39
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    elif args.dataset == 'census-income-full-mixed-binarized':
        dataset = 'CENSUS'

        args.class_names = None

        args.crop_dim = 300
        args.data_size = 500
        args.crop_size = 500
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    elif args.dataset == 'bank-additional-full-deonehot':
        dataset = 'BANK'

        args.class_names = None

        args.crop_dim = 10
        args.data_size = 20
        args.crop_size = 20
        args.n_channels, args.n_classes = 1, 2

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        working_dir = os.path.join(working_dir, args.dataset)

        dataloaders = KDD_dataloader(args, working_dir)

    else:
        NotImplementedError('{} dataset not available.'.format(args.dataset))

    return dataloaders, args


class RandomCrop(object):
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim

    def __call__(self, sequence):
        seq_len = sequence.shape[0]
        if seq_len > self.crop_dim:
            start = torch.randint(0, seq_len - self.crop_dim, (1,))
            end = start + self.crop_dim
            cropped_sequence = sequence[start:end]
            num_pad_elements = seq_len - self.crop_dim
            pad = torch.zeros(num_pad_elements)  # Assuming sequence has 2nd dimension for features
            cropped_sequence = torch.cat((cropped_sequence, pad), dim=0)
        else:
            cropped_sequence = sequence
        return cropped_sequence


class RandomZeroReplace(object):
    def __init__(self, replace_prob):
        self.replace_prob = replace_prob

    def __call__(self, sequence):
        replaced_sequence = [
            0 if random.random() < self.replace_prob else element
            for element in sequence
        ]
        replaced_sequence = torch.tensor(replaced_sequence)
        return replaced_sequence


class RandomFlip(object):
    def __call__(self, sequence):
        flip_prob = torch.rand(1)
        if flip_prob < 0.5:
            flipped_sequence = torch.flip(sequence, dims=(0,))
        else:
            flipped_sequence = sequence
        return flipped_sequence


class AddNoise(object):
    def __init__(self, noise_type='gaussian', noise_level=0.1):
        self.noise_type = noise_type
        self.noise_level = noise_level

    def __call__(self, sequence):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(sequence) * self.noise_level
        elif self.noise_type == 'uniform':
            noise = torch.rand_like(sequence) * self.noise_level
        else:
            raise ValueError("Invalid noise type")
        noisy_sequence = sequence + noise
        return noisy_sequence


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, sequence):
        normalized_sequence = (sequence - self.mean) / self.std
        return normalized_sequence


class Negated(object):
    def __call__(self, sequence):
        return sequence * -1


class OppositeTime(object):
    def __call__(self, sequence):
        sequence_numpy = sequence.numpy().copy()
        sequence_numpy = sequence_numpy[::-1]
        sequence_tensor = torch.from_numpy(sequence_numpy.copy())
        return sequence_tensor


class DataSlicer:
    def __init__(self, num_slices, augmentation_methods):
        self.num_slices = num_slices
        self.augmentation_methods = augmentation_methods

    def __call__(self, sequence):
        # slices = torch.chunk(self.input_sequence, self.num_slices)
        augmented_slices = []

        size_blocks = (len(sequence) + self.num_slices - 1) // self.num_slices
        # Cut the input sequence into multiple small pieces
        blocks = [sequence[i:i + size_blocks] for i in range(0, self.num_slices * size_blocks, size_blocks)]

        # Random permutation of indices of small blocks
        permuted_idx = list(range(self.num_slices - 1))
        random.shuffle(permuted_idx)
        permuted_idx.append(self.num_slices - 1)
        # Rearrange the tiles according to the randomly arranged indices
        permuted_sequence = [blocks[i] for i in permuted_idx]

        for index, slice in enumerate(permuted_sequence, start=0):
            augmented_slice = slice.clone()
            augmentation_method = self.augmentation_methods[index]
            augmented_slice = augmentation_method(augmented_slice)
            augmented_slices.append(augmented_slice)
        # augmented_slices.append(permuted_sequence[self.num_slices - 1])

        augmented_slices_numpy = [tensor.numpy() for tensor in augmented_slices]
        augmented_slices_numpy = np.concatenate(augmented_slices_numpy, axis=0)
        augmented_slices_tensor = torch.from_numpy(augmented_slices_numpy.copy())

        return augmented_slices_tensor


class Permuted(object):
    def __call__(self, sequence):
        block_size = 5
        num_blocks = (len(sequence) + block_size - 1) // block_size  # Calculate the number of chunks, including the last chunk that is less than block_size

        # Cut the input sequence into multiple small pieces
        blocks = [sequence[i:i + block_size] for i in range(0, num_blocks * block_size, block_size)]

        # Random permutation of indices of small blocks
        permuted_idx = list(range(num_blocks - 1))
        random.shuffle(permuted_idx)
        permuted_idx.append(num_blocks - 1)

        # Rearrange the tiles according to the randomly arranged indices
        permuted_sequence = [blocks[i] for i in permuted_idx]
        permuted_sequence_numpy = [tensor.numpy() for tensor in permuted_sequence]
        permuted_sequence_numpy = np.concatenate(permuted_sequence_numpy, axis=0)
        sequence_tensor = torch.from_numpy(permuted_sequence_numpy.copy())

        return sequence_tensor


class RandomContrast(object):
    def __init__(self, contrast_range=(0.9, 1.1)):
        self.contrast_range = contrast_range

    def __call__(self, sequence):
        # Generate a random contrast adjustment factor
        contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])

        # Contrast adjustment enhancement for the sequence
        enhanced_sequence = (sequence - np.mean(sequence.numpy())) * contrast_factor + np.mean(sequence.numpy())

        return enhanced_sequence


class RandomScale(object):
    def __init__(self, scale_factor_range=(0.9, 1.1)):
        self.scale_factor_range = scale_factor_range

    def __call__(self, sequence):
        # Generate random scaling factor
        scale_factor = np.random.uniform(self.scale_factor_range[0], self.scale_factor_range[1])

        # Scaling enhancement for time series
        scaled_sequence = sequence * scale_factor

        return scaled_sequence


class RandomBrightness(object):
    def __init__(self, brightness_range=(-0.2, 0.2)):
        self.brightness_range = brightness_range

    def __call__(self, sequence):
        # Generate random brightness adjustment values
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])

        # Enhance brightness of sequence
        enhanced_sequence = sequence + brightness_factor

        return enhanced_sequence


class RandomLogBrightness(object):
    def __init__(self, scale_range=(1, 1)):
        self.scale_range = scale_range

    def __call__(self, sequence):
        # Generate a random log-transform scale factor
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Log transform enhancement of time series
        brightened_sequence = np.log(sequence) * scale_factor * -1
        # brightened_sequence = torch.from_numpy(brightened_sequence.copy())

        return brightened_sequence

class RandomTranslation(object):
    def __init__(self, max_shift=5):
        self.max_shift = max_shift

    def __call__(self, sequence):
        # Generate a random translation
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)

        # Apply time series translation enhancement
        shifted_sequence = np.roll(sequence, shift)
        shifted_sequence = torch.from_numpy(shifted_sequence.copy())

        return shifted_sequence


class Scale(object):
    def __call__(self, sequence):
        sc = [0.5, 2, 1.5, 0.8]
        s = random.choice(sc)
        return sequence * s


def inter_data(hr, window=11):
    N = window
    time3 = savgol_filter(hr, window_length=N, polyorder=2)
    return time3


class TimeWarp(object):
    def __call__(self, sequence):
        for i in range(sequence.shape[1]):
            sequence[:, i] = inter_data(sequence[:, i], 11)  # Here you need to provide the implementation of inter_data
        return sequence


class AlternatingSampler(Sampler):
    def __init__(self, score_batch_indices):
        self.score_batch_indices = score_batch_indices

    def __iter__(self):
        yield from iter(self.score_batch_indices)

    def __len__(self):
        return len(self.score_batch_indices)


def inject_noise(seed, n_out, random_seed):
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        # >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        # >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        # [4, 4, 1, 4, 5]
        # >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        # [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    n_batch_num: int
    batchsize: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int, n_batch_num: int, batchsize: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(n_batch_num, int) or isinstance(n_batch_num, bool) or \
                n_batch_num <= 0:
            raise ValueError("n_batch_num should be a positive integer "
                             "value, but got n_batch_num={}".format(n_batch_num))
        if not isinstance(batchsize, int) or isinstance(batchsize, bool) or \
                batchsize <= 0:
            raise ValueError("batchsize should be a positive integer "
                             "value, but got batchsize={}".format(batchsize))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.n_batch_num = n_batch_num
        self.batchsize = batchsize
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = rand_tensor.tolist()
        random_samples = random.sample(rand_tensor, self.n_batch_num * self.batchsize)
        yield from iter(random_samples)

    def __len__(self) -> int:
        return self.n_batch_num * self.batchsize



def KDD_dataloader(args, working_dir):
    '''
    args:
        args (dict): Program/commandline arguments.

        dataset_paths (dict): Paths to each datset split.

    Returns:

        dataloaders (): pretrain,train,test set split dataloaders.
    '''

    transf_1d = {
        'train': transforms.Compose([
            # RandomCrop(args.autoencoder_crop_size),
            # RandomFlip(),
            # Scale(),
            # AddNoise(noise_type='gaussian', noise_level=0.1),
            # Negated(),
            # OppositeTime(),
            # Permuted(),
            # RandomScale(),
            # RandomZeroReplace(0.5),
            # RandomTranslation(),
            # RandomBrightness(),
            # RandomContrast(),
            # DataSlicer(num_slices=5, augmentation_methods=[RandomFlip(), AddNoise(), Normalize(mean=5.5, std=2.87), Negated(), OppositeTime()]),
            # Normalize(mean=(0.5,), std=(0.5,))
        ]),
        'pretrain': transforms.Compose([
            RandomCrop(args.autoencoder_crop_size),
            # RandomFlip(),
            # Scale(),
            # AddNoise(noise_type='gaussian', noise_level=0.1),
            # Negated(),
            # OppositeTime(),
            # Permuted(),
            # RandomScale(),
            # RandomZeroReplace(0.25),
            # RandomTranslation(),
            # RandomBrightness(),
            # RandomContrast(),
            # DataSlicer(num_slices=5, augmentation_methods=[RandomFlip(), AddNoise(), Normalize(mean=5.5, std=2.87), Negated(), OppositeTime()]),
            # Normalize(mean=(0.5,), std=(0.5,))
        ]),
        'test': transforms.Compose([
            # RandomCrop(args.crop_size),
            # Normalize(mean=(0.5,), std=(0.5,))
        ])
    }

    val_samples = 256

    # data extraction
    x, labels = dataLoading(working_dir + '.csv', byte_num=args.data_size)
    outlier_indices = np.where(labels == 1)[0]
    outliers = x[outlier_indices]

    # inlier_indices = np.where(labels == 0)[0]
    train_x, test_x, train_label, test_label = train_test_split(x, labels, test_size=0.2, random_state=43,
                                                                stratify=labels)
    test_outlier_indices = np.where(test_label == 1)[0]
    test_inlier_indices = np.where(test_label == 0)[0]
    train_outlier_indices = np.where(train_label == 1)[0]
    train_inlier_indices = np.where(train_label == 0)[0]

    # Data pollution and limiting negative sample size
    args.cont_rate = 0.02
    args.known_outliers = 30
    n_outliers = len(train_outlier_indices)
    n_noise = len(np.where(train_label == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
    n_noise = int(n_noise)
    random_seed = 43
    rng = np.random.RandomState(random_seed)

    # Select only a fixed number of abnormal samples and contaminate normal samples
    if n_outliers > args.known_outliers:
        mn = n_outliers - args.known_outliers
        remove_idx = rng.choice(train_outlier_indices, mn, replace=False)
        train_x = np.delete(train_x, remove_idx, axis=0)
        train_label = np.delete(train_label, remove_idx, axis=0)
    noises = inject_noise(outliers, n_noise, random_seed)
    train_x = np.append(train_x, noises, axis=0)
    train_label = np.append(train_label, np.zeros((noises.shape[0], 1)))

    train_outlier_indices = np.where(train_label == 1)[0]
    train_inlier_indices = np.where(train_label == 0)[0]

    # Pretrained full normal sample
    train_x_inlier = train_x[train_inlier_indices]
    train_label_inlier = train_label[train_inlier_indices]

    datasets = {
        'train': torch.from_numpy(train_x),
        'test': torch.from_numpy(test_x),
        'pretrain': torch.from_numpy(train_x_inlier)
    }

    datasets['train'].data = torch.from_numpy(train_x)
    datasets['train'].targets = torch.from_numpy(train_label)
    datasets['test'].data = torch.from_numpy(test_x)
    datasets['test'].targets = torch.from_numpy(test_label)
    datasets['pretrain'].data = torch.from_numpy(train_x_inlier)
    datasets['pretrain'].targets = torch.from_numpy(train_label_inlier)


    # weighted sampler weights for new training set
    f_s_weights = sample_weights(datasets['train'].targets)

    # save original full training set  #Now there are 30 anomalies in datasets['train'].
    datasets['train_valid'] = datasets['train']

    # make new pretraining set without validation samples
    datasets['pretrain'] = CustomDataset(data=datasets['pretrain'].data,
                                         labels=datasets['pretrain'].targets,
                                         transform=transf_1d['pretrain'],
                                         two_crop=args.twocrop)
                                         # two_crop = False)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=datasets['train'].data,
                                      labels=datasets['train'].targets,
                                      # transform=transf_1d['train'],
                                      two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=datasets['test'].data,
                                      labels=datasets['test'].targets,
                                      # transform=transf_1d['test'],
                                      two_crop=False)

    datasets['test'] = CustomDataset(data=datasets['test'].data,
                                     labels=datasets['test'].targets,
                                     # transform=transf_1d['test'],
                                     two_crop=False)
                                        # labels=labels['test'], transform=transf_1d['test'], two_crop=False)

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['pretrain'].labels)
    total_batches = 20
    pretrain_total_batches = 20
    batch_size = 512

    # Use train_label to calculate the index of the network sample for the score
    normal_indices = []
    anomaly_indices = []
    for index in range(len(train_label)):
        label = train_label[index]  # The CustomDataset function is called here for data enhancement.
        if label == 0:  # Assuming that a label of 0 indicates a normal sample and a label of 1 indicates an abnormal sample, return the indices of all normal and abnormal samples.
            normal_indices.append(index)
        else:
            anomaly_indices.append(index)
    # Generate a batchsize with the specified number of positive and negative data samples
    score_batch_indices = []
    for _ in range(total_batches):
        normal_batch_indices = []
        anomaly_batch_indices = []
        for i in range(batch_size // 2):
            normal_batch_indices = random.choices(normal_indices, k=1)
            anomaly_batch_indices = random.choices(anomaly_indices, k=1)
            score_batch_indices.append(normal_batch_indices)
            score_batch_indices.append(anomaly_batch_indices)
    score_batch_indices = np.squeeze(score_batch_indices)

    config = {
        'pretrain': WeightedRandomSampler(s_weights,
                                          num_samples=len(s_weights), n_batch_num=pretrain_total_batches, batchsize=batch_size, replacement=True),

        'train': AlternatingSampler(score_batch_indices),

        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), n_batch_num=pretrain_total_batches, batchsize=batch_size ,replacement=True),

        'valid': None, 'test': None

    }

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=0, pin_memory=True, drop_last=(i != 'train' and i != 'train_valid' and i !='valid' and i != 'test'),
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders
