"""Given HDF5 filename from command line, split into train, test, and validation sets."""

import argparse
import numpy as np
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('filename', type=str, help='HDF5 filename to split')
parser.add_argument('--train', type=float, default=0.8,
                    help='fraction of data for training')
parser.add_argument('--val', type=float, default=0.1,
                    help='fraction of data for validation')
parser.add_argument('--test', type=float, default=0.1,
                    help='fraction of data for testing')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()

frames = []


def get_names(name, obj):
    if isinstance(obj, h5py.Dataset):
        frames.append(name)


with h5py.File(args.filename, 'r') as f:
    f.visititems(get_names)

    train, val, test = np.split(frames, [int(
        args.train * len(frames)), int((args.train + args.val) * len(frames))])
    print(f'Train: {train[:5]}')
    print(f'Val: {val[:5]}')
    print(f'Test: {test[:5]}')

    filename = str(Path(args.filename).with_suffix(''))

    with h5py.File(filename + '_train.h5', 'w') as f_train:
        for name in train:
            f.copy(name, f_train)

    with h5py.File(filename + '_val.h5', 'w') as f_val:
        for name in val:
            f.copy(name, f_val)

    with h5py.File(filename + '_test.h5', 'w') as f_test:
        for name in test:
            f.copy(name, f_test)
