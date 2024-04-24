#XXX do not use this file, for some reason it does not work
import h5py
import numpy as np
from datasets.hdf5 import DataGenerator, FlightSimulator
from config import params
import argparse

parser = argparse.ArgumentParser()
# The input is a list of HDF5 files.
parser.add_argument('input', type=str, nargs='+')
# The output is a single HDF5 file.
parser.add_argument('output', type=str)

# Number of simulations to run.
parser.add_argument('--n', type=int, default=10)

# Seed
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# Open output file
with h5py.File(args.output, 'w') as f:

    f.create_dataset('X', (0, ) + params.INPUT_SHAPE, maxshape=(None,) + params.INPUT_SHAPE, dtype='float32')
    f.create_dataset('y', (0, 2), maxshape=(None, 2), dtype='float32')

    for filename in args.input:
        with h5py.File(filename, 'r') as g:
            print(f'Simulating {filename}')
            generator = DataGenerator(
                    FlightSimulator(g),
                    params.MAX_ALTITUDE,
                    params.BATCH_SIZE,
                    augment=True)

            for i in range(args.n):
                print(f'\tSimulation {i}')
                for X, y in generator.generate(rng):
                    start_idx = len(f['X'])
                    f['X'].resize(start_idx + len(X), axis=0)
                    f['X'][start_idx:] = X

                    f['y'].resize(start_idx + len(y), axis=0)
                    f['y'][start_idx:] = y
