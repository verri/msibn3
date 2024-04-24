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

X = None
y = None

for filename in args.input:
    with h5py.File(filename, 'r') as input_file:
        generator = DataGenerator(FlightSimulator([input_file]), params.MAX_ALTITUDE, None, augment=False)
        Xnew, ynew = generator.generate_all(rng)

        if X is None:
            X = Xnew
            y = ynew
        else:
            X = np.vstack((X, Xnew))
            y = np.vstack((y, ynew))

with h5py.File(args.output, 'w') as output_file:
    output_file.create_dataset('X', data=X)
    output_file.create_dataset('y', data=y)
