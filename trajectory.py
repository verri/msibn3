import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Extract trajectory from hdf5 file')
parser.add_argument('filename', type=str, help='hdf5 file')
parser.add_argument('--output', '-o', type=str, help='output csv file')
args = parser.parse_args()

filename = args.filename
with h5py.File(filename, 'r') as f:
    y = f['y'][:]

# Write a CSV with contents of y
# The first column is called "x" and is the second "y"
np.savetxt(args.output, y, delimiter=',', header='x,y', comments='')
