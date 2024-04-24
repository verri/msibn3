import h5py
import numpy as np
import argparse
from keras.models import load_model

parser = argparse.ArgumentParser(description='Predict trajectory')

parser.add_argument('--test', type=str, help='hdf5 file for the test set')
parser.add_argument('--model', type=str, help='checkpoint folder for the Keras model')
parser.add_argument('--output', type=str, help='output csv file')

args = parser.parse_args()

f = h5py.File(args.test, 'r')

model = load_model(args.model)

y_pred = model.predict(f['X'])

np.savetxt(args.output, y_pred, delimiter=',', header='x,y', comments='')
f.close()
