import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='HDF5 basename for the splits')
args = parser.parse_args()

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

from datasets.hdf5 import DataGenerator, FlightSimulator

import numpy as np
import h5py

# Input consists of a 160x90x6 array.
# The first 3 channels correspond to the grayscale image, altitude, and heading (yaw).
# The last 3 channels correspond to the grayscale image, altitude, and heading (yaw) of the next frame.

INPUT_SHAPE = (160, 90, 6)

BATCH_SIZE = 64
EPOCHS = 50
MAX_ALTITUDE = 150

MODEL_PATH = f"{args.input}_checkpoint"
TRAIN_HDF5 = f"{args.input}_train.h5"
VALID_HDF5 = f"{args.input}_val.h5"
TEST_HDF5 = f"{args.input}_test.h5"


train_file = h5py.File(TRAIN_HDF5, 'r')
valid_file = h5py.File(VALID_HDF5, 'r')
test_file = h5py.File(TEST_HDF5, 'r')

# Load the data
train_data = DataGenerator(FlightSimulator(train_file), MAX_ALTITUDE,
        BATCH_SIZE, augment=True)
valid_data = DataGenerator(FlightSimulator(valid_file), MAX_ALTITUDE, BATCH_SIZE)
test_data = DataGenerator(FlightSimulator(test_file), MAX_ALTITUDE, BATCH_SIZE)

# Create the model
inputs = Input(shape=INPUT_SHAPE)

x = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(2)(x)

model = Model(inputs=inputs, outputs=x)

# Compile the model
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
model.summary()

# Train the model
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

rng = np.random.default_rng(17)

STEPS_PER_EPOCH = 2000

model.fit(
    train_data.generate(rng),
    steps_per_epoch = STEPS_PER_EPOCH,
    validation_data=valid_data.generate(rng),
    validation_steps = STEPS_PER_EPOCH // 8,
    epochs = EPOCHS,
    callbacks = [checkpoint],
    verbose = 1)

# Evaluate the model
# test_generator = test_data.generate(rng)
# model.evaluate(list(next(test_generator) for _ in range(STEPS_PER_EPOCH // 8)), verbose=1)

# Close the HDF5 files
train_file.close()
valid_file.close()
test_file.close()
