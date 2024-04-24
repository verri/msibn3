import h5py
import numpy as np
from datasets.hdf5 import DataGenerator, FlightSimulator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from config import params
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--training', nargs='+', help='List of files to use as training set')
parser.add_argument('--validation', help='File to use as validation set')
parser.add_argument('--test', help='File to use as test set')

parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Input consists of a h x w x 8 array.
# The first 4 channels correspond to the grayscale image, heading (yaw) [x, y], and  altitude.
# The last 4 channels correspond to the same info of the next frame.

INPUT_SHAPE = params.INPUT_SHAPE

BATCH_SIZE = params.BATCH_SIZE
EPOCHS = params.EPOCHS
MAX_ALTITUDE = params.MAX_ALTITUDE

# Name of the model is a timestamp YYYYMMDDHHMMSS
name = datetime.now().strftime("%Y%m%d%H%M%S")

MODEL_PATH = f"{name}_checkpoint"
TRAIN_HDF5 = args.training
VALID_HDF5 = args.validation
TEST_HDF5 = args.test

train_files = [ h5py.File(file, 'r') for file in TRAIN_HDF5 ]
valid_file = h5py.File(VALID_HDF5, 'r')
test_file = h5py.File(TEST_HDF5, 'r')

# Load the data
train_data = DataGenerator(FlightSimulator(train_files), MAX_ALTITUDE,
                           BATCH_SIZE, augment=True)

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
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01))
model.summary()

# Train the model
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min')

rng = np.random.default_rng(args.seed)

STEPS_PER_EPOCH = params.STEPS_PER_EPOCH

model.fit(
    train_data.generate(rng),
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=(valid_file['X'], valid_file['y']),
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1)

# Evaluate the model
model.evaluate(test_file['X'], test_file['y'], verbose=1)

# Close the HDF5 files
for file in train_files:
    file.close()
valid_file.close()
test_file.close()
