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

# List of files to use as training set
parser.add_argument('--training', nargs='+', default=['train'], help='List of files to use as training set')
# List of files to use as validation set
parser.add_argument('--validation', nargs='+', default=['val'], help='List of files to use as validation set')
# List of files to use as test set
parser.add_argument('--test', nargs='+', default=['test'], help='List of files to use as test set')

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
valid_files = [ h5py.File(file, 'r') for file in VALID_HDF5 ]
test_files = [ h5py.File(file, 'r') for file in TEST_HDF5 ]

# Load the data
train_data = DataGenerator(FlightSimulator(train_files), MAX_ALTITUDE,
                           BATCH_SIZE, augment=True)
valid_data = DataGenerator(
    FlightSimulator(valid_files),
    MAX_ALTITUDE,
    BATCH_SIZE)
test_data = DataGenerator(FlightSimulator(test_files), MAX_ALTITUDE, BATCH_SIZE)

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
    validation_data=valid_data.generate(rng),
    validation_steps=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1)

# Evaluate the model
test_generator = test_data.generate(rng)
model.evaluate(list(next(test_generator) for _ in range(STEPS_PER_EPOCH)), verbose=1)

# Close the HDF5 files
for file in train_files:
    file.close()
for file in valid_files:
    file.close()
for file in test_files:
    file.close()
