import h5py
import numpy as np
from datasets.hdf5 import DataGenerator, FlightSimulator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='HDF5 filename with all data')
args = parser.parse_args()


# Input consists of a 160x90x6 array.
# The first 3 channels correspond to the grayscale image, altitude, and heading (yaw).
# The last 3 channels correspond to the grayscale image, altitude, and
# heading (yaw) of the next frame.

INPUT_SHAPE = (160, 90, 6)

BATCH_SIZE = 64
EPOCHS = 50
MAX_ALTITUDE = 150

input_file = h5py.File(args.input, 'r')

# Load the data
data = DataGenerator(FlightSimulator(input_file), MAX_ALTITUDE,
                     BATCH_SIZE, augment=True)

# Create the model
inputs = Input(shape=INPUT_SHAPE)

x = Conv2D(32, (10, 10), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

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

rng = np.random.default_rng(17)

STEPS_PER_EPOCH = 4000

model.fit(
    data.generate(rng),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1)

model.save('model.keras')

input_file.close()
