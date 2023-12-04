import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='HDF5 basename for the splits')
args = parser.parse_args()

from tensorflow.keras.layers import Dense, Conv2D

# Input consists of a 160x90x6 array.
# The first 3 channels correspond to the grayscale image, altitude, and heading (yaw).
# The last 3 channels correspond to the grayscale image, altitude, and heading (yaw) of the next frame.

INPUT_SHAPE = (160, 90, 6)

BATCH_SIZE = 64
EPOCHS = 50

MODEL_PATH = f"{args.input}_checkpoint"
TRAIN_HDF5 = f"{args.input}_train.h5"
VALID_HDF5 = f"{args.input}_val.h5"
TEST_HDF5 = f"{args.input}_test.h5"

# Load the data
train_data = HDF5DatasetGenerator(TRAIN_HDF5, BATCH_SIZE)
valid_data = HDF5DatasetGenerator(VALID_HDF5, BATCH_SIZE)
test_data = HDF5DatasetGenerator(TEST_HDF5, BATCH_SIZE)

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
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[MeanSquaredError()])
model.summary()

# Train the model
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(train_data, validation_data=valid_data, epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

# Evaluate the model
model.evaluate(test_data, verbose=1)
