from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, KLDivergence
from tensorflow.keras.optimizers import Adam

# Input consists of a 160x90x5 array.
# The first channel is the grayscale image at time t.
# The second channel is the grayscale image at time t+1.
# The third channel is the value of the altitude from the ground at time t repeated in the entire image.
# The fourth channel is the value of the altitude from the ground at time t+1 repeated in the entire image.
# The fifth channel is the value of the gimbal yaw repeated in the entire image.
#
# The ideia is that
INPUT_SHAPE = (160, 90, 5)

BATCH_SIZE = 64
EPOCHS = 50

MODEL_PATH = f"checkpoints/position_pred_model"
TRAIN_HDF5 = f"training.hdf5"
VALID_HDF5 = f"validation.hdf5"
TEST_HDF5 = f"test.hdf5"

# Given a HDF5 file, this class generates batches of data.
class HDF5DatasetGenerator:

    def __init__(self, filename, batch_size):
        self.batch_size = batch_size
        self.db = h5py.File(filename)
        self.input_size = self.db["x"].shape[0]


    def random_generator(self):
        while True:
            # random index to change order of the batches
            ix = np.random.permutation(np.arange(0, self.input_size))

            for i in np.arange(0, self.input_size, self.batch_size):
                pos = ix[i: i + self.batch_size]

                img1 = self.db["img1"][pos]
                img2 = self.db["img2"][pos]
                alt1 = self.db["alt1"][pos]
                alt2 = self.db["alt2"][pos]
                yaw = self.db["yaw"][pos]

                x = self.db["x"][pos]
                y = self.db["y"][pos]

                yield ([img1, img2, alt1, alt2, yaw], [x, y])


    def generator(self):
        while True:
            for i in np.arange(0, self.input_size, self.batch_size):
                img1 = self.db["img1"][i: i + self.batch_size]
                img2 = self.db["img2"][i: i + self.batch_size]
                alt1 = self.db["alt1"][i: i + self.batch_size]
                alt2 = self.db["alt2"][i: i + self.batch_size]
                yaw = self.db["yaw"][i: i + self.batch_size]

                x = self.db["x"][i: i + self.batch_size]
                y = self.db["y"][i: i + self.batch_size]

                yield ([img1, img2, alt1, alt2, yaw], [x, y])


    def close(self):
        self.db.close()


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
