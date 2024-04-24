from collections import namedtuple

Params = namedtuple('Params', ['INPUT_SHAPE', 'BATCH_SIZE', 'EPOCHS', 'MAX_ALTITUDE'])

params = Params(
    INPUT_SHAPE = (90, 160, 8), # rows x cols (instead of width x height)
    BATCH_SIZE = 64,
    MAX_ALTITUDE = 100,
    STEPS_PER_EPOCH = 1000,
    EPOCHS = 50,
)
