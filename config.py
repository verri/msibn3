from collections import namedtuple

Params = namedtuple('Params', ['INPUT_SHAPE', 'BATCH_SIZE', 'EPOCHS', 'MAX_ALTITUDE'])

params = Params(
    INPUT_SHAPE = (90, 160, 8), # rows x cols (instead of width x height)
    BATCH_SIZE = 64,
    EPOCHS = 50,
    MAX_ALTITUDE = 100,
)
