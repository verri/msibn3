from collections import namedtuple

Params = namedtuple('Params', ['INPUT_SHAPE', 'BATCH_SIZE', 'MAX_ALTITUDE',
    'STEPS_PER_EPOCH', 'EPOCHS'])

params = Params(
    INPUT_SHAPE = (90, 160, 8), # rows x cols (instead of width x height)
    BATCH_SIZE = 64,
    MAX_ALTITUDE = 100,
    STEPS_PER_EPOCH = 200, # approximately 10 minutes (each step is a batch) of data
    EPOCHS = 100,
)
