from datasets.hdf5 import FlightSimulator, polar2cartesian
import numpy as np
import h5py

file = h5py.File("pqtec-20231128.hdf5", "r")
simulator = FlightSimulator(file)

rng = np.random.default_rng()
for segment in simulator.generate(rng):
    print(*polar2cartesian((segment.distance, segment.bearing)))
