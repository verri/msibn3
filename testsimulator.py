from datasets.hdf5 import FlightSimulator

simulator = FlightSimulator("pqtec-20231128.hdf5")

rng = np.random.default_rng()
for segment in simulator.generate(rng):
    print(segment.distance, segment.bearing)
