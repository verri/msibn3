from datasets.hdf5 import FlightSimulator

simulator = FlightSimulator("pqtec-20231128.hdf5")

for frame in simulator.generator():
    print(frame.distance, frame.bearing)
