from datasets.hdf5 import FlightSimulator, polar2cartesian, Frame
import numpy as np
import h5py

import argparse

parser = argparse.ArgumentParser(description="Simulate a flight path")
parser.add_argument("filename", type=str, help="The filename of the flight path")
args = parser.parse_args()

file = h5py.File(args.filename, "r")
simulator = FlightSimulator([file])

def info(frame: Frame):
    return f"(time={frame.time}, yaw={frame.yaw}, altitude={frame.altitude})"

rng = np.random.default_rng()
for segment in simulator.generate(rng):
    print(*[info(frame) for frame in segment.frames], segment.distance,
            segment.bearing, *polar2cartesian((segment.distance, segment.bearing)))
