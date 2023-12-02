import h5py
import numpy as np
from typing import NamedTuple

from geographiclib.geodesic import Geodesic
import math


class Frame(NamedTuple):
    data: np.ndarray
    time: int

    yaw: float
    """Yaw angle of the gimbal (degrees).
    0 represents north and increases eastward.
    """

    altitude: float


class FlightSegment(NamedTuple):
    frames: list[Frame]

    distance: float
    """Distance between the coordinates of the first and last frame.

    Algorithms from:
        C. F. F. Karney, Algorithms for geodesics, J. Geodesy 87(1), 43–55 (2013); Addenda.
        https://link.springer.com/article/10.1007/s00190-012-0578-z
    """

    bearing: float
    """Angle of displacement between the coordinates of the first and last frame."""


def frame_from_dataset(dataset: h5py.Dataset) -> Frame:
    """Retrieves a frame from a HDF5 dataset.

    Args:
        dataset: h5py.Dataset
            Dataset object.

    Returns:
        Frame
            Frame object.
    """

    return Frame(
        data=dataset[...],
        time=dataset.attrs['time'],
        yaw=dataset.attrs['gimbal_yaw'],
        altitude=dataset.attrs['ground_level_altitude']
    )


class FlightSimulator:

    def __init__(self,
                 h5filename: str,
                 delta_time: tuple[float, float] = (500, 100),
                 max_time: float = 1000):
        """
        Args:
            h5filename: str
                Name of the HDF5 file to read from.
            delta_time: tuple[float, float]
                Tuple of mean and std time between frames.
            max_time: float
                Maximum time to wait for the next frame.
                If no frame is avaliable after this time, a new flight segment
                is started.
        """

        self.h5filename = h5filename
        self.delta_time = delta_time
        self.max_time = max_time
        self.datasets = []


    def _visitor(self, name, obj):
        """
        Args:
            name: str
                Name of the dataset.
            obj: h5py.Dataset
                Dataset object.
        """
        if isinstance(obj, h5py.Dataset):
            self.datasets.append((name, obj))


    def generator(self, seed=None):

        rng = np.random.default_rng(seed)
        geod = Geodesic.WGS84

        with h5py.File(self.h5filename, 'r') as f:

            # Traverse all datasets available in the HDF5 file (alphabetical order)
            f.visititems(self._visitor)

            current = 0
            while True:

                if current >= len(self.datasets):
                    return None

                name, dataset = self.datasets[current]
                # print(f'Current dataset: {name} ({dataset.attrs["time"]})')

                next_time = dataset.attrs['time'] + rng.normal(*self.delta_time)
                limit_time = dataset.attrs['time'] + self.max_time

                # print(f'\tNext time must be between {next_time} and {limit_time}')

                after = current + 1
                while after < len(self.datasets):

                    _, next_dataset = self.datasets[after]
                    # print(f'\tNext dataset: {next_dataset.name} ({next_dataset.attrs["time"]})')

                    if next_dataset.attrs['time'] >= next_time and next_dataset.attrs['time'] < limit_time:
                        break

                    if dataset.attrs['video_filename'] != next_dataset.attrs['video_filename']:
                        current = after
                        break

                    if next_dataset.attrs['time'] >= limit_time:
                        current = after
                        break

                    after += 1

                if current == after:
                    continue

                if after >= len(self.datasets):
                    return None

                # Calculate the distance between coordinates of the datasets.
                # Also calculate the angle of displacement (magnetic north).

                lat1 = dataset.attrs['latitude']
                lon1 = dataset.attrs['longitude']
                lat2 = next_dataset.attrs['latitude']
                lon2 = next_dataset.attrs['longitude']

                g = geod.Inverse(lat1, lon1, lat2, lon2)
                distance = g['s12']
                bearing = g['azi1']

                # Must correct the bearing angle to match the convention used by the gimbal.
                bearing = bearing if bearing > 0 else 360 + bearing

                yield FlightSegment(
                    frames=[frame_from_dataset(dataset), frame_from_dataset(next_dataset)],
                    distance=distance,
                    bearing=bearing,
                )
                current += 1
