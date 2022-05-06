from sqlite3 import Timestamp
from typing import List, Dict, Any

import numpy as np

from tsbench.dataset import base, DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SampleDataset(base.BaseDataset):
    def __init__(self, path: str, **__: Any) -> None:
        super().__init__()

        self.raw_data = np.loadtxt(path)
        self.trajectory_data = self.convert_dataset_to_trajectories()

    def convert_dataset_to_trajectories(self) -> List[Dict]:
        return [
            {
                "tid": 0,
                "trajectory": self.raw_data[:, :2],
                "timestamp": self.raw_data[:, 2:],
            }
        ]
