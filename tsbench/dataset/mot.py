from sqlite3 import Timestamp
from typing import List, Dict, Any

import numpy as np

from tsbench.dataset import base, DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOTDataset(base.BaseDataset):
    def __init__(self, path: str, **__: Any) -> None:
        super().__init__()

        self.raw_data = np.loadtxt(path, delimiter=",")
        self.trajectory_data = self.convert_dataset_to_trajectories()

    def convert_dataset_to_trajectories(self) -> List[Dict]:
        ids = np.unique(self.raw_data[:,1])
        
        trajectories = []
        for idx in ids:
            trajectory = self.raw_data[self.raw_data[:, 1]==idx]
            bbox = trajectory[:, 1:5]
            bbox[:, 2:] += bbox[:, :2]
            trajectories.append({
                "tid": idx,
                "bbox": bbox,
                "timestamp": trajectory[:, :1],
            })
        return trajectories
