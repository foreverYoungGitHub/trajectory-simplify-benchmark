from typing import List, Dict, Any
from pathlib import Path

import numpy as np

from tsbench.dataset import base, DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SampleDataset(base.BaseDataset):
    def __init__(self, path: str, **__: Any) -> None:
        super().__init__(path)

        self.raw_data = np.loadtxt(self.path)
        self.trajectory_data = self.convert_dataset_to_trajectories()

    def convert_dataset_to_trajectories(self) -> List[Dict]:
        return [
            {
                "tid": 0,
                "trajectory": self.raw_data[:, :2],
                "timestamp": self.raw_data[:, 2:],
            }
        ]

    def dump_data(self, output_dir: Path, trajectories: np.ndarray) -> None:
        trajectories = trajectories[0][:, [1, 2, 0]]
        np.savetxt(output_dir / "sample.txt", trajectories, fmt="%.7f", delimiter=" ")
