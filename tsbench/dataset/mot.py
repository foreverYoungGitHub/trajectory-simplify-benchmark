from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from tsbench.dataset import base, DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOTDataset(base.BaseDataset):
    def __init__(
        self, path: str, max_trajectories_per_file: int = 10000, **__: Any
    ) -> None:
        super().__init__(path)
        self.max_trajectories_per_file = max_trajectories_per_file
        self.trajectory_data, self.filenames = self.convert_dataset_to_trajectories()

    def convert_dataset_to_trajectories(self) -> List[Dict]:
        trajectories = []
        filenames = []
        for f_index, txt_file in enumerate(self.path.glob("*.txt")):
            filenames.append(txt_file.name)

            raw_data = np.loadtxt(txt_file, delimiter=",")
            ids = np.unique(raw_data[:, 1])
            for idx in ids:
                trajectory = raw_data[raw_data[:, 1] == idx]
                bbox = trajectory[:, 2:6]
                bbox[:, 2:] += bbox[:, :2]
                trajectories.append(
                    {
                        "tid": f_index * self.max_trajectories_per_file + idx,
                        "bbox": bbox,
                        "timestamp": trajectory[:, :1],
                    }
                )
        return trajectories, filenames

    def dump_data(self, output_dir: Path, trajectories: np.ndarray) -> None:
        all_trajectories = []
        for d in self.trajectory_data:
            trajectory = trajectories[d["tid"]]
            tid = np.ones(trajectory.shape[0]) * d["tid"]
            trajectory = np.concatenate((tid[:, None], trajectory), axis=1)
            all_trajectories.append(trajectory)
        all_trajectories = np.concatenate(all_trajectories, axis=0)
        all_trajectories[:, [0, 1]] = all_trajectories[:, [1, 0]]
        all_trajectories[:, 4:] -= all_trajectories[:, 2:4]

        for f_index, filename in enumerate(self.filenames):
            trajectories = all_trajectories[
                (all_trajectories[:, 1] // self.max_trajectories_per_file) == f_index
            ]
            trajectories[:, 1] = trajectories[:, 1] % self.max_trajectories_per_file
            np.savetxt(
                output_dir / filename, trajectories.astype(int), fmt="%i", delimiter=","
            )
