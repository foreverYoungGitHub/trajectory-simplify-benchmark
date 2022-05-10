from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pathlib

import numpy as np


class BaseDataset(ABC):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = pathlib.Path(path)

    @abstractmethod
    def convert_dataset_to_trajectories(self, *_: Any, **__: Any) -> List[Dict]:
        """Override this method to update trajectory data"""

    def get_trajectories(self, key: str) -> Dict[int, np.ndarray]:
        """get trajectories by key

        Args:
            key (str): the key to fetch the relative data

        Returns:
            Dict[int, np.ndarray]: trajectories with trajectory id
                and trajectory_data
        """
        return {
            d["tid"]: np.concatenate((d["timestamp"], d[key]), axis=-1)
            for d in self.trajectory_data
        }
