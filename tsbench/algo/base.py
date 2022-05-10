from time import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import tqdm
import numpy as np


class BaseTS(ABC):
    def simplify(
        self, trajectories: Dict[int, np.ndarray], **params
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """simplify trajectories, call the simplify_one_trajectory()
        to do the actually simplification

        Args:
            trajectories (Dict[int, np.ndarray]): The full continous
                trajectories

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, float]]:
                simplified trajectories, runtime per query
        """
        simplified_trajectories = {}
        runtime_per_query = {}
        for tid, traj in tqdm.tqdm(trajectories.items()):
            assert np.all(
                traj[:-1, 0] <= traj[1:, 0]
            ), "The trajectory is not sorted by time"
            ts = time()
            simplified_trajectory = self.simplify_one_trajectory(traj, **params)
            runtime_per_query[tid] = time() - ts
            simplified_trajectories[tid] = simplified_trajectory
        return simplified_trajectories, runtime_per_query

    @abstractmethod
    def simplify_one_trajectory(self, trajectory: np.ndarray, **params) -> np.ndarray:
        """Override this method to simplify the trajectory."""
