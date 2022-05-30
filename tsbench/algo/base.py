from time import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import tqdm
import numpy as np
from joblib import Parallel, delayed


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

        # parallel
        def parallel_loop(tid, traj):
            ts = time()
            simplified_trajectory = self.simplify_one_trajectory(traj, **params)
            query_runtime = time() - ts
            return tid, simplified_trajectory, query_runtime

        results = Parallel(n_jobs=4)(
            delayed(parallel_loop)(tid, traj)
            for tid, traj in tqdm.tqdm(trajectories.items())
        )
        for res in results:
            simplified_trajectories[res[0]] = res[1]
            runtime_per_query[res[0]] = res[2]

        # sequential
        # for tid, traj in tqdm.tqdm(trajectories.items()):
        #     assert np.all(
        #         traj[:-1, 0] <= traj[1:, 0]
        #     ), "The trajectory is not sorted by time"
        #     ts = time()
        #     simplified_trajectories[tid] = self.simplify_one_trajectory(traj, **params)
        #     runtime_per_query[tid] = time() - ts
        return simplified_trajectories, runtime_per_query

    @abstractmethod
    def simplify_one_trajectory(self, trajectory: np.ndarray, **params) -> np.ndarray:
        """Override this method to simplify the trajectory."""
