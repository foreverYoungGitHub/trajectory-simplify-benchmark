from typing import Dict, List, Callable
from functools import partial

import numpy as np

from tsbench.algo import base, cal_dist, dag_v2, dp, ALGO_REGISTRY


@ALGO_REGISTRY.register()
class OCDAG(base.BaseTS):
    """Observed Centered Directed Acyclic Graph Based"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_SEDs(trajectory)

    def integral_func(self, previous_integral, current_dist, p):
        return dag_v2.local_general_integral_func(previous_integral, current_dist, p)

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        high_conf_thresh: float,
        epsilon_1: float,
        epsilon_2: float,
        p: float = 1,
    ) -> np.ndarray:
        indices, _ = np.where(trajectory[:, 3] > high_conf_thresh)
        indices = np.unique(np.append(indices, 0, len(trajectory) - 1))

        search_space = []
        for i in range(len(indices) - 1):
            search_space += dp.recursive_simplify(
                trajectory, epsilon_1, self.dist_func, indices[i], indices[i] + 1
            )
        search_space = np.unique(search_space)

        indices = dag_v2.directed_acyclic_graph_search(
            trajectory,
            epsilon_2,
            self.dist_func,
            partial(self.integral_func, p=p),
            search_space,
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class OCDAG_IOU(OCDAG):
    """Observed Centered Directed Acyclic Graph Based"""

    def dist_func(self, trajectory, iou_type):
        return cal_dist.cacl_WSIOUs(trajectory, iou_type)

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        epsilon_1: float,
        epsilon_2: float,
        high_conf_thresh: float = 0,
        high_conf_precentile: int = 0,
        iou_type: str = "iou",
        p: float = 1,
    ) -> np.ndarray:
        high_conf_thresh_precentile = np.percentile(trajectory[:, 5], high_conf_precentile)
        high_conf_thresh = max(high_conf_thresh_precentile, high_conf_thresh)
        (indices,) = np.where(trajectory[:, 5] >= high_conf_thresh)
        indices = np.unique([*indices, 0, len(trajectory) - 1])
        # print(f"high conf: {len(indices)}/{len(trajectory)}")
        # print(indices)

        search_space = []
        for i in range(len(indices) - 1):
            search_space += dp.recursive_simplify(
                trajectory,
                epsilon_1,
                partial(self.dist_func, iou_type=iou_type),
                indices[i],
                indices[i+1],
            )
        search_space = np.unique(search_space)
        # print(f"search space: {len(search_space)}/{len(trajectory)}")
        # print(search_space)


        indices = dag_v2.directed_acyclic_graph_search(
            trajectory,
            epsilon_2,
            partial(self.dist_func, iou_type=iou_type),
            partial(self.integral_func, p=p),
            search_space,
        )
        # print(f"final: {len(indices)}/{len(trajectory)}")
        # assert False

        simplified_trajectory = trajectory[indices, :5]
        return simplified_trajectory
