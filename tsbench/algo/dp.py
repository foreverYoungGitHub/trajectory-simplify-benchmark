from typing import List, Callable
from functools import partial

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY


def recursive_simplify(
    trajectory: np.ndarray,
    epsilon: float,
    dist_func: Callable,
    start: int,
    end: int,
) -> List[int]:
    assert start <= end, (start, end)
    if start + 1 >= end:
        return [start, end]
    indices = []
    dists = dist_func(trajectory[start : end + 1])
    peak_index = dists.argmax()
    if dists[peak_index] > epsilon:
        peak_index += start + 1
        indices += recursive_simplify(
            trajectory,
            epsilon,
            dist_func,
            start,
            peak_index,
        )
        indices += recursive_simplify(
            trajectory,
            epsilon,
            dist_func,
            peak_index,
            end,
        )
    else:
        indices += [start, end]
    return indices


@ALGO_REGISTRY.register()
class DP(base.BaseTS):
    """DouglasPeucker"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_PEDs(trajectory)

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, epsilon: float
    ) -> np.ndarray:
        indices = np.unique(
            recursive_simplify(
                trajectory, epsilon, self.dist_func, 0, len(trajectory) - 1
            )
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class TDTR(DP):
    """Top Down Time Ratio"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_SEDs(trajectory)


@ALGO_REGISTRY.register()
class TDTR_IOU(TDTR):
    """Top Down Time Ratio with IOU distance"""

    def dist_func(self, trajectory, iou_type):
        return cal_dist.cacl_SIOUs(trajectory, iou_type)

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, epsilon: float, iou_type: str = "iou"
    ) -> np.ndarray:
        indices = np.unique(
            recursive_simplify(
                trajectory,
                epsilon,
                partial(self.dist_func, iou_type=iou_type),
                0,
                len(trajectory) - 1,
            )
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class TDTR_Points(TDTR):
    """Top Down Time Ratio with IOU distance"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_RSEDs(trajectory)

    def simplify_one_trajectory(
        self, _trajectory: np.ndarray, epsilon: float, ref_ratio: float = 0
    ) -> np.ndarray:
        trajectory = np.copy(_trajectory)
        if ref_ratio <= 0:
            trajectory[:, 3] = 1
        else:
            trajectory[:, 3] = ref_ratio * trajectory[:, 3]

        indices = np.unique(
            recursive_simplify(
                trajectory,
                epsilon,
                partial(self.dist_func),
                0,
                len(trajectory) - 1,
            )
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class TDTR_2Points(TDTR_Points):
    """Top Down Time Ratio with IOU distance"""

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, epsilon: float, ref_ratio: float = 0
    ) -> np.ndarray:
        if ref_ratio <= 0:
            ref_size = np.ones((trajectory.shape[0], 1))
        else:
            wh = trajectory[:, 3:] - trajectory[:, 1:3]
            ref_size = ref_ratio * np.linalg.norm(wh, axis=1, keepdims=True)
        lt = np.concatenate((trajectory[:, :3], ref_size), axis=-1)
        indices = recursive_simplify(
            lt, epsilon, self.dist_func, 0, len(trajectory) - 1
        )
        rb = np.concatenate((trajectory[:, :1], trajectory[:, 3:], ref_size), axis=-1)
        indices += recursive_simplify(
            rb, epsilon, self.dist_func, 0, len(trajectory) - 1
        )
        indices = np.unique(indices)
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory
