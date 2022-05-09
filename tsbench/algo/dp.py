from typing import List, Callable

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY


def recursive_simplify(
        trajectory: np.ndarray,
        epsilon: float,
        dist_func: Callable,
        start: int,
        end: int,
    ) -> List[int]:
    assert start < end, (start, end)
    if start + 1 == end:
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

    def dist_func(self, trajectory):
        return cal_dist.cacl_SIOUs(trajectory)


@ALGO_REGISTRY.register()
class TDTR_2Points(TDTR):
    """Top Down Time Ratio with IOU distance"""

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, epsilon: float
    ) -> np.ndarray:
        lt = trajectory[:, :3]
        indices = recursive_simplify(
            lt, epsilon, self.dist_func, 0, len(trajectory) - 1
        )
        rb = np.concatenate((trajectory[:, :1], trajectory[:, 3:]), axis=-1)
        indices += recursive_simplify(
            rb, epsilon, self.dist_func, 0, len(trajectory) - 1
        )
        indices = np.unique(indices)
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


# def douglas_peucker(trajectory: np.ndarray, epsilon: float, dist_func: Callable) -> np.ndarray:
#     def simplify(start: int, end: int) -> List[int]:
#         assert start < end, (start, end)
#         if start + 1 == end:
#             return [start, end]
#         indices = []
#         dists = dist_func(trajectory[start : end + 1])
#         peak_index = dists.argmax()
#         if dists[peak_index] > epsilon:
#             peak_index += start + 1
#             indices += simplify(start, peak_index)
#             indices += simplify(peak_index, end)
#         else:
#             indices += [start, end]
#         return indices

#     indices = np.unique(simplify(0, len(trajectory) - 1))
#     simplified_trajectory = trajectory[indices]
#     return simplified_trajectory

# trajectory = np.loadtxt("test.csv")
# trajectory[:, [2, 0]] = trajectory[:, [0, 2]]

# simplified_trajectory = douglas_peucker(trajectory, 0.0001)
# print(simplified_trajectory.shape)

# import matplotlib.pyplot as plt

# plt.plot(trajectory[:, 1], trajectory[:, 2], linestyle='dashed', color="g")
# plt.plot(simplified_trajectory[:, 1], simplified_trajectory[:, 2], linestyle='solid', color="r")
# plt.show()

# from tsbench import evaluation

# result = evaluation.evaluate_trajectorys(trajectory, simplified_trajectory)
# print(result)
