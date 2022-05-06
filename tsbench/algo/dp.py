from typing import List, Callable

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY

@ALGO_REGISTRY.register()
class DouglasPeucker(base.BaseTS):
    def dist_func(self, trajectory):
        return cal_dist.cacl_PEDs(trajectory)

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, epsilon: float
    ) -> np.ndarray:
        def recursive_simplify(start: int, end: int) -> List[int]:
            assert start < end, (start, end)
            if start + 1 == end:
                return [start, end]
            indices = []
            dists = self.dist_func(trajectory[start : end + 1])
            peak_index = dists.argmax()
            if dists[peak_index] > epsilon:
                peak_index += start + 1
                indices += recursive_simplify(start, peak_index)
                indices += recursive_simplify(peak_index, end)
            else:
                indices += [start, end]
            return indices

        indices = np.unique(recursive_simplify(0, len(trajectory) - 1))
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class TDTR(DouglasPeucker):
    def dist_func(self, trajectory):
        return cal_dist.cacl_SEDs(trajectory)


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
