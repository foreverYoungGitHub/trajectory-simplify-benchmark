# from typing import List

# import numpy as np

# def cacl_SEDs(trajectory):
#     time_ratio = (trajectory[1:-1, :1] - trajectory[:1, :1]) / (trajectory[-1:, :1] - trajectory[:1, :1] + 1e-10)
#     # print(time_ratio)
#     p_start = trajectory[:1, 1:]
#     p_end = trajectory[-1:, 1:]
#     points = trajectory[1:-1, 1:]
#     estimate_points = p_start * (1 - time_ratio) + p_end * time_ratio
#     d = np.linalg.norm(points - estimate_points, axis=-1)
#     return d

# def tdtr(trajectory, epsilon):
#     def simplify(start: int, end: int) -> List[int]:
#         assert start < end, (start, end)
#         if start + 1 == end:
#             return [start, end]
#         indices = []
#         dists = cacl_SEDs(trajectory[start : end + 1])
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

# simplified_trajectory = tdtr(trajectory, 0.0001)
# print(simplified_trajectory.shape)

# # import matplotlib.pyplot as plt

# # plt.plot(trajectory[:, 1], trajectory[:, 2], linestyle='dashed', color="g")
# # plt.plot(simplified_trajectory[:, 1], simplified_trajectory[:, 2], linestyle='solid', color="r")
# # plt.show()

# from tsbench import utils

# result = evaluation.evaluate_trajectorys(trajectory, simplified_trajectory)
# print(result)
