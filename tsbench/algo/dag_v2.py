from typing import Dict, List, Callable
from functools import partial

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY

def directed_acyclic_graph_search(
    trajectory, epsilon, dist_func
):
    """Path-based range query"""
    num_points = trajectory.shape[0]
    parents = np.ones(num_points, dtype=np.int) * -1
    global_dists = np.ones(num_points) * np.inf
    global_dists[0] = 0

    # visit_status: 0 for unvisited, 1 for visit(k+1), 2 for visit(k), 3 for visited cached
    indices = np.arange(num_points)
    visit_status = np.zeros(num_points, dtype=np.int8)
    visit_status[0] = 2

    while visit_status[-1] == 0:
        # initial the DAG graph/tree
        unvisited_index = indices[visit_status == 0]
        parent_index = indices[visit_status == 2]
        # for loop all the unvisited nodes
        for end in unvisited_index:
            # edge test in descending order.
            for start in parent_index[::-1]:
                # check with local integral distance
                dist = dist_func(trajectory[start : end + 1]) if start + 1 < end else np.zeros(1)
                # find the first acceptable distance at start
                if dist.max() <= epsilon:
                    visit_status[end] = 1
                    parents[end] = start
                    global_dists[end] = global_dists[start] + dist.sum()
                    break
                # the distance between start and end over than threshold
                elif dist.max() > epsilon:
                    break
            # do not check the rest of unvisited points
            if dist.max() > epsilon:
                break

        # optimize the graph by minimizing global integral distance
        # minimizes the total error from root node to each element of childs set.
        # The minimization is done by choosing the best parents of childs elements among all parents elements.
        childs_index = indices[visit_status == 1]
        for end in childs_index:
            for start in parent_index:
                # start >= parents[end] has already be calculated previously and
                # The dist of all start > parents[end] overs than the lower bound
                if start == parents[end]:
                    break
                dist = dist_func(trajectory[start : end + 1]) if start + 1 < end else np.zeros(1)
                g_dist = global_dists[start] + dist.sum()
                if dist.max() <= epsilon and g_dist < global_dists[end]:
                    parents[end] = start
                    global_dists[end] = g_dist

        # swap childs_index to parent_index
        visit_status[parent_index] += 1
        visit_status[childs_index] += 1

    # decode the ind
    indices = [num_points - 1]
    while parents[indices[-1]] != -1:
        indices.append(parents[indices[-1]])

    # reverse output indices
    return indices[::-1]

@ALGO_REGISTRY.register()
class DAGv2(base.BaseTS):
    """Directed Acyclic Graph Based"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_SEDs(trajectory)

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, epsilon: float,
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory, epsilon, self.dist_func,
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAGv2_IOU(DAGv2):
    """Directed Acyclic Graph Based"""

    def dist_func(self, trajectory, iou_type):
        return cal_dist.cacl_SIOUs(trajectory, iou_type)

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        epsilon: float,
        iou_type: str = "iou",
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory,
            epsilon,
            partial(self.dist_func, iou_type=iou_type),
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


# @ALGO_REGISTRY.register()
# class DAG_IOUv2(DAG):
#     """Directed Acyclic Graph Based"""

#     def dist_func(self, trajectory, p, iou_type):
#         return cal_dist.cacl_GILSIOUs(trajectory, p, iou_type)

#     def integral_func(self, previous_integral, current_dist, start, end, p):
#         return local_general_integral_func(previous_integral, current_dist, start, end, p)

#     def simplify_one_trajectory(
#         self,
#         trajectory: np.ndarray,
#         lower_bound: float,
#         upper_bound: float,
#         p: float = 1,
#         iou_type: str = "iou",
#     ) -> np.ndarray:
#         indices = directed_acyclic_graph_search(
#             trajectory,
#             lower_bound,
#             upper_bound,
#             partial(self.dist_func, iou_type=iou_type, p=p),
#             partial(self.integral_func, p=p),
#         )
#         simplified_trajectory = trajectory[indices]
#         return simplified_trajectory

# @ALGO_REGISTRY.register()
# class DAG_2Points(DAG):
#     """Directed Acyclic Graph Based with 2 Point"""

#     def simplify_one_trajectory(
#         self,
#         trajectory: np.ndarray,
#         lower_bound: float,
#         upper_bound: float,
#     ) -> np.ndarray:
#         lt = trajectory[:, :3]
#         indices = directed_acyclic_graph_search(
#             lt, lower_bound, upper_bound, self.dist_func, self.integral_func
#         )
#         rb = np.concatenate((trajectory[:, :1], trajectory[:, 3:]), axis=-1)
#         indices += directed_acyclic_graph_search(
#             rb, lower_bound, upper_bound, self.dist_func, self.integral_func
#         )
#         indices = np.unique(indices)
#         simplified_trajectory = trajectory[indices]
#         return simplified_trajectory

# @ALGO_REGISTRY.register()
# class DAG_Points(DAG):
#     """Directed Acyclic Graph Based with point"""

#     def dist_func(self, trajectory, p):
#         return cal_dist.cacl_GILRSED(trajectory, p)

#     def integral_func(self, previous_integral, current_dist, start, end, p):
#         return local_general_integral_func(previous_integral, current_dist, start, end, p)

#     def simplify_one_trajectory(
#         self,
#         trajectory: np.ndarray,
#         lower_bound: float,
#         upper_bound: float,
#         p: float = 2,
#     ) -> np.ndarray:
#         indices = directed_acyclic_graph_search(
#             trajectory,
#             lower_bound,
#             upper_bound,
#             partial(self.dist_func, p=p),
#             partial(self.integral_func, p=p),
#         )
#         simplified_trajectory = trajectory[indices]
#         return simplified_trajectory
