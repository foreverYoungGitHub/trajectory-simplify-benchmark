from typing import Dict, List, Callable
from functools import partial

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY


def local_integral_func(previous_integral, current_dists):
    return previous_integral + current_dists.sum()


def local_general_integral_func(previous_integral, current_dists, p):
    return np.linalg.norm(
        [previous_integral, np.linalg.norm(current_dists, ord=p)], ord=p
    )


def directed_acyclic_graph_search(
    trajectory, epsilon, dist_func, integral_func, search_list: List = []
):
    """Path-based range query"""
    num_points = trajectory.shape[0]
    parents = np.ones(num_points, dtype=np.int) * -1
    global_dists = np.ones(num_points) * np.inf
    global_dists[0] = 0

    # visit_status: 0 for unvisited, 1 for visit(k+1), 2 for visit(k), 3 for visited cached
    indices = np.arange(num_points)
    visit_status = np.zeros(num_points, dtype=np.int8)
    if len(search_list) > 0:
        visit_status += 3
        visit_status[search_list] = 0
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
                dist = (
                    dist_func(trajectory[start : end + 1])
                    if start + 1 < end
                    else np.zeros(1)
                )
                mask = visit_status[start + 1 : end] != 3
                max_dist = dist[mask].max() if len(mask) > 0 and np.any(mask) else 0
                # find the first acceptable distance at start
                if max_dist <= epsilon:
                    visit_status[end] = 1
                    parents[end] = start
                    global_dists[end] = integral_func(global_dists[start], dist)
                    break
                # the distance between start and end over than threshold
                elif max_dist > 2 * epsilon:
                    break
            # do not check the rest of unvisited points
            if max_dist > 2 * epsilon:
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
                dist = (
                    dist_func(trajectory[start : end + 1])
                    if start + 1 < end
                    else np.zeros(1)
                )
                mask = visit_status[start + 1 : end] != 3
                max_dist = dist[mask].max() if len(mask) > 0 and np.any(mask) else 0
                g_dist = integral_func(global_dists[start], dist)
                if max_dist <= epsilon and g_dist < global_dists[end]:
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

    def integral_func(self, previous_integral, current_dist, p):
        return local_general_integral_func(previous_integral, current_dist, p)

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        epsilon: float,
        p: float = 1,
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory,
            epsilon,
            self.dist_func,
            partial(self.integral_func, p=p),
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
        p: float = 1,
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory,
            epsilon,
            partial(self.dist_func, iou_type=iou_type),
            partial(self.integral_func, p=p),
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAGv2_Points(DAGv2):
    """Directed Acyclic Graph Based with point"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_RSEDs(trajectory)

    def simplify_one_trajectory(
        self,
        _trajectory: np.ndarray,
        epsilon: float,
        ref_ratio: float = 0,
        p: float = 1,
    ) -> np.ndarray:
        trajectory = np.copy(_trajectory)
        if ref_ratio <= 0:
            trajectory[:, 3] = 1
        else:
            trajectory[:, 3] = ref_ratio * trajectory[:, 3]

        indices = directed_acyclic_graph_search(
            trajectory,
            epsilon,
            self.dist_func,
            partial(self.integral_func, p=p),
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAGv2_2Points(DAGv2_Points):
    """Directed Acyclic Graph Based with 2 Point"""

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        epsilon: float,
        ref_ratio: float = 0,
        p: float = 1,
    ) -> np.ndarray:
        if ref_ratio <= 0:
            ref_size = np.ones((trajectory.shape[0], 1))
        else:
            wh = trajectory[:, 3:] - trajectory[:, 1:3]
            ref_size = ref_ratio * np.linalg.norm(wh, axis=1, keepdims=True)
        lt = np.concatenate((trajectory[:, :3], ref_size), axis=-1)
        indices = directed_acyclic_graph_search(
            lt,
            epsilon,
            self.dist_func,
            partial(self.integral_func, p=p),
        )
        rb = np.concatenate((trajectory[:, :1], trajectory[:, 3:], ref_size), axis=-1)
        indices += directed_acyclic_graph_search(
            rb,
            epsilon,
            self.dist_func,
            partial(self.integral_func, p=p),
        )
        indices = np.unique(indices)
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory
