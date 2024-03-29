from typing import Dict, List, Callable
from functools import partial

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY


def local_average_func(previous_integral, current_dist, start, end):
    return previous_integral + current_dist * (end - start - 1)


def local_integral_func(previous_integral, current_dist, start, end):
    return previous_integral + current_dist


def local_general_integral_func(previous_integral, current_dist, start, end, p):
    return np.linalg.norm((previous_integral, current_dist), ord=p)


def directed_acyclic_graph_search(
    trajectory, lower_bound, upper_bound, dist_func, integral_func
):
    """Path-based range query"""
    if False:
        res = local_dist_analysis(trajectory, dist_func, integral_func)
        import matplotlib.pyplot as plt

        x = np.arange(res.shape[0])
        plt.plot(x, res[:, 0], label="line 1")
        plt.plot(x, res[:, 1], label="line 2")
        plt.plot(x, res[:, 2], label="curve 1")
        plt.legend()
        plt.show()
        assert False

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
                dist = dist_func(trajectory[start : end + 1])
                # find the first acceptable distance at start
                if dist <= lower_bound:
                    visit_status[end] = 1
                    parents[end] = start
                    global_dists[end] = integral_func(
                        global_dists[start], dist, start, end
                    )
                    break
                # the distance between start and end over than threshold
                elif dist > upper_bound:
                    break
            # do not check the rest of unvisited points
            if dist > upper_bound:
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
                dist = dist_func(trajectory[start : end + 1])
                g_dist = integral_func(global_dists[start], dist, start, end)
                if dist <= lower_bound and g_dist < global_dists[end]:
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


def local_dist_analysis(trajectory, dist_func, integral_func):
    num_points = trajectory.shape[0]
    start = 0
    local_start = [
        dist_func(trajectory[start : end + 1]) for end in range(1, num_points)
    ]
    end = num_points - 1
    local_end = [
        dist_func(trajectory[start : end + 1]) for start in range(0, num_points - 1)
    ]
    global_dist = [
        integral_func(local_start[start], local_end[start], start + 1, num_points - 1)
        for start in range(0, num_points - 1)
    ]
    return np.array([local_start, local_end, global_dist]).T


@ALGO_REGISTRY.register()
class DAG(base.BaseTS):
    """Directed Acyclic Graph Based"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_LISSED(trajectory)

    def integral_func(self, previous_integral, current_dist, start, end):
        return local_integral_func(previous_integral, current_dist, start, end)

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, lower_bound: float, upper_bound: float
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory, lower_bound, upper_bound, self.dist_func, self.integral_func
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAG_IOU(DAG):
    """Directed Acyclic Graph Based"""

    def dist_func(self, trajectory, iou_type):
        return cal_dist.cacl_LISSIOUs(trajectory, iou_type)

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        iou_type: str = "iou",
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory,
            lower_bound,
            upper_bound,
            partial(self.dist_func, iou_type=iou_type),
            self.integral_func,
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAG_IOUv2(DAG):
    """Directed Acyclic Graph Based"""

    def dist_func(self, trajectory, p, iou_type):
        return cal_dist.cacl_GILSIOUs(trajectory, p, iou_type)

    def integral_func(self, previous_integral, current_dist, start, end, p):
        return local_general_integral_func(
            previous_integral, current_dist, start, end, p
        )

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        p: float = 1,
        iou_type: str = "iou",
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory,
            lower_bound,
            upper_bound,
            partial(self.dist_func, iou_type=iou_type, p=p),
            partial(self.integral_func, p=p),
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAG_2Points(DAG):
    """Directed Acyclic Graph Based with 2 Point"""

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        lower_bound: float,
        upper_bound: float,
    ) -> np.ndarray:
        lt = trajectory[:, :3]
        indices = directed_acyclic_graph_search(
            lt, lower_bound, upper_bound, self.dist_func, self.integral_func
        )
        rb = np.concatenate((trajectory[:, :1], trajectory[:, 3:]), axis=-1)
        indices += directed_acyclic_graph_search(
            rb, lower_bound, upper_bound, self.dist_func, self.integral_func
        )
        indices = np.unique(indices)
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory


@ALGO_REGISTRY.register()
class DAG_Points(DAG):
    """Directed Acyclic Graph Based with point"""

    def dist_func(self, trajectory, p):
        return cal_dist.cacl_GILRSED(trajectory, p)

    def integral_func(self, previous_integral, current_dist, start, end, p):
        return local_general_integral_func(
            previous_integral, current_dist, start, end, p
        )

    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        p: float = 2,
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory,
            lower_bound,
            upper_bound,
            partial(self.dist_func, p=p),
            partial(self.integral_func, p=p),
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory
