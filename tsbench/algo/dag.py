from typing import Dict, List, Callable
from dataclasses import dataclass

import numpy as np

from tsbench.algo import base, cal_dist, ALGO_REGISTRY


def directed_acyclic_graph_search(trajectory, lower_bound, upper_bound, dist_func):
    """Path-based range query"""
    num_points = trajectory.shape[0]
    parents = np.ones(num_points, dtype=np.int) * -1
    global_dists = np.ones(num_points) * np.inf
    global_dists[0] = 0

    v_k = [0]
    v_l = []
    unvisited = list(range(1, num_points))

    while unvisited or parents[-1] == -1:
        # initial the DAG graph/tree
        # edge test in descending order.
        for start in sorted(v_k, reverse=True):
            num_unvisited = len(unvisited)
            # for loop all the unvisited nodes
            for _ in range(num_unvisited):
                end = unvisited.pop(0)
                # check with local integral distance
                dist = dist_func(trajectory[start:end])
                if dist <= lower_bound:
                    v_l.append(end)
                    parents[end] = start
                    global_dists[end] = global_dists[start] + dist
                    pass
                elif dist > upper_bound:
                    unvisited.append(end)
                    break
                else:
                    unvisited.append(end)

        # optimize the graph by minimizing global integral distance
        # minimizes the total error from root node to each element of Vl set.
        # The minimization is done by choosing the best parents of Vl elements among Vk elements.
        for end in v_l:
            for start in v_k:
                if start == parents[end]:
                    continue
                g_dist = global_dists[start] + dist_func(trajectory[start:end])
                if g_dist < global_dists[end]:
                    parents[end] = start
                    global_dists[end] = g_dist

        # swap v_l to v_k
        v_k = v_l
        v_l = []

    # decode the ind
    indices = [num_points - 1]
    while parents[indices[-1]] != -1:
        indices.append(parents[indices[-1]])

    # reverse output indices
    return indices[::-1]


@ALGO_REGISTRY.register()
class DAG(base.BaseTS):
    """Directed Acyclic Graph Based"""

    def dist_func(self, trajectory):
        return cal_dist.cacl_LISSED(trajectory)

    def simplify_one_trajectory(
        self, trajectory: np.ndarray, lower_bound: float, upper_bound: float
    ) -> np.ndarray:
        indices = directed_acyclic_graph_search(
            trajectory, lower_bound, upper_bound, self.dist_func
        )
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory
