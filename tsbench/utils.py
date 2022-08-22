from collections import defaultdict

import numpy as np
from scipy import interpolate

from tsbench.algo import cal_dist


def interpolate_trajectory(simplified_trajectory, timestamps):
    if np.all(simplified_trajectory[:, 0] == timestamps):
        return simplified_trajectory
    n_interp = []
    for i in range(1, simplified_trajectory.shape[1]):
        t = simplified_trajectory[:, 0]
        x = simplified_trajectory[:, i]
        f_x = interpolate.interp1d(t, x, kind="linear")
        x_interp = f_x(timestamps)
        n_interp.append(x_interp)
    return np.stack((timestamps, *n_interp), axis=-1)


def interpolate_trajectories(trajectories, simplified_trajectories):
    return {
        tid: interpolate_trajectory(simplified_trajectories[tid], traj[:, 0])
        for tid, traj in trajectories.items()
    }


def trajectorys_l2_distance(trajectory_1, trajectory_2):
    return np.linalg.norm(trajectory_1[:, 1:3] - trajectory_2[:, 1:3], axis=-1)


def evaluate_trajectory(trajectory, simplified_trajectory):
    trajectory_interp = interpolate_trajectory(simplified_trajectory, trajectory[:, 0])
    dist = trajectorys_l2_distance(trajectory, trajectory_interp)
    return {
        "ratio": simplified_trajectory.shape[0] / trajectory.shape[0],
        "mean": dist.mean(),
        # "median": dist.median(),
        "max": dist.max(),
    }


def evaluate_trajectories(trajectories, simplified_trajectories):
    metrics = defaultdict(dict)
    num_nodes = 0
    num_simplified_nodes = 0
    for tid, traj in trajectories.items():
        simplified_traj = simplified_trajectories[tid]
        interp_traj = interpolate_trajectory(simplified_traj, traj[:, 0])
        assert traj.shape[1] in [3, 4, 5], traj.shape[1]
        if traj.shape[1] == 3:
            dist = trajectorys_l2_distance(traj, interp_traj)
        elif traj.shape[1] == 4:
            dist = trajectorys_l2_distance(
                traj[:, :3], interp_traj[:, :3]
            )  # / traj[:,3]
        elif traj.shape[1] == 5:
            dist = 1 - cal_dist.ious(traj[:, 1:], interp_traj[:, 1:])
        metrics["ratio"][tid] = traj.shape[0] / simplified_traj.shape[0]
        metrics["mean"][tid] = dist.mean()
        metrics["max"][tid] = dist.max()
        num_nodes += traj.shape[0]
        num_simplified_nodes += simplified_traj.shape[0]
    # metrics["total_ratio"] = {0: num_nodes / num_simplified_nodes}
    return metrics
