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
    return np.linalg.norm(trajectory_1[:, 1:] - trajectory_2[:, 1:], axis=-1)


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
    for tid, traj in trajectories.items():
        simplified_traj = simplified_trajectories[tid]
        interp_traj = interpolate_trajectory(simplified_traj, traj[:, 0])
        if traj.shape[1] == 3:
            dist = trajectorys_l2_distance(traj, interp_traj)
        elif traj.shape[1] == 5:
            dist = 1 - cal_dist.ious(traj[:, 1:], interp_traj[:, 1:])
        metrics["ratio"][tid] = simplified_traj.shape[0] / traj.shape[0]
        metrics["mean"][tid] = dist.mean()
        metrics["max"][tid] = dist.max()
    return metrics
