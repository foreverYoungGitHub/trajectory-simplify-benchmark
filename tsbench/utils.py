from collections import defaultdict

import numpy as np
from scipy import interpolate

def interpolate_trajectory(simplified_trajectory, timestamps):
    t = simplified_trajectory[:,0]
    x = simplified_trajectory[:,1]
    y = simplified_trajectory[:,2]
    f_x =interpolate.interp1d(t,x,kind='linear')
    f_y =interpolate.interp1d(t,y,kind='linear')
    x_interp = f_x(timestamps)
    y_interp = f_y(timestamps)
    return np.stack((timestamps, x_interp, y_interp), axis=-1)

def trajectorys_l2_distance(trajectory_1, trajectory_2):
    return np.linalg.norm(trajectory_1[:,1:] - trajectory_2[:,1:], axis=-1)

def evaluate_trajectory(trajectory, simplified_trajectory):
    trajectory_interp = interpolate_trajectory(simplified_trajectory, trajectory[:,0])
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
        interp_traj = interpolate_trajectory(simplified_traj, traj[:,0])
        dist = trajectorys_l2_distance(traj, interp_traj)
        metrics["ratio"][tid] = simplified_traj.shape[0] / traj.shape[0]
        metrics["mean"][tid] = dist.mean()
        metrics["max"][tid] = dist.max()
    return metrics