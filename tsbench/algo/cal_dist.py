import numpy as np


def cacl_PEDs(trajectory: np.ndarray) -> np.ndarray:
    """Compute the PED distance for the middle points [1:-1] of the trajectory."""
    p_start = trajectory[:1, 1:]
    p_end = trajectory[-1:, 1:]
    points = trajectory[1:-1, 1:]
    line_end_to_start = p_end - p_start
    d = np.abs(np.cross(line_end_to_start, p_start - points)) / np.linalg.norm(line_end_to_start, axis=-1)
    return d

def cacl_SEDs(trajectory: np.ndarray) -> np.ndarray:
    """Compute the SED distance for the middle points [1:-1] of the trajectory."""
    time_ratio = (trajectory[1:-1, :1] - trajectory[:1, :1]) / (trajectory[-1:, :1] - trajectory[:1, :1] + 1e-10)
    p_start = trajectory[:1, 1:]
    p_end = trajectory[-1:, 1:]
    points = trajectory[1:-1, 1:]
    estimate_points = p_start * (1 - time_ratio) + p_end * time_ratio
    d = np.linalg.norm(points - estimate_points, axis=-1)
    return d

