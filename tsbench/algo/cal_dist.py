import numpy as np


def cacl_PEDs(trajectory: np.ndarray) -> np.ndarray:
    """Compute the PED distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 3
    ), f"To calculate PED, The feature dim for trajectory must be 3 (vs {trajectory.shape[1]})"
    p_start = trajectory[:1, 1:]
    p_end = trajectory[-1:, 1:]
    points = trajectory[1:-1, 1:]
    line_end_to_start = p_end - p_start
    d = np.abs(np.cross(line_end_to_start, p_start - points)) / np.linalg.norm(
        line_end_to_start, axis=-1
    )
    return d


def cacl_SEDs(trajectory: np.ndarray) -> np.ndarray:
    """Compute the Synchronized Euclidean Distance (SED) distance
    for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 3
    ), f"To calculate SED, The feature dim for trajectory must be 3 (vs {trajectory.shape[1]})"
    time_ratio = (trajectory[1:-1, :1] - trajectory[:1, :1]) / (
        trajectory[-1:, :1] - trajectory[:1, :1] + 1e-10
    )
    p_start = trajectory[:1, 1:]
    p_end = trajectory[-1:, 1:]
    points = trajectory[1:-1, 1:]
    estimate_points = p_start * (1 - time_ratio) + p_end * time_ratio
    d = np.linalg.norm(points - estimate_points, axis=-1)
    return d


def ious(bbox_1: np.ndarray, bbox_2: np.ndarray) -> np.ndarray:
    """compute the IoU score for two list of bounding boxes

    Args:
        bbox_1 (np.ndarray): bounding boxes
        bbox_2 (np.ndarray): bounding boxes

    Returns:
        np.ndarray: IoU score
    """
    assert (
        bbox_1.shape[0] == bbox_2.shape[0]
    ), "The length of the dataset has to be same"
    xmin = np.max((bbox_1[:, 0], bbox_2[:, 0]), axis=0)
    xmax = np.min((bbox_1[:, 2], bbox_2[:, 2]), axis=0)
    ymin = np.max((bbox_1[:, 1], bbox_2[:, 1]), axis=0)
    ymax = np.min((bbox_1[:, 3], bbox_2[:, 3]), axis=0)

    if np.all(xmax <= xmin) or np.all(ymax <= ymin):
        return 0.0

    area_1 = (bbox_1[:, 2] - bbox_1[:, 0]) * (bbox_1[:, 3] - bbox_1[:, 1])
    area_2 = (bbox_2[:, 2] - bbox_2[:, 0]) * (bbox_2[:, 3] - bbox_2[:, 1])
    area_int = (xmax - xmin) * (ymax - ymin)
    iou = area_int / (area_1 + area_2 - area_int).astype(np.float)

    return iou


def cacl_SIOUs(trajectory: np.ndarray) -> np.ndarray:
    """Compute the SIOU distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 5
    ), f"To calculate SIOU, The feature dim for trajectory must be 5 (vs {trajectory.shape[1]})"
    time_ratio = (trajectory[1:-1, :1] - trajectory[:1, :1]) / (
        trajectory[-1:, :1] - trajectory[:1, :1] + 1e-10
    )
    box_start = trajectory[:1, 1:]
    box_end = trajectory[-1:, 1:]
    boxes = trajectory[1:-1, 1:]
    estimate_boxes = box_start * (1 - time_ratio) + box_end * time_ratio
    d = 1 - ious(boxes, estimate_boxes)
    return d


def cacl_LISSED(trajectory: np.ndarray) -> np.ndarray:
    """Compute the Integral Square Synchronized Euclidean Distance (LISSED)
    distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 3
    ), f"To calculate SED, The feature dim for trajectory must be 3 (vs {trajectory.shape[1]})"
    if trajectory.shape[0] == 2:
        return 0
    d = cacl_SEDs(trajectory)
    return (d * d).sum()


def cacl_LASED(trajectory: np.ndarray) -> np.ndarray:
    """Compute the Average Synchronized Euclidean Distance (LISSED)
    distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 3
    ), f"To calculate SED, The feature dim for trajectory must be 3 (vs {trajectory.shape[1]})"
    if trajectory.shape[0] == 2:
        return 0
    d = cacl_SEDs(trajectory)
    return d.mean()
