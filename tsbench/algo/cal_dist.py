from re import L
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


def ious(bbox_1: np.ndarray, bbox_2: np.ndarray, iou_type: str = "iou") -> np.ndarray:
    """compute the IoU score for two list of bounding boxes

    Args:
        bbox_1 (np.ndarray): bounding boxes, format with ltrb
        bbox_2 (np.ndarray): bounding boxes, format with ltrb
        iou_type (str): the type of the iou scores

    Returns:
        np.ndarray: IoU score
    """
    assert iou_type in [
        "iou",
        "diou",
        "ciou",
    ], "Expected iou_type are [iou, diou, ciou], " "but got {}".format(iou_type)

    assert (
        bbox_1.shape[0] == bbox_2.shape[0]
    ), "The length of the dataset has to be same"

    lt = np.max((bbox_1[:, :2], bbox_2[:, :2]), axis=0)
    rb = np.min((bbox_1[:, 2:], bbox_2[:, 2:]), axis=0)
    wh_1 = bbox_1[:, 2:] - bbox_1[:, :2]
    wh_2 = bbox_2[:, 2:] - bbox_2[:, :2]

    area_i = np.prod(rb - lt, axis=1) * (lt < rb).all(axis=1)
    area_1 = np.prod(wh_1, axis=1)
    area_2 = np.prod(wh_2, axis=1)

    area_union = area_1 + area_2 - area_i
    iou = (area_i + 1e-7) / (area_union + 1e-7)

    if iou_type == "iou":
        return iou

    ctr_1 = (bbox_1[:, :2] + bbox_1[:, 2:]) / 2
    ctr_2 = (bbox_2[:, :2] + bbox_2[:, 2:]) / 2
    outer_lt = np.min((bbox_1[:, :2], bbox_2[:, :2]), axis=0)
    outer_rb = np.max((bbox_1[:, 2:], bbox_2[:, 2:]), axis=0)

    inter_diag = ((ctr_1 - ctr_2) ** 2).sum(axis=1)
    outer_diag = ((outer_rb - outer_lt) ** 2).sum(axis=1) + 1e-7

    if iou_type == "diou":
        diou = iou - inter_diag / outer_diag
        return np.clip(diou, -1.0, 1.0)

    if iou_type == "ciou":
        v = (4 / (np.pi ** 2)) * np.power(
            (np.arctan(wh_1[:, 0] / wh_1[:, 1]) - np.arctan(wh_2[:, 0] / wh_2[:, 1])),
            2,
        )
        alpha = v / (1 - iou + v + 1e-7)
        ciou = iou - (inter_diag / outer_diag + alpha * v)
        return np.clip(ciou, -1.0, 1.0)


def cacl_SIOUs(trajectory: np.ndarray, iou_type: str = "iou") -> np.ndarray:
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
    d = 1 - ious(boxes, estimate_boxes, iou_type)
    return d


def cacl_LISSED(trajectory: np.ndarray) -> np.ndarray:
    """Compute the Local Integral Square Synchronized Euclidean Distance (LISSED)
    distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 3
    ), f"To calculate SED, The feature dim for trajectory must be 3 (vs {trajectory.shape[1]})"
    if trajectory.shape[0] == 2:
        return 0
    d = cacl_SEDs(trajectory)
    return (d * d).sum()


def cacl_LASED(trajectory: np.ndarray) -> np.ndarray:
    """Compute the Local Average Synchronized Euclidean Distance (LASED)
    distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 3
    ), f"To calculate SED, The feature dim for trajectory must be 3 (vs {trajectory.shape[1]})"
    if trajectory.shape[0] == 2:
        return 0
    d = cacl_SEDs(trajectory)
    return d.mean()


def cacl_LISSIOUs(trajectory: np.ndarray, iou_type: str = "iou") -> np.ndarray:
    """Compute the Local Integral Square Synchronized IOU Distance (LISSIOUs)
    distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 5
    ), f"To calculate SED, The feature dim for trajectory must be 5 (vs {trajectory.shape[1]})"
    if trajectory.shape[0] == 2:
        return 0
    d = cacl_SIOUs(trajectory, iou_type)
    return (d * d).sum()


def cacl_LAIOUs(trajectory: np.ndarray, iou_type: str = "iou") -> np.ndarray:
    """Compute the Local Average Synchronized IOU Distance (LASIOUs)
    distance for the middle points [1:-1] of the trajectory."""
    assert (
        trajectory.shape[1] == 5
    ), f"To calculate SED, The feature dim for trajectory must be 5 (vs {trajectory.shape[1]})"
    if trajectory.shape[0] == 2:
        return 0
    d = cacl_SIOUs(trajectory, iou_type)
    return (d * d).sum()
