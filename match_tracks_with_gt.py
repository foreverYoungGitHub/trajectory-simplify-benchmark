import logging
from pathlib import Path

import tqdm
import numpy as np

seed = 42


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def match_bbox(tracks: np.ndarray, gt_tracks: np.ndarray):
    """add bounding box jitter and use the original tid

    Args:
        tracks (np.ndarray): original mot tracks
        gt_tracks (np.ndarray): groundtruth mot tracks

    Returns:
        np.ndarray: mot tracks with assigned trackid
    """
    max_frames_seq = tracks[:, 0].max()
    res = []
    for t in tqdm.trange(1, int(max_frames_seq) + 1):
        framedata = tracks[tracks[:, 0] == t]
        dets = framedata[:, 2:6]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        gt_framedata = gt_tracks[gt_tracks[:, 0] == t]
        gt_bboxs = gt_framedata[:, 2:6]
        gt_bboxs[:, 2:4] += gt_bboxs[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        # match track
        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(dets, gt_bboxs)
        gt_labels = np.ones((dets.shape[0], 1)) * -1
        gt_labels[matches[:, 0]] = gt_framedata[matches[:, 1], 1:2]

        #try to assign -1 label back based on previous label


        if len(unmatched_detections) > 0:
            logging.info(
                f"timestamp {t} has {len(unmatched_detections)} unmatched detection, which are: \n {framedata[unmatched_detections, 1]}"
            )
        if len(unmatched_trackers) > 0:
            logging.info(
                f"timestamp {t} has {len(unmatched_trackers)} unmatched tracks, which are: \n {gt_framedata[unmatched_trackers, 1]}"
            )
        
        # generate output tracks
        dets[:, 2:4] -= dets[:, 0:2]
        track_data = np.concatenate([framedata[:,:10], gt_labels], axis=1)
        res.append(track_data)
    res = np.concatenate(res)
    gt_labels, gt_counts = np.unique(res[:,10], return_counts=True)
    all_gt_labels, all_gt_counts = np.unique(gt_tracks[:,1], return_counts=True)
    keep_gt_labels = []
    for label, count in zip(all_gt_labels, all_gt_counts):
        if label not in gt_labels:
            continue
        track_count = gt_counts[gt_labels==label]
        assert track_count.shape[0] == 1
        if track_count[0] < count / 5:
            continue
        keep_gt_labels.append(label)
    # print(keep_gt_labels)
    logging.warning(f"num keep/all tracks in gt: {len(keep_gt_labels)}/{len(all_gt_labels)}")
    mask = np.isin(gt_tracks[:,1], keep_gt_labels)
    return res, gt_tracks[mask]


def match_track_data_with_gt(tracker_dir: Path, gt_dir: Path):
    tracker_dir = Path(tracker_dir)
    gt_dir = Path(gt_dir)
    for txt_file in gt_dir.glob("*.txt"):
        gt_tracks = np.loadtxt(txt_file, delimiter=",")
        tracks = np.loadtxt(tracker_dir / txt_file.name, delimiter=",")
        if gt_tracks.shape[1] > 6:
            gt_tracks = gt_tracks[(gt_tracks[:, 7] <= 7)]
        update_tracks, update_gt_tracks = match_bbox(tracks, gt_tracks)
        np.savetxt(
            tracker_dir / txt_file.name,
            update_tracks,
            fmt="%0.2f",
            delimiter=",",
        )
        (tracker_dir/"gt").mkdir(exist_ok=True, parents=True)
        np.savetxt(
            tracker_dir/"gt" / txt_file.name,
            update_gt_tracks,
            fmt="%i",
            delimiter=",",
        )


# match_track_data_with_gt("dataset/MOT20/ByteTrack", "dataset/MOT20")
match_track_data_with_gt("dataset/DanceTrack/DanceTrack-OCSort", "dataset/DanceTrack")
