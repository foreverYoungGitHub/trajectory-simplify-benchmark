import logging
from pathlib import Path
from typing import Callable, Dict, List

import tqdm
import numpy as np
from filterpy.kalman import KalmanFilter

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
        v = (4 / (np.pi**2)) * np.power(
            (np.arctan(wh_1[:, 0] / wh_1[:, 1]) - np.arctan(wh_2[:, 0] / wh_2[:, 1])),
            2,
        )
        alpha = v / (1 - iou + v + 1e-7)
        ciou = iou - (inter_diag / outer_diag + alpha * v)
        return np.clip(ciou, -1.0, 1.0)

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


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


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


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id + 1])).reshape(1, -1)
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def add_noise_det(
    dets: np.ndarray,
    jitter_prob: float,
    jitter_scale: float,
    rng,
) -> np.ndarray:
    """add bounding box jitter to the original trajectories

    Args:
        dets (np.ndarray): detection bounding box with (l,t,r,b,conf). Defaults to np.empty((0, 5)).
        jitter_prob (float, optional): the prob of bounding box jitter. Defaults to 0.25.
        jitter_scale (float, optional): the scale of the bounding box jitter. Defaults to 0.1.
        rng (_type_): random generator. Defaults to np.random.default_rng(seed).

    Returns:
        np.ndarray: jitter bounding box with (l,t,r,b,conf)
    """
    mu, sigma = 0, jitter_scale
    ctrs = (dets[:, 2:4] + dets[:, :2]) / 2
    whs = (dets[:, 2:4] - dets[:, :2]) / 1
    # add noise to center
    scale = np.linalg.norm(dets[:, 2:4] - dets[:, :2], axis=1, keepdims=True)
    noise = rng.normal(mu, sigma / 2, (dets.shape[0], 2))
    noise *= scale
    ctrs += noise
    # add noise to wh
    noise = rng.normal(mu, sigma, (dets.shape[0], 2))
    noise = 1 + noise
    whs *= noise
    
    # use clip to avoid outside of image
    lt = np.clip(
        ctrs - whs / 2, a_min=dets[:, :2].min(axis=0), a_max=dets[:, :2].max(axis=0)
    )
    rb = np.clip(
        ctrs + whs / 2, a_min=dets[:, 2:4].min(axis=0), a_max=dets[:, 2:4].max(axis=0)
    )
    # random jitter
    jitter_dets = np.concatenate([lt, rb, dets[:,4:]], axis=1)
    jitter_dets[:,4] = ious(jitter_dets[:,:4], dets[:,:4], "diou") # get scores
    jitter_dets[:,4] = np.clip(jitter_dets[:, 4], 0, 1)
    mask = rng.random(jitter_dets.shape[0]) > jitter_prob
    jitter_dets[mask] = dets[mask]
    return jitter_dets


def add_idswitch(
    dets: np.ndarray,
    tids: np.ndarray,
    iou_thresh: float,
    switch_prob: float,
    rng,
) -> List[List[float]]:
    """add id switches for the high overlaped bounding boxes

    Args:
        dets (np.ndarray): detection bounding box with (l,t,r,b,conf). Defaults to np.empty((0, 5)).
        tids (np.ndarray): the current trajectory id. Defaults to np.empty((0)).
        iou_thresh (float, optional): the iou threshold for bounding box overlap. Defaults to 0.5.
        switch_prob (float, optional): the probability of id switches . Defaults to 0.5.
        rng (_type_, optional): random generator. Defaults to np.random.default_rng(seed).

    Returns:
        List[List[float]]: the switch pair
    """
    ious = iou_batch(dets, dets)
    switched_id = []
    switch_pair = []
    for r_idx, row in enumerate(ious):
        if r_idx in switched_id:
            continue
        (indices,) = np.where(row > iou_thresh)
        indices = indices[indices != r_idx]
        indices = indices[~np.isin(indices, switched_id)]
        prob = rng.random()
        if len(indices) == 0:
            continue
        if prob > switch_prob:
            continue
        switch_id = rng.choice(indices)
        switched_id += [r_idx, switch_id]
        switch_pair += [[tids[r_idx], tids[switch_id]], [tids[switch_id], tids[r_idx]]]
    return switch_pair


def noise_bbox_sort(tracks: np.ndarray, config: Dict, rng) -> np.ndarray:
    """add bounding box jitter and use sort to generate the tid

    Args:
        tracks (np.ndarray): original mot tracks

    Returns:
        np.ndarray: mot tracks with noise
    """
    mot_tracker = Sort()
    max_frames_seq = tracks[:, 0].max()
    res = []
    for t in tqdm.trange(1, max_frames_seq + 1):
        framedata = tracks[tracks[:, 0] == t]
        dets = framedata[:, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        dets[:, 4] = 1

        # adding bounding box noisy
        # dets = add_noise_det(dets, rng=rng, **config)

        # tracking
        trackers = mot_tracker.update(dets)

        # matching
        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(dets, trackers, 0.01)
        if len(unmatched_detections) > 0:
            logging.info(
                f"timestamp {t} has {len(unmatched_detections)} unmatched detection, which are: \n {framedata[unmatched_detections, 1]}"
            )
        if len(unmatched_trackers) > 0:
            logging.warning(
                f"timestamp {t} has {len(unmatched_trackers)} unmatched tracks, which are: \n {trackers[unmatched_trackers, 4]}"
            )
        dets[:, 2:4] -= dets[:, 0:2]
        # generate output tracks
        tid = trackers[matches[:,1], 4:]
        frames = framedata[matches[:,0], :1]
        gt_labels = framedata[matches[:,0], 1:2]
        dets = framedata[matches[:,0], 2:7]
        xyz = np.ones([matches.shape[0], 3]) * -1
        track_data = np.concatenate([frames, tid, dets, xyz, gt_labels], axis=1)

        # remove redundant index
        res.append(track_data)
    # reset KalmanBoxTracker.count
    KalmanBoxTracker.count = 0
    res = np.concatenate(res)
    gt_labels, gt_counts = np.unique(res[:,10], return_counts=True)
    all_gt_labels, all_gt_counts = np.unique(tracks[:,1], return_counts=True)
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
    # mask = np.isin(gt_tracks[:,1], keep_gt_labels)
    return res
    


def noise_bbox(tracks: np.ndarray, config: Dict, rng):
    """add bounding box jitter and use the original tid

    Args:
        tracks (np.ndarray): original mot tracks

    Returns:
        np.ndarray: mot tracks with noise
    """
    max_frames_seq = tracks[:, 0].max()
    res = []
    for t in tqdm.trange(1, max_frames_seq + 1):
        framedata = tracks[tracks[:, 0] == t]
        dets = framedata[:, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        dets[:, 4] = 1

        # adding bounding box noisy
        dets = add_noise_det(dets, rng=rng, **config)

        # generate output tracks
        frames = framedata[:, :1]
        gt_labels = framedata[:, 1:2]
        dets[:, 2:4] -= dets[:, 0:2]
        xyz = np.ones([framedata.shape[0], 3]) * -1
        track_data = np.concatenate([frames, gt_labels, dets, xyz, gt_labels], axis=1)
        res.append(track_data)
    return np.concatenate(res)


def noise_idswitch(tracks: np.ndarray, config: Dict, rng) -> np.ndarray:
    """use the original bounding box jitter and add id switch
    for high overlapped bounding boxes

    Args:
        tracks (np.ndarray): original mot tracks

    Returns:
        np.ndarray: mot tracks with noise
    """
    max_frames_seq = tracks[:, 0].max()
    res = []
    gt_labels = np.copy(tracks[:, 1:2])
    for t in tqdm.trange(1, max_frames_seq + 1):
        framedata = tracks[tracks[:, 0] == t]
        dets = framedata[:, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        dets[:, 4] = 1

        # adding bounding box noisy
        switch_pair = add_idswitch(dets, framedata[:, 1], rng=rng, **config)
        for switch in switch_pair:
            tracks[(tracks[:, 0] >= t) & (tracks[:, 1] == switch[0]), 1] = (
                10000 + switch[1]
            )
        tracks[:, 1] %= 10000

        # generate output tracks
        framedata = tracks[tracks[:, 0] == t]
        frames = framedata[:, :1]
        tids = framedata[:, 1:2]
        dets[:, 2:4] -= dets[:, 0:2]
        xyz = np.ones([framedata.shape[0], 3]) * -1
        gt_label = gt_labels[tracks[:, 0] == t]
        track_data = np.concatenate([frames, tids, dets, xyz, gt_label], axis=1)
        res.append(track_data)
    return np.concatenate(res)


def generate_noisy_mot_data(mot_dir: Path, output_dir: Path, noise_method: Callable, config: Dict):
    mot_dir = Path(mot_dir)
    output_dir = Path(output_dir)
    for txt_file in mot_dir.glob("*.txt"):
        tracks = np.loadtxt(txt_file, delimiter=",").astype(int)
        if tracks.shape[1] > 6:
            tracks = tracks[(tracks[:, 7] <= 7)]
        update_tracks = noise_method(tracks, config, rng=np.random.default_rng(seed))
        output_dir.mkdir(exist_ok=True, parents=True)
        np.savetxt(
            output_dir / txt_file.name,
            update_tracks,
            fmt="%.2f",
            delimiter=",",
        )

# config = [{"jitter_prob": jitter_prob, "jitter_scale": jitter_scale} for jitter_prob in np.arange(0.1,1.,0.1) for jitter_scale in np.arange(0.05,0.2,0.05)]
# config = [{"jitter_prob": 1., "jitter_scale": 0.05}]
# for c in config:
    # generate_noisy_mot_data("dataset/MOT20", f"dataset/MOT20-noisy-sort/prob_{c['jitter_prob']:.2f}_scale_{c['jitter_scale']:.2f}", noise_bbox_sort, c)
    # generate_noisy_mot_data("dataset/MOT20", f"dataset/MOT20-noisy-bbox/prob_{c['jitter_prob']:.2f}_scale_{c['jitter_scale']:.2f}", noise_bbox, c)
    # generate_noisy_mot_data("dataset/DanceTrack", f"dataset/DanceTrack/DanceTrack-noisy-bbox-prob_{c['jitter_prob']:.2f}_scale_{c['jitter_scale']:.2f}", noise_bbox, c)

# generate_noisy_mot_data("dataset/MOT20", f"dataset/MOT20-gt-sort/", noise_bbox_sort, {})

config = [{"iou_thresh": iou_thresh, "switch_prob": switch_prob} for switch_prob in np.arange(0.05,0.30,0.05) for iou_thresh in [0.5]]
for c in config:
    generate_noisy_mot_data("dataset/MOT20", f"dataset/MOT20/MOT20-noisy-idswitch-prob_{c['switch_prob']:.2f}_iou_{c['iou_thresh']:.2f}", noise_idswitch, c)
