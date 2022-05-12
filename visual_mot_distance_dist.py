import numpy as np
import matplotlib.pyplot as plt

from tsbench.algo import cal_dist

tracks_file = "dataset/DanceTrack/sample/0004.txt"
gt_tracks = np.loadtxt(tracks_file, delimiter=",").astype(int)


tracks_file = "output/MOTDataset/DanceTrack/TDTR_IOU/0.2_iou/0004.txt"
# tracks_file = "output/MOTDataset/DanceTrack/TDTR_2Points/25_0/0004.txt"
# tracks_file = "output/MOTDataset/DanceTrack/TDTR_2Points/8_0.01/0004.txt"
ts_tracks = np.loadtxt(tracks_file, delimiter=",").astype(int)

tids = np.unique(ts_tracks[:,1])

ref_ratio = 0.01
for tid in tids:
    gt_track = gt_tracks[gt_tracks[:,1]==tid]
    ts_track = ts_tracks[ts_tracks[:,1]==tid]

    wh = gt_track[:, 4:6] 
    gt_track[:, 4:6] += gt_track[:, 2:4]
    ts_track[:, 4:6] += ts_track[:, 2:4]
    
    iou_dist = 1 - cal_dist.ious(gt_track[:, 2:6], ts_track[:, 2:6])
    iou_dist = iou_dist * 100
    
    lt_dist = np.linalg.norm(gt_track[:, 2:4] - ts_track[:, 2:4], axis=-1)
    rb_dist = np.linalg.norm(gt_track[:, 4:6] - ts_track[:, 4:6], axis=-1)
    pt_dist = (lt_dist+rb_dist)/2
    
    ref_size = ref_ratio * np.linalg.norm(wh, axis=1)
    lt_ref_dist = lt_dist / ref_size
    rb_ref_dist = rb_dist / ref_size
    pt_ref_dist = (lt_ref_dist+rb_ref_dist)/2

    # x = np.arange(gt_track.shape[0])
    x = np.arange(100)
    plt.plot(x, iou_dist[:100], label = "iou_dist")
    plt.plot(x, pt_dist[:100], label = "pt_dist")
    # plt.plot(x, rb_dist, label = "rb_dist")
    plt.plot(x, pt_ref_dist[:100], label = "pt_ref_dist")
    # plt.plot(x, rb_ref_dist, label = "rb_ref_dist")
    plt.legend()
    plt.show()
    assert False
    