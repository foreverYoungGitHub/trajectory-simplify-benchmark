import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from tsbench import dataset

POSETRACK18_LM_NAMES = [  # This is used to identify the IDs.
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "head_bottom",
    "nose",
    "head_top",
]

posetrack = dataset.PoseTrackDataset(
    "./dataset/PoseTrack/sample", ref_type="whole", ref_ratio=1
)
posetrack_len = len(posetrack.trajectory_data)
posetrack_traj = posetrack.get_trajectories("joints")

center_all = (posetrack_traj[2] + posetrack_traj[3])/2
for tid, traj in tqdm.tqdm(posetrack_traj.items()):
    if tid >= len(POSETRACK18_LM_NAMES):
        break
    jid = tid % len(POSETRACK18_LM_NAMES)

    center = center_all[np.isin(center_all[:,0], traj[:, 0])]
    # x = np.arange(traj.shape[0])
    # plt.plot(traj[:, 0], traj[:, 1] - traj[:, 1].mean(), label="x")
    # plt.plot(traj[:, 0], traj[:, 2] - traj[:, 2].mean(), label="y")
    plt.plot(traj[:, 0], traj[:, 1] - center[:, 1], label="x")
    plt.plot(traj[:, 0], traj[:, 2] - center[:, 2], label="y")
    # plt.plot(traj[:, 0], traj[:, 1] - center[:, 1] - traj[0, 1] + center[0, 1], label="x")
    # plt.plot(traj[:, 0], traj[:, 2] - center[:, 1] - traj[0, 2] + center[0, 2], label="y")
    plt.legend()
    plt.savefig(f"output/PoseTrackDataset/test_figures/{tid:06d}_{POSETRACK18_LM_NAMES[jid]}_x_y.jpg")
    plt.clf()

    plt.plot(traj[:, 1] - center[:, 1], traj[:, 2] - center[:, 2], label="xy")
    # plt.plot(traj[:, 1], traj[:, 2], label="xy")
    plt.legend()
    # plt.show()
    plt.savefig(f"output/PoseTrackDataset/test_figures/{tid:06d}_{POSETRACK18_LM_NAMES[jid]}_xy.jpg")
    plt.clf()
    # assert False
