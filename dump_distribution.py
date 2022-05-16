import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tsbench import dataset


def get_mot_distribution(trajectories):
    bbox_size = []
    trajectories_len = []
    for t in trajectories.values():
        trajectories_len.append(len(t))
        wh = t[:, 3:5] - t[:, 1:3]
        wh = np.linalg.norm(wh, axis=1)
        bbox_size += wh.tolist()
    return sorted(trajectories_len), sorted(bbox_size)


def get_posetrack_distribution(trajectories):
    bbox_size = []
    trajectories_len = []
    for t in trajectories.values():
        trajectories_len.append(len(t))
        wh = t[:, 3]
        bbox_size += wh.tolist()
    return sorted(trajectories_len), sorted(bbox_size)


mot17 = dataset.MOTDataset("./dataset/MOT17")
mot17_len = len(mot17.trajectory_data)
mot17_traj = mot17.get_trajectories("bbox")
mot17_len_dist, mot17_box_dist = get_mot_distribution(mot17_traj)
mot17_len_dist, mot17_box_dist = (
    mot17_len_dist[: int(0.95 * len(mot17_len_dist))],
    mot17_box_dist[: int(0.95 * len(mot17_box_dist))],
)

mot20 = dataset.MOTDataset("./dataset/MOT20")
mot20_len = len(mot20.trajectory_data)
mot20_traj = mot20.get_trajectories("bbox")
mot20_len_dist, mot20_box_dist = get_mot_distribution(mot20_traj)
mot20_len_dist, mot20_box_dist = (
    mot20_len_dist[: int(0.95 * len(mot20_len_dist))],
    mot20_box_dist[: int(0.95 * len(mot20_box_dist))],
)


dancetrack = dataset.MOTDataset("./dataset/DanceTrack")
dancetrack_len = len(dancetrack.trajectory_data)
dancetrack_traj = dancetrack.get_trajectories("bbox")
dancetrack_len_dist, dancetrack_box_dist = get_mot_distribution(dancetrack_traj)
dancetrack_len_dist, dancetrack_box_dist = (
    dancetrack_len_dist[: int(0.95 * len(dancetrack_len_dist))],
    dancetrack_box_dist[: int(0.95 * len(dancetrack_box_dist))],
)

# posetrack = dataset.PoseTrackDataset(
#     "./dataset/PoseTrack/trainval", ref_type="whole", ref_ratio=1
# )
# posetrack_len = len(posetrack.trajectory_data)
# posetrack_traj = posetrack.get_trajectories("joints")
# posetrack_len_dist, posetrack_box_dist = get_posetrack_distribution(posetrack_traj)
# posetrack_len_dist, posetrack_box_dist = (
#     posetrack_len_dist[:int(0.95*len(posetrack_len_dist))],
#     posetrack_box_dist[:int(0.95*len(posetrack_box_dist))],
# )

len_dataset = {
    "mot17": mot17_len,
    "mot20": mot20_len,
    "dancetrack": dancetrack_len,
    # "posetrack": posetrack_len,
}
print(len_dataset)

sns.violinplot(
    data=[mot17_len_dist, mot20_len_dist, dancetrack_len_dist]  # , posetrack_len_dist]
)
plt.xticks([0, 1, 2], ["mot17", "mot20", "dancetrack"])  # , "posetrack"])
plt.savefig("len_dist.pdf")
# plt.show()
plt.clf()

sns.violinplot(
    data=[mot17_box_dist, mot20_box_dist, dancetrack_box_dist]  # , posetrack_box_dist]
)
plt.xticks([0, 1, 2], ["mot17", "mot20", "dancetrack"])  # , "posetrack"])
plt.savefig("box_dist.pdf")
plt.clf()
