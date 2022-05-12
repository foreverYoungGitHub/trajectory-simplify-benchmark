import json
import copy
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np

from tsbench.dataset import base, DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PoseTrackDataset(base.BaseDataset):
    def __init__(
        self,
        path: str,
        ref_type: str = "head",
        num_joints: int = 15,
        max_trajectories_per_file: int = 10000,
        **__: Any
    ) -> None:
        super().__init__(path)
        self.ref_type = ref_type
        self.num_joints = num_joints
        self.max_trajectories_per_file = max_trajectories_per_file
        self.trajectory_data, (
            self.filenames,
            self.imagenames,
        ) = self.convert_dataset_to_trajectories()

    def convert_dataset_to_trajectories(self) -> List[Dict]:
        trajectories = defaultdict(list)
        filenames = []
        all_imagenames = []
        for f_index, txt_file in enumerate(self.path.glob("*.json")):
            filenames.append(txt_file.name)
            imagenames = []
            with open(txt_file, "r") as f:
                raw_data = json.load(f)

            for d in raw_data["annolist"]:
                image = d["image"][0]["name"]
                timestamp = d["imgnum"]
                imagenames.append((timestamp[0], image))

                for track in d["annorect"]:
                    track_id = (
                        f_index * self.max_trajectories_per_file + track["track_id"][0]
                    )
                    bbox = np.array(
                        [track["x1"][0], track["y1"][0], track["x2"][0], track["y2"][0]]
                    )
                    joints = -1 * np.ones([15, 2])
                    if len(track["annopoints"]):
                        for point in track["annopoints"][0]["point"]:
                            joints[point["id"]] = [point["x"][0], point["y"][0]]
                    trajectories[track_id].append(
                        {
                            # "image": image,
                            "timestamp": timestamp,
                            "bbox": bbox,
                            "joints": joints,
                            "joints_vis": joints[:, 0] != -1,
                        }
                    )
            all_imagenames.append(imagenames)

        trajectories = [
            dict(
                {
                    k: np.array(
                        [
                            t[k]
                            for t in sorted(
                                trajectories[tid], key=lambda t: t["timestamp"]
                            )
                        ]
                    )
                    for k in trajectories[tid][0].keys()
                },
                **{"tid": tid}
            )
            for tid in trajectories.keys()
        ]
        return trajectories, (filenames, all_imagenames)

    def convert_to_original_json_format(self, trajectories_data: List[Dict]):
        all_trajectories = defaultdict(list)
        for t in trajectories_data:
            all_trajectories[t["tid"] // self.max_trajectories_per_file].append(t)

        all_res = {}
        for f_index, trajectories in all_trajectories.items():
            # construct timestap, index dict to use timestamp fetch the relative data
            timestamp_index_dict = defaultdict(list)
            for idx, traj in enumerate(trajectories):
                for loc, timestamp in enumerate(traj["timestamp"]):
                    timestamp_index_dict[timestamp[0]].append((idx, loc))

            res = []
            for timestamp, imagename in self.imagenames[f_index]:
                annorect = []
                for idx, loc in timestamp_index_dict[timestamp]:
                    traj = trajectories[idx]
                    points = [
                        {
                            "id": [j],
                            "x": [int(traj["joints"][loc, j, 0])],
                            "y": [int(traj["joints"][loc, j, 1])],
                            "score": [1],
                        }
                        for j in range(self.num_joints)
                        if traj["joints_vis"][loc, j]
                    ]
                    annorect.append(
                        {
                            "x1": [int(traj["bbox"][loc, 0])],
                            "y1": [int(traj["bbox"][loc, 1])],
                            "x2": [int(traj["bbox"][loc, 2])],
                            "y2": [int(traj["bbox"][loc, 3])],
                            "score": [1],
                            "track_id": [
                                int(traj["tid"] % self.max_trajectories_per_file)
                            ],
                            "annopoints": [{"point": points}],
                        }
                    )
                res.append(
                    {
                        "image": [{"name": imagename}],
                        "annorect": annorect,
                    }
                )
            all_res[self.filenames[f_index]] = {"annolist": res}
        return all_res

    def get_ref_size(self, trajectory):
        if self.ref_type == "head":
            bbox = trajectory["bbox"]
            wh = bbox[:, 2:] - bbox[:, :2]
            ref_size = np.linalg.norm(wh, axis=1, keepdims=True)
        else:
            joints = trajectory["joints"]
            wh = joints.max(axis=1) - joints.min(axis=1)
            ref_size = np.linalg.norm(wh, axis=1, keepdims=True)
        return ref_size

    def get_trajectories(self, key: str) -> Dict[int, np.ndarray]:
        """get trajectories by key

        Args:
            key (str): the key to fetch the relative data

        Returns:
            Dict[int, np.ndarray]: trajectories with trajectory id
                and trajectory_data
        """
        return {
            d["tid"] * self.num_joints
            + i: np.concatenate(
                (
                    d["timestamp"][d["joints_vis"][:, i]],
                    d[key][d["joints_vis"][:, i], i],
                    self.get_ref_size(d)[d["joints_vis"][:, i]],
                ),
                axis=-1,
            )
            for d in self.trajectory_data
            for i in range(self.num_joints)
            if d["joints_vis"][:, i].sum() > 2
        }

    def dump_data(self, output_dir: Path, trajectories: np.ndarray) -> None:
        trajectory_data = copy.deepcopy(self.trajectory_data)
        idx = 0
        for idx in range(len(trajectory_data)):
            for j in range(self.num_joints):
                if trajectory_data[idx]["joints_vis"][:, j].sum() <= 2:
                    continue
                joints_vis = trajectory_data[idx]["joints_vis"][:, j]
                tid = trajectory_data[idx]["tid"] * self.num_joints + j
                trajectory_data[idx]["joints"][joints_vis, j] = trajectories[tid][
                    :, 1:3
                ]

        all_res = self.convert_to_original_json_format(trajectory_data)
        for filename, data in all_res.items():
            with open(output_dir / filename, "w") as f:
                json.dump(data, f, indent=2)
