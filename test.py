
import pandas as pd
from yaml import safe_load

with open('output/MOTDataset/ByteTrack/DAGv2_2Points/result_bbox.yaml', 'r') as f:
    df = pd.json_normalize(safe_load(f))

df.to_csv("test.txt")


from pathlib import Path
# output/MOTDataset/prob_0.10_iou_0.50/UniformSample
# data ={}
# dirname=Path("output/MOTDataset/")
# for fname in sorted(dirname.glob("prob_*_iou_0.50/TDTR_IOU/result_bbox.yaml")):
#     with open(fname, 'r') as f:
#         df = pd.json_normalize(safe_load(f))
#     df["prob"]=fname.parent.parent.name
#     data[fname.parent.parent.name] = df
# data = pd.concat(data, axis=0, ignore_index=True)
# data.to_csv("test.txt")

# dirname=Path("output/PoseTrackDataset/trainval/TDTR_Points/")
# lines = []
# for fname in sorted(dirname.glob("*_0.log"), key=lambda p: float(p.name[:-6])):
#     with open(fname, "r") as f:
#         line = fname.name[:-4] + " " + f.readlines()[614][:-3] + "\n"
#         line = line.replace(" & ", ",")
#         lines.append(line)
# with open("test.txt", "w") as f:
#     f.writelines(lines)

# dirname=Path("/Users/yaliu/Documents/git/TrackEval/data/trackers/mot_challenge/DanceTrack-train/")
# lines = []
# for fname in sorted(dirname.glob("DAGv2_2Points_*"), key=lambda p: float(p.name.split("_")[2])):
#     with open(fname/"pedestrian_summary.txt", "r") as f:
#         line = fname.name + " " + f.readlines()[1]
#         # print(line)
#         # assert False
#         lines.append(line)
# with open("test.txt", "w") as f:
#     f.writelines(lines)

# import numpy as np
# dirname=Path("output/MOTDataset/MOT20/DAGv2_IOU")
# for fname in dirname.glob("*/*.txt"):
#     tracks = np.loadtxt(fname, delimiter=",").astype(int)
#     if tracks.shape[1] <= 7:
#         ends = np.ones([tracks.shape[0], 4])
#         ends[:, 1:] *= -1
#         tracks = np.concatenate((tracks, ends), axis=1)
#         np.savetxt(
#             fname, tracks.astype(int), fmt="%i", delimiter=","
#         )

# from pathlib import Path
# import shutil

# path1 = Path("/Users/yaliu/Downloads/DanceTrack/train/")
# path2 = Path("/Users/yaliu/Documents/git/TrackEval/data/gt/mot_challenge/DanceTrack-train")
# # for dirs in path.iterdir():
# #     filename = dirs.name[10:]
# #     src_path = dirs/"gt/gt.txt"
# #     dst_path = f"dataset/DanceTrack/{filename}.txt"
# #     if src_path.exists():
# #         shutil.copy(src_path, dst_path)
# for dirs in path1.iterdir():
#     dirname = dirs.name #.name[10:]
#     (path2/dirname/"gt").mkdir(exist_ok=True, parents=True)
    
#     src_path = dirs/"gt/gt.txt"
#     dst_path = path2/dirname/"gt/gt.txt"
#     if src_path.exists():
#         shutil.copy(src_path, dst_path)
#     src_path = dirs/"seqinfo.ini"
#     dst_path = path2/dirname/"seqinfo.ini"
#     if src_path.exists():
#         shutil.copy(src_path, dst_path)
