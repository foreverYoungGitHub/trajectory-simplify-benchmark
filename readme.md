# Trajectory Simplification Benchmark

This is the codebase for A Scale-Invariant Trajectory Simplification Method for Efficient Data Collection in Videos.

## Usage

### Config
This repo is using hydra as the config system. When we run the pipeline, there are two/three fields we need to check:

- `dataset`: the input trajectories
- `dataset@gt_dataset`: the trajectories used to compare / correct / evaluate
- `algo`: the simplification method

#### `dataset` or `dataset@gt_dataset`

For the dataset, we mainly support the MOT dataset format. An example config for it can be found in `configs/dataset/mot.yaml`. It mainly contains three fields:

- name: the class name of the dataset
- path: the path to the trajectory files
- key: the fields we want to have: for `bbox`, it only contains the bounding boxes with ltrb format; for `bbox_c`, it contains the bounding boxes and the corresponding confidence scores.

#### `algo`

For algorithm, it mainly contains two fields:

- name: the class name of the algorithm
- params: the hyperparameter for the algorithm

An example of it can be find in `configs/algo/ocdag_iou.yaml`.

Currently we experimental test some algorithm, the corresponding class name can be found in following:

- UniformSample: uniform sampling;
- TDTR_2Points: tdtr method, only check the outlier with SED distance;
- TDTR_IoU: tdtr method, only check the outlier with IoU distance;
- DAGv2_2Points: dag method, simplified the trajectories by minimizing the SED distance;
- DAGv2_IoU: dag method, simplified the trajectories by minimizing the IoU distance;
- OCDAG_IOU: our method with IoU distance;

### Trajectory Simplification

This is the normal trajectory simplification pipeline. After correcting the keyframes, we interpolate the trajectory and report metrics comparing the interpolated corrected trajectory and the original trajectories.

Please modify the config file `configs/run.yaml` if needed.

```bash
python run.py
```

### Trajectory Correction

This is the simulated correction pipeline, where we match the predicted bounding boxes on the selected keyframes to the ground truth data. That is, this experimental protocol assumes that any frame selected for manual review is perfectly corrected. After correcting the keyframes, we interpolate the trajectory and report metrics comparing the interpolated corrected trajectory and the ground truth.

Please modify the config file `configs/clean.yaml` if needed.

```bash
python clean.py
```

### Synthetic noisy trajectories generation

In order to evaluate the impact on the algorithm as we vary the level of noise in the input tracking data, we also consider synthetic data. We corrupt the ground truth trajectories with two common tracking mistakes: bounding box jitter and track switches.

```bash
python mot_noisy_bbox_sort.py
```

### Trajectory Visualization

The trajectory can be visualized with the images and video by

```bash
python visual_mot.py
```

Please modify the config file `configs/visual_mot.yaml` if needed.
