import logging
from pathlib import Path
from collections import defaultdict

import yaml
import hydra
import omegaconf
import numpy as np

import tsbench
from tsbench.utils import evaluate_trajectories, interpolate_trajectories

logger = logging.getLogger(__name__)

def clean_simplified_trajectory_with_gt(simplified_trajectories, gt_tids, gt_trajectories):
    selected_timestamps = defaultdict(list)
    for tid, traj in simplified_trajectories.items():
        # get all the timestamps in the simplified traj
        timestamps = traj[:, 0]
        # get all the correct track id
        t_gt_tids = gt_tids[tid]
        for timestamp in timestamps:
            gt_tid = t_gt_tids[t_gt_tids[:,0]==timestamp]
            assert len(gt_tid) == 1
            gt_tid = gt_tid[0,1]
            selected_timestamps[gt_tid].append(timestamp)
    corrected_simplied_trajectory = {}
    for tid in gt_trajectories.keys():
        timstamps = selected_timestamps[tid]
        gt_traj = gt_trajectories[tid]
        mask = np.isin(gt_traj[:,0],timstamps)
        # make sure the first and last objects are covered
        mask[0] = True
        mask[-1] = True
        corrected_simplied_trajectory[tid] = gt_traj[mask]
    return corrected_simplied_trajectory

@hydra.main(config_path="configs", config_name="clean")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    logger.info(f"Configuration Parameters:\n {omegaconf.OmegaConf.to_yaml(cfg)}")

    # Instantiate a search algorithm class
    algo = tsbench.ALGO_REGISTRY.get(cfg.algo.name)()

    # Instantiate a dataset class
    params = cfg.dataset.params if "params" in cfg.dataset else {}
    dataset = tsbench.DATASET_REGISTRY.get(name=cfg.dataset.name)(
        path=hydra.utils.to_absolute_path(cfg.dataset.path), **params
    )

    params = cfg.gt_dataset.params if "params" in cfg.gt_dataset else {}
    gt_dataset = tsbench.DATASET_REGISTRY.get(name=cfg.gt_dataset.name)(
        path=hydra.utils.to_absolute_path(cfg.gt_dataset.path), **params
    )

    for key in cfg.dataset.key:
        trajectories = dataset.get_trajectories(key)
        gt_tids = dataset.get_trajectories("gt_tids")
        gt_trajectories = gt_dataset.get_trajectories(key)
        res = []
        for param in cfg.algo.params:
            simplified_trajectories, runtime_per_query = algo.simplify(
                trajectories, **param
            )
            simplified_trajectories = clean_simplified_trajectory_with_gt(simplified_trajectories, gt_tids, gt_trajectories)
            metrics = evaluate_trajectories(gt_trajectories, simplified_trajectories)
            metrics["runtime"] = runtime_per_query
            metrics = {
                    "param": dict(param),
                    "metrics": {
                        k: float(np.mean(list(v.values()))) for k, v in metrics.items()
                    },
                }
            res.append(metrics)
            logger.info(metrics)

            # (0) Dump the interpolate trajectories
            interp_trajectories = interpolate_trajectories(
                gt_trajectories, simplified_trajectories
            )
            out = (
                Path(hydra.utils.to_absolute_path(cfg.output))
                / cfg.dataset.name
                / Path(cfg.dataset.path).name
                / cfg.algo.name
                / "_".join([str(v) for v in param.values()])
            )
            out.mkdir(exist_ok=True, parents=True)
            dataset.dump_data(out, interp_trajectories)

        # (1) Save the result on the local log directory
        with open(f"result_{key}.yaml", "wt") as f:
            yaml.dump(res, f)

        # (2) And the output directory.
        out = (
            Path(hydra.utils.to_absolute_path(cfg.output))
            / cfg.dataset.name
            / Path(cfg.dataset.path).name
            / cfg.algo.name
            / f"result_{key}.yaml"
        )
        out.parent.mkdir(
            exist_ok=True, parents=True
        )  # Make sure the parent directory exists
        with out.open("wt") as f:
            yaml.dump(res, f)


if __name__ == "__main__":
    main()
