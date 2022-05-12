import logging
from pathlib import Path

import yaml
import hydra
import omegaconf
import numpy as np

import tsbench
from tsbench.utils import evaluate_trajectories, interpolate_trajectories

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="run")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    logger.info(f"Configuration Parameters:\n {omegaconf.OmegaConf.to_yaml(cfg)}")

    # Instantiate a search algorithm class
    algo = tsbench.ALGO_REGISTRY.get(cfg.algo.name)()

    # Instantiate a dataset class
    params = cfg.dataset.params if "params" in cfg.dataset else {}
    dataset = tsbench.DATASET_REGISTRY.get(name=cfg.dataset.name)(
        path=hydra.utils.to_absolute_path(cfg.dataset.path), **params
    )

    for key in cfg.dataset.key:
        trajectories = dataset.get_trajectories(key)
        res = []
        for param in cfg.algo.params:
            simplifid_trajectories, runtime_per_query = algo.simplify(
                trajectories, **param
            )
            metrics = evaluate_trajectories(trajectories, simplifid_trajectories)
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
                trajectories, simplifid_trajectories
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
