import numpy as np

from tsbench.algo import base, ALGO_REGISTRY

@ALGO_REGISTRY.register()
class UniformSample(base.BaseTS):
    """Directed Acyclic Graph Based"""
    def simplify_one_trajectory(
        self,
        trajectory: np.ndarray,
        sample_rate: float,
    ) -> np.ndarray:
        indices = np.arange(0, trajectory.shape[0], sample_rate, dtype=int)
        if len(indices) < 2:
            indices = [0, trajectory.shape[0]-1]
        if indices[-1] != trajectory.shape[0]-1:
            indices[-1] = trajectory.shape[0]-1
        indices = np.unique(indices)
        simplified_trajectory = trajectory[indices]
        return simplified_trajectory
