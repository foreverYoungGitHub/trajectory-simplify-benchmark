from tsbench.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset
It must returns an instance of :class:`Dataset`.
"""

from .sample import SampleDataset
from .mot import MOTDataset
from .pose_track import PoseTrackDataset