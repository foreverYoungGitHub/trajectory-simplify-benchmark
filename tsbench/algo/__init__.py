from tsbench.registry import Registry

ALGO_REGISTRY = Registry("ALGO")
ALGO_REGISTRY.__doc__ = """
Registry for algorithm for trajectory simplify
It must returns an instance of :class:`BaseTS`.
"""

from .dp import DP, TDTR, TDTR_IOU, TDTR_2Points
from .dag import DAG, DAG_IOU
