"""DAGNN + PI-STNet v3 research implementation.

The package is intentionally separate from :mod:`dagnn_pistnet`: v2 remains a
frozen reference implementation while v3 can evolve under an explicit research
protocol.
"""

from .config import V3ExperimentConfig
from .model import DAGNNPISTNetV3, ForecastDistribution

__all__ = ["DAGNNPISTNetV3", "ForecastDistribution", "V3ExperimentConfig"]
__version__ = "0.3.0-research"
