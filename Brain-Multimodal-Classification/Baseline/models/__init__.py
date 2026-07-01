"""
Baseline Models Module

Provides simple baseline classifiers for brain network/image classification:
  - FCMlpClassifier: FC matrix → MLP (flattened connectivity)
  - sMRIMlpClassifier: 3D sMRI volume → AvgPool → MLP
"""

from .fc_mlp_baseline import FCMlpClassifier
from .smri_mlp_baseline import sMRIMlpClassifier

__all__ = [
    "FCMlpClassifier",
    "sMRIMlpClassifier",
]