"""
Features module for quantum encoding utilities.

This module provides amplitude encoding for 60-D quaternion frame vectors
used in quantum DTW for action recognition.
"""

from .amplitude_encoding import (
    encode_unit_vector,
    batch_encode_unit_vectors,
    verify_normalization,
    EPS
)

__all__ = [
    'encode_unit_vector',
    'batch_encode_unit_vectors',
    'verify_normalization',
    'EPS'
]
