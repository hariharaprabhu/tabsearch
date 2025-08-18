"""
HybridVectorizer: Unified embedding for tabular, text, and multimodal data.

A robust vectorization library that automatically handles mixed data types
and provides powerful similarity search capabilities.
"""

__version__ = "0.1.1"

from .core import HybridVectorizer
from .exceptions import (
    HybridVectorizerError,
    ModelNotFittedError,
)

__all__ = [
    "HybridVectorizer", 
    "HybridVectorizerError",
    "ModelNotFittedError"
]