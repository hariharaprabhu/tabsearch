"""Custom exceptions for HybridVectorizer."""

class HybridVectorizerError(Exception):
    """Base exception for HybridVectorizer."""
    pass

class ModelNotFittedError(HybridVectorizerError):
    """Raised when model hasn't been fitted."""
    def __init__(self):
        super().__init__(
            "Model not fitted. Please call fit_transform() with your training data first.\n"
            "Example: vectorizer.fit_transform(df)"
        )

class InvalidQueryError(HybridVectorizerError):
    """Raised when query is invalid."""
    pass

class DimensionMismatchError(HybridVectorizerError):
    """Raised when vector dimensions don't match."""
    pass

class UnsupportedDataTypeError(HybridVectorizerError):
    """Raised when data type is not supported."""
    pass