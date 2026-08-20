"""Training-data contracts and small development sources."""

from zero_ttt.data.contracts import BatchSource, TrainBatch
from zero_ttt.data.synthetic import SyntheticBatchSource

__all__ = ["BatchSource", "SyntheticBatchSource", "TrainBatch"]
