"""Training compatibility surface; the implementation lives in zero_ttt.learner."""

from typing import Any

__all__ = ["Learner", "LearnerDataIdentity", "Trainer"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from zero_ttt import learner

        return getattr(learner, name)
    raise AttributeError(name)
