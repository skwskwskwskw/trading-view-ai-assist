"""Feature modules and registry bootstrap."""

from . import kernel, momentum, trend, volatility  # noqa: F401
from .base import IndicatorSpec
from .registry import REGISTRY, make_spec

__all__ = ["REGISTRY", "IndicatorSpec", "make_spec"]
