"""Compatibility imports; estimator implementations now live in separate modules."""
from importlib import import_module
from typing import Any

_MODULES = {
    "BaseEstimator": "baseestimator",
    "LinearModel": "linearmodel",
    "LinearRegression": "LinearRegression",
    "RidgeRegression": "RidgeRegression",
    "LassoRegression": "LassoRegression",
    "PolynomialRegression": "PolynomialRegression",
    "GLM": "GLM",
}
__all__ = list(_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve legacy class imports without importing unrelated solver dependencies."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"rbg.Regression.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
