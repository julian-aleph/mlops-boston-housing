"""Transformadores de preprocesamiento embebidos en el sklearn Pipeline.

Mantener el preprocesamiento dentro del Pipeline garantiza que los mismos
pasos aprendidos en training se apliquen en serving sin lógica duplicada.
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Aplica clipping por cuantiles aprendidos durante fit."""

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x: pd.DataFrame | np.ndarray, y: object = None) -> Self:
        self._validate_quantiles()
        frame = self._to_dataframe(x)
        self.lower_bounds_ = frame.quantile(self.lower_quantile)
        self.upper_bounds_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        self._validate_is_fitted()
        frame = self._to_dataframe(x)
        clipped = frame.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

        if isinstance(x, pd.DataFrame):
            return clipped

        return clipped.to_numpy()

    def _validate_quantiles(self) -> None:
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")

    def _validate_is_fitted(self) -> None:
        if not hasattr(self, "lower_bounds_") or not hasattr(self, "upper_bounds_"):
            raise ValueError("QuantileClipper must be fitted before transform.")

    @staticmethod
    def _to_dataframe(x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()

        array = np.asarray(x)
        if array.ndim != 2:
            raise ValueError("Input must be two-dimensional.")

        return pd.DataFrame(array)
