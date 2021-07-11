from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


__all__ = ['Numeric', 'FillNa']


class Numeric(BaseEstimator, TransformerMixin):
    def __init__(self, errors="coerce", fillna_value=0, dtype=np.int32,
                 suffix="_prep"):
        self.errors = errors
        self.fillna_value = fillna_value
        self.dtype = dtype
        self.suffix = suffix

    def fit(self, X, y=None):
        self.columns = getattr(X, 'columns', None)
        return self

    def transform(self, X):
        check_is_fitted(self, ['columns'])
        for col in self.columns:
            X[col+self.suffix] = pd.to_numeric(X[col], errors=self.errors)\
                                    .fillna(value=self.fillna_value)\
                                    .apply(self.dtype)
        if self.columns is not None:
            X = X.convert_dtypes()
        return X


FILLING_MAP = {
    "median": lambda X: X.quantile(0.5),
    "mode": lambda X: mode(X)[0]
}


class FillNa(BaseEstimator, TransformerMixin):
    def __init__(self, method="median", suffix="_prep"):
        method = FILLING_MAP.get(method, method)
        if not callable(method):
            raise ValueError("Unknown filling method")
        self.method = method
        self.suffix = suffix

    def fit(self, X, y=None):
        self.columns = getattr(X, 'columns', None)
        self.fill_values = {}
        for col in self.columns:
            fill_value = self.method(X[col])
            self.fill_values[col] = fill_value
        return self

    def transform(self, X):
        check_is_fitted(self, ['columns', 'fill_values'])
        for col in self.columns:
            fill_value = self.fill_values[col]
            X[col+self.suffix] = X[col].fillna(value=fill_value)
        return X
