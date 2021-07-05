import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class SimpleImputerFeatureNames(SimpleImputer):

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self._feature_names = np.asarray(X.columns)
        else:
            self._feature_names = [d for d in range(X.shape[1])]
        if hasattr(X, "index"):
            self._indexes = np.asarray(X.index)
        else:
            self._indexes = [d for d in range(X.shape[0])]
        super().fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        _temp = super().fit_transform(X, y)
        return pd.DataFrame(_temp,
                            index=self._indexes,
                            columns=self._feature_names)

    def get_feature_names(self):
        return self._feature_names

