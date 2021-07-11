import pandas as pd
from sklearn.impute import SimpleImputer

__all__ = ['SimpleImputerFeatureNames']


class SimpleImputerFeatureNames(SimpleImputer):

    def fit(self, X, y=None):
        self._feature_names = getattr(X, 'columns', None)
        if self._feature_names is None:
            self._feature_names = [d for d in range(X.shape[1])]
        self._index = getattr(X, 'index', None)
        if self._index is None:
            self._index = [d for d in range(X.shape[0])]
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        temp = super().fit_transform(X, y)
        if self._feature_names is not None and self._index is not None:
            return pd.DataFrame(temp, columns=self._feature_names,
                                index=self._index)
        return temp

    def get_feature_names(self):
        return self._feature_names
