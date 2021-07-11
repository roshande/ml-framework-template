import pandas as pd
import sklearn.feature_selection as fs
from sklearn.utils.validation import check_is_fitted


__all__ = ['SelectKBestWrapper', 'SelectFromModelWrapper',
           'VarianceThresholdWrapper', 'SelectPercentileWrapper',
           'GenericUnivariateSelectWrapper', 'quant_col_selector',
           'cat_col_selector']


def quant_col_selector(X):
    return X.select_dtypes(include="number").columns.to_list()


def cat_col_selector(X):
    return X.select_dtypes(exclude="number").columns.to_list()


class PandasWrapper:
    def fit(self, X, y=None):
        self._columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['_columns'])
        ret = super().transform(X)
        if self._columns is not None:
            return pd.DataFrame(ret, index=X.index, columns=self._columns)


class SelectKBestWrapper(fs.SelectKBest):
    def fit(self, X, y=None):
        self._columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['_columns'])
        ret = super().transform(X)
        if self._columns is not None:
            mask = self.get_support()
            columns = self._columns[mask]
            return pd.DataFrame(ret, index=X.index, columns=columns)
        return ret


class GenericUnivariateSelectWrapper(fs.GenericUnivariateSelect):
    def fit(self, X, y=None):
        self._columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['_columns'])
        ret = super().transform(X)
        if self._columns is not None:
            mask = self.get_support()
            columns = self._columns[mask]
            return pd.DataFrame(ret, index=X.index, columns=columns)
        return ret


class SelectPercentileWrapper(fs.SelectPercentile):
    def fit(self, X, y=None):
        self._columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['_columns'])
        ret = super().transform(X)
        if self._columns is not None:
            mask = self.get_support()
            columns = self._columns[mask]
            return pd.DataFrame(ret, index=X.index, columns=columns)
        return ret


class VarianceThresholdWrapper(fs.VarianceThreshold):
    def fit(self, X, y=None):
        self._columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['_columns'])
        ret = super().transform(X)
        if self._columns is not None:
            mask = self.get_support()
            columns = self._columns[mask]
            return pd.DataFrame(ret, index=X.index, columns=columns)
        return ret


class SelectFromModelWrapper(fs.SelectFromModel):
    def fit(self, X, y=None):
        self._columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['_columns'])
        ret = super().transform(X)
        if self._columns is not None:
            mask = self.get_support()
            columns = self._columns[mask]
            return pd.DataFrame(ret, index=X.index, columns=columns)
        return ret
