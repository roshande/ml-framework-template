import numpy as np
import pandas as pd
from typing import Any
from itertools import chain
from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from data_transformers import Identity
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators
from sklearn.compose import ColumnTransformer
from scipy import sparse
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')

__all__ = ['PandasFeatureUnion', 'PandasColumnTransformer', 'make_union']


class PandasColumnTransformer(BaseEstimator, TransformerMixin):
    """A wrapper around sklearn.column.ColumnTransformer to facilitate
    recovery of column (feature) names"""

    def __init__(self, transformers, *, remainder="drop", sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False):
        """Initialize by creating ColumnTransformer object
        https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
        Args:
            transformers (list of length-3 tuples): (name, Transformer, target columns); see docs
            kwargs: keyword arguments for sklearn.compose.ColumnTransformer
        """
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.col_transformer = ColumnTransformer(
            transformers, remainder, sparse_threshold, n_jobs, transformer_weights, verbose)
        #self.transformed_col_names: List[str] = []

    def _get_col_names(self, X: pd.DataFrame):
        """Get names of transformed columns from a fitted self.col_transformer
        Args:
            X (pd.DataFrame): DataFrame to be fitted on
        Yields:
            Iterator[Iterable[str]]: column names corresponding to each transformer
        """
        for name, transformer, cols in self.col_transformer.transformers_:
            print(name, transformer, cols)
            if hasattr(transformer, "get_feature_names"):
                yield transformer.get_feature_names(cols)
                # print(transformer.get_feature_names(cols))
            elif name == "remainder" and self.col_transformer.remainder=="passthrough":
                yield X.columns[cols].tolist()
                # print(X.columns[cols].tolist())
            elif name == "remainder" and self.col_transformer.remainder=="drop":
                continue
            else:
                yield cols

    def fit(self, X: pd.DataFrame, y: Any=None):
        """Fit ColumnTransformer, and obtain names of transformed columns in advance
        Args:
            X (pd.DataFrame): DataFrame to be fitted on
            y (Any, optional): Purely for compliance with transformer API. Defaults to None.
        """
        assert isinstance(X, pd.DataFrame)
        self.col_transformer = self.col_transformer.fit(X)
        self.transformed_col_names = list(chain.from_iterable(self._get_col_names(X)))
        #print(self.transformed_col_names)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a new DataFrame using fitted self.col_transformer
        Args:
            X (pd.DataFrame): DataFrame to be transformed
        Returns:
            pd.DataFrame: DataFrame transformed by self.col_transformer
        """
        assert isinstance(X, pd.DataFrame)
        check_is_fitted(self, ['col_transformer', 'transformed_col_names'])
        transformed_X = self.col_transformer.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(transformed_X, columns=self.transformed_col_names, index=X.index)
        return pd.DataFrame.sparse.from_spmatrix(
            transformed_X, columns=self.transformed_col_names, index=X.index)


class ColumnTransformerWrapper(ColumnTransformer):
    def fit_transform(self, X, y=None):
        self.columns = getattr(X, 'columns', None)
        transformed_X = super().fit_transform(X, y)
        if self.columns is not None:
            transformed_X = pd.DataFrame(transformed_X, index=X.index, columns=self.columns)
            transformed_X = transformed_X.convert_dtypes(convert_string=False)
        return transformed_X

    def fit(self, X, y=None):
        self.columns = getattr(X, 'columns', None)
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, ['columns'])
        transformed_X = super().transform(X)
        if self.columns is not None:
            transformed_X = pd.DataFrame(transformed_X, index=X.index, columns=self.columns)
        return transformed_X


class PandasFeatureUnion(FeatureUnion):
    """
        FeatureUnion implementation for pandas' DataFrame by retaining its properties
    """
    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        #X, y = self._validate_data(X, y, accept_sparse='csr', dtype=np.float_, order="C")
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return self._stack(Xs)

    def _merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def _stack(self, Xs):
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        elif all(isinstance(x, pd.DataFrame) for x in Xs):
            Xs = self._merge_dataframes_by_column(Xs)
        return Xs

    def transform(self, X: pd.DataFrame):
        #self._validate_data(X, reset=False)
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        return self._stack(Xs)


def make_union(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return PandasFeatureUnion(
        _name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)


if __name__ == '__main__':
    check_estimator(linear_model.LogisticRegression(), generate_only=True)
    check_estimator(PandasFeatureUnion([('id', Identity())]))
