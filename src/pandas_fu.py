import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from data_transformers import Identity
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators
from scipy import sparse
from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')
__all__ = ['PandasFeatureUnion', 'make_union']

class PandasFeatureUnion(FeatureUnion):
    """
        FeatureUnion implementation for pandas' DataFrame by retaining its properties
    """
    def fit_transform(self, X, y=None, **fit_params):
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


    def transform(self, X):
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
