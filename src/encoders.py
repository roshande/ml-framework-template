import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import pandas as pd

__all__ = ['OneHotEncoder_custom']

class OneHotEncoder(ce.OneHotEncoder):
    pass

class OneHotEncoder_custom(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_temp = pd.get_dummies(X, drop_first=True)
        self.ret_cols = X_temp.columns
        return self

    def transform(self, X):
        check_is_fitted(self, ['ret_cols'])
        X_temp = pd.get_dummies(X, drop_first=False)
        return X_temp[self.ret_cols]

