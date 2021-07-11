import re
import pandas as pd
from collections.abc import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

pd.options.mode.chained_assignment = None

__all__ = ['FeatureSelector', 'DictionaryVectorizer', 'TopFeatures',
           'SumTransformer', 'Binarizer', 'DateTransformer',
           'ItemCounter', 'MeanTransformer', 'Identity', 'FeatureRename']


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Identity(BaseEstimator, TransformerMixin, metaclass=Singleton):
    """
        An identity transformer.
    """
    def fit(self, X, y=None):
        self.fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, ['fitted'])
        return X


class FeatureRename(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.columns = getattr(X, 'columns', None)
        if self.columns is None:
            return self
        self.rename_mappings = {}
        for col in self.columns:
            if col.endswith("_prep"):
                self.rename_mappings[col] = col[:-5]
        return self

    def transform(self, X):
        check_is_fitted(self, ['columns', 'rename_mappings'])
        return X.rename(columns=self.rename_mappings)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
        This transformer is really straightforward - it simply takes the name
        of the column we want to extract and if we use it, it will 'spit out'
        the data column of our Data Frame.
    """
    def __init__(self, features):
        if callable(features):
            self.features = features
        elif isinstance(features, Sequence):
            self.features = features
        else:
            raise ValueError("Invalid features attribute")
        self.features = features

    def fit(self, X, y=None):
        if callable(self.features):
            feature_names = self.features(X)
            assert all(map(lambda val: val in X.columns, feature_names))
        elif isinstance(self.features, Sequence):
            assert all(map(lambda val: val in X.columns, self.features))
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        if callable(self.features):
            feature_names = self.features(X)
        elif isinstance(self.features, Sequence):
            feature_names = self.features
        return X[feature_names]


class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    """
        This one is a bit more complex. It's role is to:
            1st - extract values from dictionaries,
            2nd - join them in one string,
            3rd - dummify it using sklearn Count Vectorizer.
    """

    def __init__(self, key, all_=True):
        self.key = key
        self.all_ = all_

    @staticmethod
    def extract_items(list_, key, all_=True):
        def sub(x):
            return re.sub(r'[^A-Za-z0-9]', '_', x)
        if all_:
            target = []
            for dict_ in eval(list_):
                target.append(sub(dict_[key].strip()))
            return ' '.join(target)
        elif not eval(list_):
            return 'no_data'
        return sub(eval(list_)[0][key].strip())

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        genres = X.apply(lambda x: self.extract_items(x, self.key, self.all_))
        self.vectorizer = CountVectorizer().fit(genres)
        self.columns = self.vectorizer.get_feature_names()
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        genres = X.apply(lambda x: self.extract_items(x, self.key))
        data = self.vectorizer.transform(genres)
        return pd.DataFrame(data.toarray(),
                            columns=self.vectorizer.get_feature_names(),
                            index=X.index)


class TopFeatures(BaseEstimator, TransformerMixin):
    """
        This transformer expects dummified data set and extract
        most popular features.
    """

    def __init__(self, percent):
        if percent > 100:
            self.percent = 100
        else:
            self.percent = percent

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        counts = X.sum().sort_values(ascending=False)
        index_ = int(counts.shape[0]*self.percent/100)
        self.columns = counts[:index_].index
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X[self.columns]


class SumTransformer(BaseEstimator, TransformerMixin):
    """
        Sum Transformer simply computes a sum across given features.
        We'll use it on our sparse data (after dummification).
    """
    def __init__(self, series_name):
        self.series_name = series_name

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X.sum(axis=1).to_frame(self.series_name)


class Binarizer(BaseEstimator, TransformerMixin):
    """
        Biniarizer takes as an input function that decides whether or not label
        value as True or False.
    """

    def __init__(self, condition, name):
        self.condition = condition
        self.name = name

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X.apply(lambda x: int(self.condition(x))).to_frame(self.name)


class DateTransformer(BaseEstimator, TransformerMixin):
    """
        This transformer takes a date in string format and extract values of
        interest.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        date = pd.to_datetime(X)
        year = date.dt.year.rename("year")
        month = date.dt.month.rename("month")
        day = date.dt.year.rename("day")
        return pd.concat([year, month, day], axis=1)


class ItemCounter(BaseEstimator, TransformerMixin):
    """
        It counts how many items are in a list.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        def get_list_len(list_):
            return len(eval(list_))
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X.apply(lambda x: int(get_list_len(x)))


class MeanTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = check_array(X)
        if X.shape[1] != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X.mean(axis=1).to_frame(self.name)


class DFStandardScaler(StandardScaler):

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        index = getattr(X, 'index', None)
        columns = getattr(X, 'columns', None)
        X = super().transform(X, copy)
        if columns is not None and index is not None:
            X = pd.DataFrame(X, columns=columns, index=index)
        return X


if __name__ == '__main__':
    check_estimator(Identity())
    check_estimator(FeatureSelector())
    check_estimator(TopFeatures())
    check_estimator(DictionaryVectorizer())
    check_estimator(SumTransformer())
    check_estimator(Binarizer())
    check_estimator(DateTransformer())
    check_estimator(MeanTransformer())
    check_estimator(ItemCounter())
