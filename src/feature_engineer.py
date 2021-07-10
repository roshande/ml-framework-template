import numpy as np
import pandas as pd
import math
import numbers
import warnings
from itertools import combinations
from numpy.random import uniform
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

__all__ = ['CatCombine', 'CatQuantCombine', 'Clustering', 'Binning']


def engineer_categorical_features(data, cat_columns=None, max_comb=None):
    if cat_columns is None:
        cat_columns = data.select_dtypes(exclude="number").columns
    ret_df = pd.DataFrame()
    for idx, col in enumerate(cat_columns):
        ret_df[col] = data[col]+ str(idx)
    max_comb = len(cat_columns)+1 if max_comb is None else max_comb
    for r in range(2, max_comb):
        for comb in combinations(cat_columns, r):
            col_name = "+".join(comb)
            ret_df[col_name] = ret_df[list(comb)].apply(lambda row: "+".join(row.values), axis=1)

    new_cols = ret_df.columns[~ret_df.columns.isin(cat_columns)]
    return ret_df[new_cols]


class CatCrosses(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features=None):
        assert cat_features is None or len(cat_features) > 1, "Combine more than one features"
        self.cat_features = cat_features

    def fit(self, X, y=None):
        self.input_shape_ = X.shape[1]
        return self

    def __colname(self):
        return "+".join(self.cat_features)

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        if X.shape[1] != self.input_shape_:
            raise ValueError("Shape of input is different from what was seen"
                             "in `fit`")
        if self.cat_features is None:
            self.cat_features = X.columns.to_list()
        join_values = lambda row: "+".join([row[feat]+str(idx) for idx, feat in enumerate(self.cat_features)])
        colname = self.__colname()
        return X[self.cat_features].apply(join_values, axis=1).to_frame(colname)


class CatQuantCrosses(BaseEstimator, TransformerMixin):
    def __init__(self, cats, quants, mappers=['mean', 'median']):
        self.cats = cats
        self.quants = quants
        self.mappers = mappers

    def fit(self, X, y=None):
        self.input_shape_ = X.shape[1]
        self.params = self.learn_params(X)
        return self

    def __colname(self):
        return "+".join(self.cats)

    def combine_cats(self, X):
        join_values = lambda row: "+".join([row[feat]+str(idx) for idx, feat in enumerate(self.cats)])
        colname = self.__colname()
        return X[self.cats].apply(join_values, axis=1).to_frame(colname)

    def learn_params(self, X):
        new_df = self.combine_cats(X)
        new_df = pd.concat([new_df, X[self.quants]], axis=1)
        colname = self.__colname()
        agg_features = new_df.groupby(colname).agg({quant: self.mappers for quant in self.quants})
        agg_features.columns = [colname + '_'.join(c).strip("_") for c in agg_features.columns]
        return agg_features

    def transform(self, X):
        check_is_fitted(self, ['input_shape_', 'params'])
        if X.shape[1] != self.input_shape_:
            raise ValueError("Shape of input is different from what was seen"
                             "in `fit`")
        new_df = self.combine_cats(X)
        colname = self.__colname()
        ret = pd.merge(new_df[colname], self.params, on=[colname], how='left')
        ret.index = X.index
        if len(self.cats) == 1:
            ret.drop(columns=colname, inplace=True)
        return ret


class PolyTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, transformation=np.log10, suffix="_log"):
        self.features = features
        self.transformation = transformation
        self.suffix = suffix

    def fit(self, X, y=None):
        if self.features is None:
            self.features = X.columns
        elif not all(map(lambda feat: feat in X.columns, self.features)):
            raise ValueError("Not all features present {}".format(self.features))
        return self

    def transform(self, X):
        fitted = self.transformation(X[self.features]+1)
        fitted = pd.DataFrame(fitted, index=X.index, columns=X.columns)
        fitted = fitted.add_suffix(self.suffix)
        return fitted


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        common = np.intersect1d(self.features_to_drop, X.columns)
        self.features_to_drop = common
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop)


class Clustering(BaseEstimator, TransformerMixin):
    def __init__(self, name, clus_algo):
        self.clus_algo = clus_algo
        self.name = name

    def hopkins(self, X):
        d = X.shape[1]
        #d = len(vars) # columns
        #n = len(X) # rows
        n = X.shape[0]
        m = int(0.1 * n)
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)

        rand_X = sample(range(0, n, 1), m)

        ujd = []
        wjd = []
        for j in range(0, m):
            u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0),np.amax(X, axis=0), d).reshape(1, -1),
                                        2, return_distance=True)
            ujd.append(u_dist[0][1])
            w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].to_numpy().reshape(1, -1), 2, return_distance=True)
            wjd.append(w_dist[0][1])

        H = sum(ujd) / (sum(ujd) + sum(wjd))
        if math.isnan(H):
            #print(ujd, wjd)
            H = 0
        return H

    def fit(self, X, y=None):
        self.columns = getattr(X, 'columns', None)
        X = check_array(X)
        #hopkin_score = self.hopkins(X)
        self.clus_algo.fit(X)
        self.input_shape_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_', 'columns'])
        index = getattr(X, 'index', None)
        X = check_array(X)
        if self.input_shape_ != X.shape[1]:
            raise ValueError("Shape of input is different from what was seen"
                             "in `fit`")
        clus = self.clus_algo.predict(X)
        if index is not None:
            clus = pd.Series(clus, index=index, name=self.name, dtype="category").to_frame()
        return clus


class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, bins, labels=None, suffix="_bin", right=True):
        self.bins = bins
        self.suffix = suffix
        self.right = right
        if not (labels is None or any(map(lambda val: isinstance(val, numbers.Number), labels))):
            warnings.warn("Ordinality of data missing by the labels `{}`".format(labels))
        self.labels = labels

    def fit(self, X, y=None):
        if X.shape[1] != 1:
            raise ValueError("Binning with only 1-dimensional data")
        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        if X.shape[1] != 1:
            raise ValueError("Binning with only 1-dimensional data")
        trans = pd.cut(X.iloc[:,0], self.bins, self.right, self.labels, ordered=True)
        return trans.to_frame().add_suffix(self.suffix)

