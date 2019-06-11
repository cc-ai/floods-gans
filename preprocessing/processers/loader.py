from sklearn.base import BaseEstimator, TransformerMixin
from skimage.io import imread
from pathlib import Path


class Loader(BaseEstimator, TransformerMixin):
    def __init__(self, inputdir):
        self._in = Path(inputdir).resolve()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X["data"] = X["path"].apply(imread)
        return X
