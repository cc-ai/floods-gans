from sklearn.base import BaseEstimator, TransformerMixin
from skimage.io import imread
from pathlib import Path
import numpy as np
import pandas as pd


class Flipper(BaseEstimator, TransformerMixin):
    def __init__(self, ratio):
        self.ratio = ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__, self.ratio)
        if self.ratio <= 0:
            return X
        tmp = X.sample(frac=self.ratio)
        tmp["data"] = tmp["data"].apply(np.fliplr)
        tmp["path"] = tmp["path"].apply(
            lambda p: p.parent / (p.name + "_flipped" + p.suffix)
        )
        return pd.concat([X, tmp], axis=0)
