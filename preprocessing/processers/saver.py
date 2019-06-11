from sklearn.base import BaseEstimator, TransformerMixin
from skimage.io import imsave
from pathlib import Path


class Saver(BaseEstimator, TransformerMixin):
    def __init__(self, outputdir, out_format):
        self._out = Path(outputdir)
        self._format = out_format if not out_format.startswith(".") else out_format[1:]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.__class__.__name__)
        X.apply(
            lambda row: imsave(
                self._out / (row["path"].stem + "." + self._format), row["data"]
            ),
            axis=1,
        )
        return X
