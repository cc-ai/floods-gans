# Preprocessing the data

## Outline

`pipe.py` runs a pipeline (`sklearn.pipeline.Pipeline`) of preprocessing steps. Each of these steps is defined in `processers/`.

Running `python pipe.py {args}` will run sequentially each of the preprocessing steps in order.

Current processers are:

* Loader: reads image arrays in memory for downstream use by other processers
* Flipper: flips left/right a ratio (`0 <= r <= 1`) of the data 
* Saver: saves processed data into designated format

The input data to the pipleine should be structured like:

```
.
├── data
│   ├── floods
│   └── houses
```

## Usage

```
usage: pipe.py [-h] [-i INPUT] [-e {houses,floods}] [-o OUTPUT] [-f FORMAT]
               [-p FLIP] [-x]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input directory w/ data in input/houses or
                        input/floods/ subdirs
  -e {houses,floods}, --exclude {houses,floods}
                        Exclude either input/floods/ or input/houses from
                        processing
  -o OUTPUT, --output OUTPUT
                        Path to output directory
  -f FORMAT, --format FORMAT
                        Format in which skimage.io.imsave will store final
                        processed images
  -p FLIP, --flip FLIP  Proportion of inputs to flip l/r (in addition to
                        original image)
  -x, --force           Force overwrite potentially existing output dir
```

For instance:

```
$ python pipe.py -i /path/to/data -o /path/to/output -f jpg -p 0.3 --force
```

This will convert all images in `data/floods` and `data/houses` into `.jpg` format in the `output/floods` and `output/houses` directories. 30% (random) of the images of each domain will be flipped left/right. If there already is an `output/` folder it will be entirely overwritten and pre-existing data will be lost (`--force`).

## Add Processer

### 1. Customize base

In a new file `processers/myproc.py`:

```python
from sklearn.base import BaseEstimator, TransformerMixin


class MyProcesser(BaseEstimator, TransformerMixin):
    def __init__(self, somearg):
        self.arg = somearg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = dosomethingwithX(X)
        return X

```

### 2 Add it to processers/__init__.py

```python
from .flipper import Flipper
from .saver import Saver
from .loader import Loader
from .myproc import MyProcesser
```

### Add it to the `Pipeline`

Set appropriate flag if the processer is optional/configurable

To make it optional, you can just pass an argument in its `__init__(self, ...)` function and do something like this in `transform`:

```python
def transform(self, X, y=None):
    if self.ignore_from_flags:
        return X
```

And finally add it to the pipeline steps in `pipe.py`. 

*Put it **before** the `Saver`*

```python
Pipeline(
    steps=[
        ("Loader", procs.Loader(input_dir)),
        ("Flipper", procs.Flipper(args.flip)),
        ("MyProcesser", procs.MyProcesser(yourArgs)),
        ("Saver", procs.Saver(out, args.format)),
    ]
)
```

