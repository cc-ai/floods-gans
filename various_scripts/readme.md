An unstructured set of scripts of various utilities, relevance and quality

## Merging data from `googleimagesdownload`

-> ref: https://github.com/hardikvasa/google-images-download

folder structure:

```
.
├── run1
│   ├── downloads
│   │   └── query1.1
│   └── logs
├── run2
│   ├── downloads
│   │   └── query2.1
│   └── logs
└── merged
    ├── downloads
    └── logs
```

running `$ python merge.py` creates a `merged` folder with unique images from runs+queries based on logs.

uniqueness baset on image url (`image_link` field in the `json` log)