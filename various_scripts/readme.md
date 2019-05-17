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

### Example query

```
googleimagesdownload --keywords "flooded houses" --limit 1000 --extract_metadata -cd /usr/local/bin/chromedriver
```

To download more than 100 images at once, you should install `chromedriver` which is easy on both mac and linux.

```
# mac
$ brew cask install chromedriver
```