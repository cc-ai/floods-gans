An unstructured set of scripts of various utilities, relevance and quality

## For the metadata_script.py :
It compares the downloads folder where you have image folders for all the queries you search and the logs folder which contains json files for the queries and updates the log files with deleted images after manual fine-tuning (deletes the logs for deleted images).

running `python3 metadata_script.py log_dir img_dir` compares the two folders and creates updated log files with names same as original filename in the logs folder(log_dir). log_dir refers to logs folder and img_dir refers to downloads folder generated from running googleimagesdownload command.


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

## Purging data from `googleimagesdownload`

If you've already downloaded and merged runs of `googleimagesdownload` and run a new one, you can purge it: **the script will delete all photos in this new run which are already in the merged/ folder**. This allows you to ignore already downloaded images when going through the new data

```
$ python purge.py run3 run4
```

## Pink to mask

Create a third set of images, `{i}m.png` in addition to `{i}p.png` `{i}f.png` and `{i}.png`. Run with `--source path/to/folder/with/images` ; default is current working directory. For instance: 

```
$ python pink_to_mask.py --source ./data/SimFlood 50-50f-50p
```