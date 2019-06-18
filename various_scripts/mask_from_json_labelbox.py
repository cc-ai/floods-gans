import tqdm as tq
import json
import requests
import os
import argparse
from pathlib import Path
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("fname", type=str, default="export-2019-06-12T18_33_17.792Z.json")
args = parser.parse_args()

with open(args.fname, "r") as f:
    x = json.load(f)

# create destination directories
dest = Path("./")
if not (dest / "masks").exists():
    (dest / "masks").mkdir()
if not (dest / "imgs").exists():
    (dest / "imgs").mkdir()
# find already downloaded imgs + masks
existing_masks = set(f.stem for f in (dest / "masks").iterdir() if f.is_file())
existing_imgs = set(f.stem for f in (dest / "imgs").iterdir() if f.is_file())

# img = Path("/Users/Directory/image323.png")
# img.stem -> image323

# Download the masks
for k in tq.tqdm(range(len(x))):
    p = x[k]
    mask_name = Path(p["External ID"]).stem + "_0"
    if p["Label"] != "Skip" and mask_name not in existing_masks:
        masks = p["Masks"]
        name = p["External ID"]
        filename, file_extension = os.path.splitext(name)
        for m, l in masks.items():
            r = requests.get(l, allow_redirects=False)
            open("./masks/" + filename + "_0.png", "wb").write(r.content)

# Download the imgs
for k in tq.tqdm(range(len(x))):
    p = x[k]
    if p["Label"] != "Skip" and Path(p["External ID"]).stem not in existing_imgs:
        labeled_data = p["Labeled Data"]
        name = p["External ID"]
        r = requests.get(labeled_data, allow_redirects=False)
        open("./imgs/" + name, "wb").write(r.content)


# Convert the image in png format
list_img = os.listdir("./imgs/")
for img_path in tq.tqdm(range(len(list_img))):
    filename, file_extension = os.path.splitext(list_img[img_path])
    img_ = imageio.imread("./imgs/" + list_img[img_path])
    imageio.imsave("./imgs/" + filename + ".png", img_)
