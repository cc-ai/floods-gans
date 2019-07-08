import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import operator


def pink(i):
    return str(i.parent / (i.stem + "p" + i.suffix))


def flood(i):
    return str(i.parent / (i.stem + "f" + i.suffix))


def get_mask(img, rgb=None, upper_bound=100):
    if rgb is None:
        u = defaultdict(int)
        for x in img:
            for y in x:
                u[tuple(y)] += 1
        rgb = list(max(u.items(), key=operator.itemgetter(1))[0])
        print("max occurence: ", rgb)
    diff = np.abs(img - [255, 117, 245])
    mask = (diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2] < upper_bound) * 255
    # mask = np.expand_dims(mask,-1)
    # mask = img*(mask/255).astype(int)
    return mask, rgb


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", help="Where to find and write the images", type=str, default="."
)
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    path = Path(args.source)
    ims = sorted(
        [str(i.parent / str(i.name).replace("f", "")) for i in path.glob("*f.png")]
    )
    ims = [Path(i) for i in ims]

    rgb = None
    upper_bound = 100
    for im in ims:
        print(str(im), end="\r")
        img = cv2.imread(pink(im), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask, rgb = get_mask(img, rgb, upper_bound)
        cv2.imwrite(str(im.parent / (im.stem + "m" + im.suffix)), mask)

    print("\nDone.")
