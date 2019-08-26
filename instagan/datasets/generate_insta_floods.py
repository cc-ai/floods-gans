from pathlib import Path
from tqdm import tqdm
import numpy as np
from shutil import copyfile, rmtree
import sys
import os


def copy_images(name, dest, domain, train, train_masks, val, val_masks):
    folder = dest / "train{}".format(domain)
    for i in tqdm(train, desc=name + folder.name):
        copyfile(i, folder / i.name)

    folder = dest / "train{}_seg".format(domain)
    for i in tqdm(train_masks, desc=name + folder.name):
        target = i.stem
        if not target.endswith("_0"):
            target += "_0"
        copyfile(i, folder / (target + i.suffix))

    folder = dest / "val{}".format(domain)
    for i in tqdm(val, desc=name + folder.name):
        copyfile(i, folder / i.name)

    folder = dest / "val{}_seg".format(domain)
    for i in tqdm(val_masks, desc=name + folder.name):
        target = i.stem
        if not target.endswith("_0"):
            target += "_0"
        copyfile(i, folder / (target + i.suffix))

    print()
    print()


def check(dest):
    # sanity check
    for time in ["train", "val"]:
        for domain in ["A", "B"]:
            imgs = dest / (time + domain)
            msks = dest / (time + domain + "_seg")
            imgs = set(i.stem + "_0" + i.suffix for i in imgs.iterdir())
            msks = set(i.name for i in msks.iterdir())
            try:
                assert len(imgs - msks) == 0
            except AssertionError:
                for i in imgs:
                    if i not in msks:
                        print("Mask not found: ", i)
                for i in msks:
                    if i not in imgs:
                        print("Image not found: ", i)


def process_water_video_db(val_ratio=0.25, domain="A", dest=Path("./insta-floods")):
    name = "[video_water_database] "

    source = Path("/network/tmp1/ccai/data/video_water_database").resolve()

    domains = ["canal", "fountain", "lake", "ocean", "pond", "river", "stream"]
    images = list((source / "images").glob("*.png"))
    data = {d: [] for d in domains}
    for i in tqdm(images, desc=name + "building list"):
        d = i.stem.split("_")[0]
        data[d].append(i)

    perms = {d: np.random.permutation(list(range(len(data[d])))) for d in domains}
    train_lengths = {d: int(len(perms[d]) * (1 - val_ratio)) for d in domains}
    train = {d: [data[d][i] for i in perms[d][: train_lengths[d]]] for d in domains}
    val = {d: [data[d][i] for i in perms[d][train_lengths[d] :]] for d in domains}

    train = [i for d in train.values() for i in d]
    train_masks = [source / "masks" / i.name for i in train]
    val = [i for d in val.values() for i in d]
    val_masks = [source / "masks" / i.name for i in val]

    assert len(train) == len(train_masks)
    assert len(val) == len(val_masks)

    copy_images(name, dest, domain, train, train_masks, val, val_masks)


def process_deeplab_segmentation_houses(
    val_ratio=0.25, domain="B", dest=Path("./insta-floods")
):
    name = "[deepleab seg]"
    source = Path("/network/tmp1/ccai/data/deeplab_segmented_houses").resolve()

    images = list((source / "houses_png").glob("*.png"))
    perm = np.random.permutation(len(images))
    train_size = int(len(perm) * (1 - val_ratio))

    train = [images[i] for i in perm[:train_size]]
    train_masks = [source / "houses_mask" / i.name for i in train]
    val = [images[i] for i in perm[train_size:]]
    val_masks = [source / "houses_mask" / i.name for i in val]

    assert len(train) == len(train_masks)
    assert len(val) == len(val_masks)

    copy_images(name, dest, domain, train, train_masks, val, val_masks)


def process_210_flooded_houses(val_ratio=0.25, domain="A", dest=Path("./insta-floods")):
    name = "[210_flooded_houses seg]"
    source = Path("/network/tmp1/ccai/data/210_flooded_houses").resolve()

    images = list((source / "imgs_png").glob("*.png"))
    perm = np.random.permutation(len(images))
    train_size = int(len(perm) * (1 - val_ratio))

    train = [images[i] for i in perm[:train_size]]
    train_masks = [source / "masks" / (i.stem + "_0" + i.suffix) for i in train]
    val = [images[i] for i in perm[train_size:]]
    val_masks = [source / "masks" / (i.stem + "_0" + i.suffix) for i in val]

    assert len(train) == len(train_masks)
    assert len(val) == len(val_masks)

    copy_images(name, dest, domain, train, train_masks, val, val_masks)

def process_280_flooded_houses(val_ratio=0.25, domain="A", dest=Path("./insta-floods")):
    name = "[280_flooded_houses seg]"
    source = Path("/network/tmp1/ccai/data/280_flooded_houses").resolve()

    images = list((source / "imgs_png").glob("*.png"))
    perm = np.random.permutation(len(images))
    train_size = int(len(perm) * (1 - val_ratio))

    train = [images[i] for i in perm[:train_size]]
    train_masks = [source / "masks" / (i.stem + "_0" + i.suffix) for i in train]
    val = [images[i] for i in perm[train_size:]]
    val_masks = [source / "masks" / (i.stem + "_0" + i.suffix) for i in val]

    assert len(train) == len(train_masks)
    assert len(val) == len(val_masks)

    copy_images(name, dest, domain, train, train_masks, val, val_masks)


def mkdirs(dest):
    if not dest.exists():
        dest.mkdir()

    folders = []
    for domain in ["A", "B"]:
        folders += [
            dest / "train{}".format(domain),
            dest / "train{}_seg".format(domain),
            dest / "val{}".format(domain),
            dest / "val{}_seg".format(domain),
        ]
    warn = False
    for folder in folders:
        if not folder.exists():
            folder.mkdir()
        if len(list(folder.glob("**/*"))):
            warn = True
            print(str(folder) + " is not empty")
    return warn


if __name__ == "__main__":

    val_ratio = 0.15
    dest = Path("/network/tmp1/ccai/inference_data/instagan/depth_floods_280")
    np.random.seed(123)

    warn = mkdirs(dest)

    if warn:
        print("Some folders are not empty")
        inp = input("Overwrite (o), Abort (a, default) or Continue and ignore? (c) : ")
        if "o" in inp:
            rmtree(str(dest))
            _ = mkdirs(dest)
        elif "c" in inp:
            pass
        else:
            print("Aborting")
            sys.exit()

    # process_water_video_db(val_ratio, "A", dest)
    # process_210_flooded_houses(val_ratio, "A", dest)
    process_280_flooded_houses(val_ratio, "B", dest)
    process_deeplab_segmentation_houses(val_ratio, "A", dest)
    print()
    check(dest)
