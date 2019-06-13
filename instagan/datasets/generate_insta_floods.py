from pathlib import Path
from tqdm import tqdm
import numpy as np
from shutil import copyfile, rmtree
import sys


def copy_images(name, dest, domain, train, train_masks, val, val_masks):
    folder = dest / "train{}".format(domain)
    for i in tqdm(train, desc=name + folder.name):
        copyfile(i, folder / i.name)

    folder = dest / "train{}_seg".format(domain)
    for i in tqdm(train_masks, desc=name + folder.name):
        copyfile(i, folder / i.name)

    folder = dest / "val{}".format(domain)
    for i in tqdm(val, desc=name + folder.name):
        copyfile(i, folder / i.name)

    folder = dest / "val{}_seg".format(domain)
    for i in tqdm(val_masks, desc=name + folder.name):
        copyfile(i, folder / i.name)

    print()
    print()


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
    masks = list((source / "houses_mask").glob("*.png"))
    perm = np.random.permutation(len(images))
    train_size = int(len(perm) * (1 - val_ratio))

    train = [images[i] for i in perm[:train_size]]
    train_masks = [masks[i] for i in perm[:train_size]]
    val = [images[i] for i in perm[train_size:]]
    val_masks = [masks[i] for i in perm[train_size:]]

    assert len(train) == len(train_masks)
    assert len(val) == len(val_masks)

    copy_images(name, dest, domain, train, train_masks, val, val_masks)


def process_210_flooded_houses(val_ratio=0.25, domain="A", dest=Path("./insta-floods")):
    name = "[210_flooded_houses seg]"
    source = Path("/network/tmp1/ccai/data/210_flooded_houses").resolve()

    images = list((source / "imgs_png").glob("*.png"))
    masks = list((source / "masks").glob("*.png"))
    perm = np.random.permutation(len(images))
    train_size = int(len(perm) * (1 - val_ratio))

    train = [images[i] for i in perm[:train_size]]
    train_masks = [masks[i] for i in perm[:train_size]]
    val = [images[i] for i in perm[train_size:]]
    val_masks = [masks[i] for i in perm[train_size:]]

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
    dest = Path("/network/tmp1/ccai/inference_data/instagan/floods_with_waterdb")
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

    process_water_video_db(val_ratio, "A", dest)
    process_210_flooded_houses(val_ratio, "A", dest)
    process_deeplab_segmentation_houses(val_ratio, "B", dest)
    print()
