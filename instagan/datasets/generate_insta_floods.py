from pathlib import Path
from tqdm import tqdm
import numpy as np
from shutil import copyfile


def process_water_video_db(val_ratio=0.25, domain="A", dest=Path("./insta-floods")):
    name = "[VideoWaterDB] "

    source = Path("./floods-source/VideoWaterDB").resolve()

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


if __name__ == "__main__":

    val_ratio = 0.25
    domain = "A"
    dest = Path("./insta-floods")

    folders = [
        dest / "train{}".format(domain),
        dest / "train{}_seg".format(domain),
        dest / "val{}".format(domain),
        dest / "val{}_seg".format(domain),
    ]
    for folder in folders:
        if not folder.exists():
            folder.mkdir()
        if len(list(folder.glob("**/*"))):
            raise ValueError(str(folder) + " is not empty")

    process_water_video_db(val_ratio, domain, dest)
