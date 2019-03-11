import pathlib
import sys
import uuid

import imageio
from imgaug import augmenters as iaa


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


augmentors = [
    iaa.contrast.LogContrast((0.75, 1.25)),
    iaa.AddElementwise((-25, 25)),
    iaa.flip.Fliplr(1),
    #iaa.Affine(scale=(0.75, 1.25)),
    #iaa.Affine(rotate=(-5, 5)),
    #iaa.Affine(shear=(-5, 5)),
]
seq = iaa.Sequential([sometimes(a) for a in augmentors])

if __name__ == "__main__":

    source = "./"
    dest = None
    formats = [".jpg", ".png", ".jpeg", ".JPG"]
    batch_size = 256
    keep_originals = True

    # as augmentors have randomized parameters and are randomly selected,
    # different sequences will run different modifications
    augmentor_num = 4

    args = sys.argv[1:]
    for a in args:
        if "--dest" in a:
            dest = a.split("=")[-1]
        if "--source" in a:
            source = a.split("=")[-1]
        if "--batch_size" in a:
            batch_size = int(a.split("=")[-1])
        if "--augmentor_num" in a:
            augmentor_num = int(a.split("=")[-1])
        if "--keep_originals" in a:
            keep_originals = bool(a.split("=")[-1])
        if "--help" in a:
            print("--batch_size=256: number of images in memory")
            print("--source=./: path to images to augment")
            print("--dest=./augmented: where to write augmented images")
            print("--augmentor_num=4 number of runs for the sequence of augmentors")
            print("--keep_orginals=True: copy original images in destination folder?")
            sys.exit()

    if not pathlib.Path(dest).exists():
        pathlib.Path(dest).mkdir()

    inp = f"If {dest} is not empty content will be overwritten."
    inp += "\nContinue? (y/n default:y)"

    if "n" in input(inp):
        sys.exit()

    source = pathlib.Path(source)
    if dest is None:
        dest = source / "augmented"
    dest = pathlib.Path(dest)
    imwritten = 0

    impaths = []
    for ext in formats:
        impaths += list(source.glob(f"*{ext}"))

    # split the load in memory depending on the batch_size
    epochs = len(impaths) // batch_size + 1
    for e in range(epochs):
        # load images
        imgs = [
            imageio.imread(str(i))
            for i in impaths[e * batch_size : (e + 1) * batch_size]
        ]

        print(f"Loaded {len(imgs)} images")

        print(f"Augmenting {augmentor_num * len(imgs)} augmented images")
        # process images
        augmented = [seq.augment_images(imgs) for _ in range(augmentor_num)]
        if keep_originals:
            augmented = [imgs] + augmented

        # write images in order image > augmenter
        # imagenumber_augmenternumber.png
        for i in range(len(imgs)):
            for a, aug in enumerate(augmented):
                print("epoch", e, imwritten + 1, end="\r")
                imageio.imwrite(f"{dest}/{i}_{a}.png", aug[i])
                imwritten += 1
