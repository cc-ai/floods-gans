import cv2
import sys
from pathlib import Path


def new_size(height, width, target):
    mi = min((width, height))
    r = target / mi
    h = max((512, r * height))
    w = max((512, r * width))
    return int(h), int(w)


if __name__ == "__main__":
    args = sys.argv[1:]

    source = None
    dest = None
    size = None

    for a in args:
        if "--dest" in a:
            dest = Path(a.split("--dest=")[-1]).resolve()
        elif "--source" in a:
            source = Path(a.split("--source=")[-1]).resolve()
        elif "--size" in a:
            size = int(a.split("--size=")[-1])

    if source is None:
        print("Specify --source")
        sys.exit()
    if dest is None:
        print("Specify --dest")
        sys.exit()
    if size is None:
        print("Specify --size")
        sys.exit()

    print("Source: ", str(source))
    print("Destination: ", str(dest))
    print("Size: ", size)

    for i, im in enumerate(source.glob("*.png")):
        img = cv2.imread(str(im), cv2.IMREAD_UNCHANGED)
        dim = new_size(img.shape[0], img.shape[1], size)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest / im.name), resized)
        print("\r", i, im, f"{img.shape[:2]} to {dim}", end="")
