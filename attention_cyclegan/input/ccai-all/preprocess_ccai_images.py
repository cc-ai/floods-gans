from PIL import Image
from pathlib import Path


def convert_to_jpg(src, dst):
    """Convert image located at path `src` into
    a JPEG image located at `dst`
    
    Args:
        src (pathlib.Path or str): image to convert
        dst (pathlib.Path or str): destination of converted image
    """
    im = Image.open(str(src))
    im.save(str(dst))


if __name__ == "__main__":

    p = Path().resolve()
    image_sources = [p / "ln-s_trainA", p / "ln-s_trainB"]
    image_dests = [p / "trainA", p / "trainB"]
    formats = {".JPG", ".jpeg", ".jpg", ".png"}
    target_format = ".png"

    # check image sources and image dests are in reach of script
    try:
        g = set(str(_) for _ in p.glob("*"))
        assert all(
            str(ims) in g for folder in [image_dests, image_sources] for ims in folder
        )
    except AssertionError as e:
        print(e)
        print("Image sources and/or destinations can't be reached")

    # Store already converted images
    existing = set(
        str(imdest / (im.stem + target_format))
        for imdest in image_dests
        for im in imdest.glob("*")
    )

    k = 0
    # For each source / destination pair
    # Typically trainA (various formats) to trainA (only png)
    for i, (imsource, imdest) in enumerate(zip(image_sources, image_dests)):
        images = [im for im in imsource.glob("*") if im.suffix in formats]
        for im in images:
            dst = imdest / (im.stem + target_format)
            # if image is not already converted
            if str(dst) not in existing:
                convert_to_jpg(im.resolve(), dst)
                k += 1
            print("Processed", k, "images", end="\r")
        # Compare source and dest sizes
        original = len(list(imsource.glob("*")))
        converted = len(list(imdest.glob("*")))

        print("\nConverted", converted, "images out of", original)

    print()