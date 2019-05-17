from pathlib import Path
import json
import os
import shutil

if __name__ == "__main__":

    dest = Path() / "merged"

    if dest.exists():
        if "y" in input(
            "There's already a merged folder, OVERWRITE AND DELETE? [y/n default:n]\n"
        ):
            shutil.rmtree(str(dest))
        else:
            os.exit()

    dest.mkdir()

    folders = list(  # relevant folders contain a downloads/ and logs/ subfolder
        [
            p.resolve()
            for p in Path().iterdir()
            if p.is_dir()
            and all(
                f in set(_.name for _ in p.iterdir() if _.is_dir())
                for f in ["logs", "downloads"]
            )
            and str(p) != str(dest)  # don't include merged destination
        ]
    )

    new_logs = dest / "logs"
    new_logs.mkdir()

    new_downloads = dest / "downloads"
    new_downloads.mkdir()

    refs = set()
    images = []
    total = 0

    for f in folders:
        downloads = f / "downloads"
        logs = f / "logs"
        jsons = logs.glob("*.json")

        for j in jsons:  # may have several terms in one search
            with j.open("r") as jfile:
                images_logs = json.load(jfile)
                total += len(images_logs)

                for log in images_logs:  # itererate over downloaded images
                    local_image_path = (
                        downloads
                        / j.name.split(".json")[0].replace("_", " ")
                        / log["image_filename"]
                    )  # find where the actual image is in filesystem
                    if log["image_link"] not in refs:
                        # don't repeat images based on url
                        refs.add(log["image_link"])
                        log["local_image_path_str"] = str(local_image_path)
                        images.append(log)

    print("Listed")

    for i, log in enumerate(images):
        new_name = "{}. ".format(i) + ". ".join(log["image_filename"].split(". ")[1:])
        log["original_image_filename"] = log["image_filename"]
        log["image_filename"] = new_name
        shutil.copy(log["local_image_path_str"], new_downloads / new_name)

    print("Copied")

    new_json = new_logs / "merged-{}.json".format(
        "-".join(list(f.name for f in folders))
    )

    with new_json.open("w") as njf:
        json.dump(images, njf)

    print(
        "Copied {} images, {} were ignored (repeated image)".format(
            len(images), total - len(images)
        )
    )

    print("Updated logs")
