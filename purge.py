from pathlib import Path
import json
import sys

# If there are already merged folders, compares the directories given as arguments
# (called targets) to the merged/ folder
# it is used as a reference and images already present in it are deleted from the
# targets

# sample log instance:
# {'image_description': 'Balkans Struck by Worst Flooding in 120 Years - The Atlantic',
#  'image_filename': '0. main_900.jpg',
#  'image_format': 'jpg',
#  'image_height': 599,
#  'image_host': 'theatlantic.com',
#  'image_link': 'https://cdn.theatlantic.com/assets/media/img/photo/2014/05/balkans-struck-by-worst-flooding-in-120-years/f01_RTR3PGKK/main_900.jpg?1420504073',
#  'image_source': 'https://www.theatlantic.com/photo/2014/05/balkans-struck-by-worst-flooding-in-120-years/100739/',
#  'image_thumbnail_url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4keC8Ic1EQbqit-n0zk5XPZlBIcbSNr5o1simuMP63hBIeg5O',
#  'image_width': 900,
#  'local_image_path_str': '/Users/victor/Documents/Experiments/flooded_houses/d1/downloads/flooded houses/25. main_900.jpg',
#  'original_image_filename': '25. main_900.jpg'}

# usage: $ python purge.py run3 run4

if __name__ == "__main__":

    p = Path().resolve()

    merged = p / "merged"
    if not merged.exists():
        print("No pre-existing merged folder ; stopping here no purge to be done")
        sys.exit()

    targets = sys.argv[1:]
    for target in targets:
        # target contains the output of a googleimagesdownload run
        target = Path(target)
        assert target.exists() and target.is_dir()
        downloads = target / "downloads"

        logs = (merged / "logs").glob("**/*.json")
        refs = sum((json.load(f.open("r")) for f in logs), [])
        refs = set(r["image_link"] for r in refs)
        # now refs is a set of image_links in merged/

        target_logs = (target / "logs").glob("**/*.json")
        target_refs = sum((json.load(f.open("r")) for f in target_logs), [])
        target_refs = {r["image_filename"]: r for r in target_refs}
        # now target_refs is a dictionnary mapping filenames to log instances

        deleting = 0

        for query in downloads.iterdir():
            if not query.is_dir():
                # ignore stuff like .DS_STORE on macs
                continue
            for im in query.iterdir():
                # for each image in the query
                if im.is_file() and im.name in target_refs:
                    # check that it is referenced in the target logs
                    # (failsafe if there are other unexpected directories
                    # when iterating over downloads)
                    if target_refs[im.name]["image_link"] in refs:
                        # delete the image if its image_link is in the ref
                        im.unlink()
                        deleting += 1
                        print(f"Deleting {im.name} ({deleting} total)")
