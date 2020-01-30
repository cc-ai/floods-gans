from pathlib import Path
import re
import subprocess
import datetime
import argparse

template = """\
#!/bin/bash
#SBATCH -c 6
#SBATCH --gres=gpu:titanxp:1
#SBATCH --mem=16G
#SBATCH --partition=unkillable
#SBATCH -o /network/home/schmidtv/mega_depth_sbatch_inferences/slurm-%j.out

module load anaconda/3

source $CONDA_ACTIVATE

conda activate base
conda deactivate
conda activate clouds

cp -r {} $SLURM_TMPDIR

cd /network/home/schmidtv/ccai/github/floods_gans/mega_depth

python infer.py -i {} -o {}

cp -r {} {}


"""


def get_increasable_name(file_path):
    f = Path(file_path)
    while f.exists():
        name = f.name
        s = list(re.finditer(r"--\d+", name))
        if s:
            s = s[-1]
            d = int(s.group().replace("--", "").replace(".", ""))
            d += 1
            i, j = s.span()
            name = name[:i] + f"--{d}" + name[j:]
        else:
            name = f.stem + "--1" + f.suffix
        f = f.parent / name
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_mode", action="store_true")
    opts = parser.parse_args()

    main_out = "/network/home/schmidtv/mega_depth_sbatch_inferences"
    sbatch_name = "inference.sh"
    if not opts.test_mode:
        Path(main_out).mkdir(parents=True, exist_ok=True)

    munit_dataset = Path("/network/tmp1/ccai/data/munit_dataset")

    data = [
        {
            "source": "non_flooded/deeplab_segmented_houses_400/houses_png",
            "dest": "non_flooded/deeplab_segmented_houses_400/pseudo_depth",
        },
        {
            "source": "non_flooded/mapillary/testA",
            "dest": "non_flooded/mapillary/testA_pseudo_depth",
        },
        {
            "source": "non_flooded/mapillary/trainA",
            "dest": "non_flooded/mapillary/trainA_pseudo_depth",
        },
        {
            "source": "non_flooded/streetview_mvp",
            "dest": "non_flooded/streetview_mvp_pseudo_depth",
        },
        {"source": "non_flooded/testA", "dest": "non_flooded/testA_pseudo_depth"},
    ]

    for d in data:
        source = munit_dataset / d["source"]
        dest = munit_dataset / d["dest"]

        if not opts.test_mode:
            assert source.exists()
            dest.mkdir(parents=True, exist_ok=True)

        t = template.format(
            str(source),
            "$SLURM_TMPDIR/" + source.name,
            "$SLURM_TMPDIR/" + dest.name,
            "$SLURM_TMPDIR/" + dest.name,
            str(dest),
        )
        sbfile = get_increasable_name(Path(main_out) / sbatch_name)

        if not opts.test_mode:
            with open(sbfile, "w") as f:
                f.write(t)
            with open(dest / "mega_depth.txt", "w") as f:
                f.write(
                    "Pseudo depth from mega_depth\nsbatch file:{}\ndate:{}".format(
                        str(sbfile), str(datetime.datetime.now())
                    )
                )

            print(sbfile, ":")
            print(subprocess.check_output(f"sbatch {str(sbfile)}", shell=True))
            print()

        else:
            print("Would have written:")
            print(t)
            print("In: {}".format(str(sbfile)))
