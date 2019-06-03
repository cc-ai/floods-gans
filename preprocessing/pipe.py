import argparse
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
from importlib import reload
import sys
import shutil

import processers as procs

reload(procs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default="./input",
        help="Path to input directory w/ data in input/houses or input/floods/ subdirs",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        choices=["houses", "floods"],
        help="Exclude either input/floods/ or input/houses from processing",
    )
    parser.add_argument(
        "-o", "--output", default="./output", help="Path to output directory"
    )
    parser.add_argument(
        "-f",
        "--format",
        default="png",
        help="Format in which skimage.io.imsave will store final processed images",
    )
    parser.add_argument(
        "-p",
        "--flip",
        default=0.0,
        type=float,
        help="Proportion of inputs to flip l/r (in addition to original image)",
    )
    parser.add_argument(
        "-x",
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite potentially existing output dir",
    )

    args = parser.parse_args()
    print(args)

    dirs = [
        d
        for d in Path(args.input).iterdir()
        if d.is_dir() and (args.exclude is None or args.exclude not in d.name)
    ]

    if not Path(args.output).exists():
        Path(args.output).mkdir()

    for input_dir in dirs:

        print(f"\n### Processing {str(input_dir)} ###")

        out = Path(args.output) / input_dir.name

        if out.exists():
            print("Output destination not empty | overwrite?")
            if args.force or "y" in input("[y/n] : "):
                print("Cleaning", str(out))
                shutil.rmtree(str(out))
                out.mkdir()
            else:
                print("Stopping here")
                sys.exit()
        else:
            out.mkdir()

        pipe = Pipeline(
            steps=[
                ("Loader", procs.Loader(input_dir)),
                ("Flipper", procs.Flipper(args.flip)),
                ("Saver", procs.Saver(out, args.format)),
            ]
        )

        df = pd.DataFrame({"path": [f for f in input_dir.iterdir() if f.is_file()]})
        pipe.fit_transform(df)
