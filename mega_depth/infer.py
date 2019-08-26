import argparse
import datetime
import os
import shutil
import sys
import zipfile
from pathlib import Path

import addict
import cv2
import numpy as np
import torch
from numpy import inf
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions

# opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
opt = addict.Dict(
    {
        "batchSize": 1,
        "loadSize": 286,
        "fineSize": 256,
        "input_nc": 3,
        "output_nc": 3,
        "ngf": 64,
        "ndf": 64,
        "which_model_netG": "unet_256",
        "gpu_ids": [0, 1],
        "name": "test_local",
        "model": "pix2pix",
        "nThreads": 2,
        "checkpoints_dir": "./checkpoints/",
        "norm": "instance",
        "serial_batches": False,
        "display_winsize": 256,
        "display_id": 1,
        "identity": 0.0,
        "use_dropout": False,
        "max_dataset_size": inf,
        "display_freq": 100,
        "print_freq": 100,
        "save_latest_freq": 5000,
        "save_epoch_freq": 5,
        "continue_train": False,
        "phase": "train",
        "which_epoch": "latest",
        "niter": 100,
        "niter_decay": 100,
        "beta1": 0.5,
        "lr": 0.0002,
        "no_lsgan": False,
        "lambda_A": 10.0,
        "lambda_B": 10.0,
        "pool_size": 50,
        "no_html": False,
        "no_flip": False,
        "isTrain": True,
    }
)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-w", "--web", action="store_true", default=False)
    args = parser.parse_args()

    HTML = args.web

    print(args)

    inf = Path() / "inference"
    if not inf.exists():
        inf.mkdir()

    date = str(datetime.datetime.now())[:19]

    input_folder = (
        Path(args.input)
        if args.input is not None
        else (
            Path("/network/tmp1/ccai/inference_data/instagan/floods_with_waterdb")
            / "valA"
        )
    )

    # output_folder = inf / date
    output_folder = (
        Path(args.output)
        if args.output is not None
        else (
            Path("/network/tmp1/ccai/inference_data/instagan/floods_with_waterdb")
            / "valA_depth"
        )
    )
    output_folder.mkdir(exist_ok=True)

    print("Infering in ", str(output_folder))
    print()

    model = create_model(opt)

    input_height = 384
    input_width = 512

    to_load = [t.resolve() for t in input_folder.iterdir() if t.is_file()]
    to_load_str = list(map(str, to_load))
    print("============================= TEST ============================")
    model.switch_to_eval()

    for i, img_path in tqdm(enumerate(to_load_str)):

        # print(img_path, end="\r")
        read_image = io.imread(img_path)
        if len(read_image.shape) == 1:
            if len(read_image) == 2:
                read_image = read_image[0]
            else:
                print("Error at step", i, "for image", img_path)
                break

        img = np.float32(read_image) / 255.0
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGRA2BGR)
        img = resize(img, (input_height, input_width), order=1)
        input_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda())
        pred_log_depth = model.netG.forward(input_images)
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
        pred_inv_depth = 1 / pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)

        io.imsave(output_folder / (to_load[i].stem + ".png"), pred_inv_depth)
        # print(pred_inv_depth.shape)

    if HTML:
        src = output_folder / "source"
        src.mkdir()
        html = '<html lang="en"><head><meta charset="utf-8"><style>{}</style>\n</head>'
        html += '<body><div id="all">{}</div></body></html>'
        imgs = ""
        for im in to_load[:i]:
            imgs += "<div class='comparison'><img src='{}'><img src='source/{}'></div>\n".format(
                im.stem + ".png", im.name
            )
            shutil.copy(str(im), str(src / im.name))
        style = """
            img {
                width: 50%
            }
            .comparison {
                margin: 20px
            }
        """
        html = html.format(style, imgs)
        with (output_folder / "view.html").open("w") as f:
            f.write(html)

        zipf = zipfile.ZipFile(inf / f"{date}.zip", "w", zipfile.ZIP_DEFLATED)
        zipdir(str(output_folder), zipf)
        zipf.close()
