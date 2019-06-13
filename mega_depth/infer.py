import datetime
import sys
from pathlib import Path
import shutil
import zipfile
import os

import numpy as np
from skimage import io
from skimage.transform import resize

import cv2
import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
from torch.autograd import Variable

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


if __name__ == "__main__":

    inf = Path() / "inference"
    if not inf.exists():
        inf.mkdir()

    date = str(datetime.datetime.now())[:19]
    output_folder = inf / date
    output_folder.mkdir()

    print("Infering in ", str(output_folder))
    print()

    input_folder = Path() / "input"

    model = create_model(opt)

    input_height = 384
    input_width = 512

    to_load = [t.resolve() for t in input_folder.iterdir() if t.is_file()]
    to_load_str = list(map(str, to_load))
    print("============================= TEST ============================")
    model.switch_to_eval()

    for i, img_path in enumerate(to_load_str):

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
