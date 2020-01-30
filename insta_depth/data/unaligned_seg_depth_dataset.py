import os.path
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class UnalignedSegDepthDataset(BaseDataset):
    def name(self):
        return "UnalignedSegDepthDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @staticmethod
    def center_to_unit_ball(array):
        x = array.max()
        n = array.min()
        a = 2 / (x - n)
        b = -(n + x) / (x - n)
        return a * array + b

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        self.max_instances = 20  # default: 20
        self.seg_dir = "seg"  # default: 'seg'
        self.depth_dir = "depth"

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def fixed_transform(self, image, seed):
        random.seed(seed)
        return self.transform(image)

    def read_segs(self, seg_path, seed):
        segs = list()
        #             print(seg_path,"seg_path")
        #             print(self.max_instances,"self.max_instances")
        for i in range(self.max_instances):
            path = seg_path.replace(".png", "_{}.png".format(i))
            # print(os.path.isfile(path),"cndsjbce")
            #                 print(path,"12")
            if os.path.isfile(path):
                #                     print("1")
                seg = Image.open(path).convert("L")
                seg = self.fixed_transform(seg, seed)
                segs.append(seg)
            else:
                #                     print("2")
                #                     print(path," vnf")
                segs.append(-torch.ones(segs[0].size()))
        return torch.cat(segs)

    # 	def read_segs(self, seg_path, seed):
    #             segs = list()
    #             #print(seg_path)
    #             for i in range(self.max_instances):
    #                 path = seg_path.replace('_seg/', '_seg/0_')
    # #                 print(path,"bansal")
    #                 #print(os.path.isfile(path),"cndsjbce")
    #                 if os.path.isfile(path):
    # #                     print("0")
    #                     seg = Image.open(path).convert('L')
    # #                     print(seg,"1")
    #                     seg = self.fixed_transform(seg, seed)
    # #                     print(seg,"2")
    #                     segs.append(seg)
    #                 else:
    # #                     print("3")
    #                     segs.append(-torch.ones(segs[0].size()))
    #             return torch.cat(segs)

    def __getitem__(self, index):
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        A_path = Path(self.A_paths[index_A])
        B_path = Path(self.B_paths[index_B])
        A_seg_path = (
            A_path.parent.parent
            / A_path.parent.name.replace("A", "A_seg")
            / A_path.name
        )
        B_seg_path = (
            B_path.parent.parent
            / B_path.parent.name.replace("B", "B_seg")
            / B_path.name
        )
        A_depth_path = (
            A_path.parent.parent
            / A_path.parent.name.replace("A", "A_depth")
            / A_path.name
        )
        B_depth_path = (
            B_path.parent.parent
            / B_path.parent.name.replace("B", "B_depth")
            / B_path.name
        )

        A_path = str(A_path)
        B_path = str(B_path)
        A_seg_path = str(A_seg_path)
        B_seg_path = str(B_seg_path)
        A_depth_path = str(A_depth_path)
        B_depth_path = str(B_depth_path)

        A_idx = A_path.split("/")[-1].split(".")[0]
        B_idx = B_path.split("/")[-1].split(".")[0]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        seed = random.randint(-sys.maxsize, sys.maxsize)

        A = Image.open(A_path).convert("RGB")
        B = Image.open(B_path).convert("RGB")
        A = self.fixed_transform(A, seed)
        B = self.fixed_transform(B, seed)

        A_depth = np.array(Image.open(A_depth_path)).astype(np.float32)
        B_depth = np.array(Image.open(B_depth_path)).astype(np.float32)
        A_depth = self.center_to_unit_ball(A_depth.reshape(*A_depth.shape, 1))
        B_depth = self.center_to_unit_ball(B_depth.reshape(*B_depth.shape, 1))
        A_depth = F.to_pil_image(A_depth)
        B_depth = F.to_pil_image(B_depth)
        A_depth = self.fixed_transform(A_depth, seed)
        B_depth = self.fixed_transform(B_depth, seed)

        A_segs = self.read_segs(A_seg_path, seed)
        B_segs = self.read_segs(B_seg_path, seed)

        if self.opt.direction == "BtoA":
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {
            "A": A,
            "B": B,
            "A_idx": A_idx,
            "B_idx": B_idx,
            "A_segs": A_segs,
            "B_segs": B_segs,
            "A_depth": A_depth,
            "B_depth": B_depth,
            "A_paths": A_path,
            "B_paths": B_path,
        }

    def __len__(self):
        return max(self.A_size, self.B_size)
