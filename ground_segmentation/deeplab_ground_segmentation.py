# Pytorch implementation of deeplab with resnet_34_8s_cityscapes_best

import numpy
import torch
import torch.nn as nn
from resnet import resnet34
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from PIL import Image
import tqdm 
from scipy.misc import imresize
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--size_mask', action="store")
parser.add_argument('--path_to_images', action="store")
parser.add_argument('--dir_mask', action="store")
parser.add_argument('--batch_size', action="store")
parser.add_argument('--weight_pth', action="store")


args = parser.parse_args();
# Parameters 
size_mask_1 = args.size_mask
size_mask = (int(size_mask_1),int(size_mask_1))
path_to_images = args.path_to_images        # Image dir with ./images/
dir_mask = args.dir_mask
batch_size = int(args.batch_size)
weight_pth = args.weight_pth

path_list = os.listdir(path_to_images)

class Resnet34_8s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)

        self.resnet34_8s = resnet34_8s

        self._normal_initialization(self.resnet34_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet34_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x



# Perform image transformation
from torchvision import transforms
valid_transform = transforms.Compose(
                [
                     transforms.Resize(size_mask),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

# Init model and send to Cuda
model = Resnet34_8s(num_classes=19).to('cuda')
PATH  = weight_pth
model.load_state_dict(torch.load(PATH))


class MyDataset(Dataset):
    def __init__(self,root,transform=None):
        self.images=[root+f for f in os.listdir(root)]
        self.transform=transform
        self.root = root

    def __getitem__(self,index):
        path=self.images[index]
        image=Image.open(path).convert('RGB')
        if self.transform is not None:
           image=self.transform(image)
        return image
    def __len__(self):
        return len(self.images)

    def getPaths(self):
        return os.listdir(self.root)


val_ds=MyDataset(path_to_images,transform = valid_transform)

val_dl=torch.utils.data.DataLoader(val_ds,batch_size=batch_size,shuffle=False)

model.eval()

it_= 0 
list_paths = val_ds.getPaths()

with torch.no_grad():
    for i_batch, images in tqdm.tqdm(enumerate(val_dl)):
        imgs =images.to('cuda')
        out_batch= model(imgs.float()).cpu()
        batch_size_ = len(out_batch)
        for j in range(batch_size_):
          out1=out_batch[j,:,:,:]
          res=out1.squeeze(0).max(0)

          original_image=plt.imread(path_to_images+list_paths[it_*batch_size+j])
          original_image = imresize(original_image,size=size_mask)
          patches =[]
          mask_flat=np.zeros(size_mask)
          for p in [0,1,9]: 
            mask_flat[np.where(res[1]==p)]=255
          # Sasha: to help with the multi-channel jpg masks
          filename = dir_mask+list_paths[it_*batch_size+j]
          extension_length = len(filename.split('.')[-1])
          filename_png = filename[:-extension_length-1]+'.png'
          cv2.imwrite(filename_png,mask_flat.astype(int))
          #cv2.imwrite(dir_mask+list_paths[it_*batch_size+j],mask_flat.astype(int))
        
        it_+=1
