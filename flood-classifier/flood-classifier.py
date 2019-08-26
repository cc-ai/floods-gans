from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the Model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('./checkpoints/resnet-18-epoch24.pth'))
model.eval()
model.to(device)

# Normalization for inference PIL -> torch.Variable
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Normalization for inference torch.Variable -> torch.Variable
transform_torchVar = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

criterion = nn.CrossEntropyLoss()


# Example with an image path
image_to_label = PIL.Image.open('test.png')
input_transformed = data_transforms(image_to_label).unsqueeze(0)

# Example with a torch tensor as the output of a gan generator
# x_ab = self.gen_b.decode(c_a, s_b)
# input_transformed = transform_torchVar(x_ab) # x_ab must be image format with 3 channels

label_flood = 0
label_non_flood = 1
cuda_label_flood = torch.tensor([label_flood]).cuda()
cuda_label_non_flood = torch.tensor([label_non_flood]).cuda()

with torch.no_grad():
    input_transformed = input_transformed.to(device)
    outputs = model(input_transformed)
    loss = criterion(outputs,cuda_label_flood)
    print('loss = ',loss)


