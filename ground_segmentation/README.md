# DeepLab - Segment the ground (Road, sidewalk, terrain)

We use [DeepLab](https://arxiv.org/pdf/1606.00915.pdf) model trained on Cityscapes dataset and merge some labels to output a binary mask of the Ground. Most of the code has been inspired from :[github/warmspringwinds](https://github.com/warmspringwinds/pytorch-segmentation-detection).

Merged labels: 

- Road
- Sidewalk
- Terrain

 
 **Pre-trained Model and Performances** 
 
 | Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-34-8s   | Validation set  |69.1  | in prog.           | in prog.       |50 ms.| [Dropbox](https://www.dropbox.com/s/jeaw9ny0jtl60uc/resnet_34_8s_cityscapes_best.pth?dl=0)            | 

**Label Table**

| Label            | Description |
|------------------|-----------|
| 0            | road |
| 1            | sidewalk |
| 2            | building |
| 3            | wall |
| 4            | fence |
| 5            | pole |
| 6            | traffic light |
| 7            | traffic sign |
| 8 | vegetation |
| 9            | terrain|
| 10            | sky |
| 11            | person |
| 12            | rider |
| 13            | car |
| 14            | truck|
| 15            | bus|
| 16            | train|
| 17            | motorcycle|
| 18            | bicycle|


## Installation : 

This code requires:

- Pytorch.
- SciPy==1.0.0
- Cuda

- Model weights downloaded

## How to use ?
### Several images
```
Usage: deeplab_ground_segmentation.py  [OPTIONS]

  Inference from a single folder

Options:
  --size_mask int             Size of the binary mask output [required]
  --path_to_images PATH       Folder of images to be processed [required]
  --dir_mask PATH             Folder where the masks will be stored [required]
  --batch_size int            Batch Size [required]
  --weight_pth PATH           PyTorch model to be loaded [required]

```

Example:

```python deeplab_ground_segmentation.py --size_mask 512 --path_to_images './images/' --dir_mask './masks/' --batch_size 16 --weight_pth 'resnet_34_8s_cityscapes_best.pth'```

Note:
The generated masks are non-binary, and multi-channel. They need to be processed to be binary 1-channel masks. This is done in the MUNIT dataloading code.
