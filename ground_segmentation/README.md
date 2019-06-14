# DeepLab - Segment the ground (Road, sidewalk, terrain)

**Training dataset Cityscapes**

 The dataset contains video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of  ```5 000``` frames. The annotations contain ```19``` classes which represent cars, road, traffic signs and so on.
 
 **Performances**
 
 | Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-34-8s   | Validation set  |69.1  | in prog.           | in prog.       |50 ms.| [Dropbox](https://www.dropbox.com/s/jeaw9ny0jtl60uc/resnet_34_8s_cityscapes_best.pth?dl=0)            |


Link to the article : [DeepLab](https://arxiv.org/pdf/1606.00915.pdf)

Most of the code has been inspired from :[github/warmspringwinds](https://github.com/warmspringwinds/pytorch-segmentation-detection)

## Installation : 

This code requires:

- Pytorch.
- SciPy==1.0.0
- Cuda

- Pretrained weight downloaded

**Examples**

```python deeplab_ground_segmentation.py --size_mask 512 --path_to_images './images/' --dir_mask './masks/' --batch_size 16 --weight_pth 'resnet_34_8s_cityscapes_best.pth'```
