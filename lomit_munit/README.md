[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
## MUNIT: Multimodal UNsupervised Image-to-image Translation

## Running the code
Under the configs folder is the file : ```street2flood_list.yaml``` which lists all the parameters and the location where the data is stored and the format in which the data is input

You have to create 4 .txt files (trainA.txt, trainB.txt, testA.txt, testB.txt) in which every line contains the absolute address to image followed by a space and then absolute address to corresponding mask where the image and mask belong to respective folders i.e. (trainA, trainB, testA, testB)

## Running command
```python train.py --config configs/street2flood_list.yaml```

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Code usage

Please check out the [user manual page](USAGE.md).

### Paper

[Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Ming-Yu Liu](http://mingyuliu.net/), [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), [Jan Kautz](http://jankautz.com/), "[Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732)", ECCV 2018

### Results Video
[![](results/video.jpg)](https://youtu.be/ab64TWzWn40)

### Edges to Shoes/handbags Translation

![](results/edges2shoes_handbags.jpg)

### Animal Image Translation

![](results/animal.jpg)

### Street Scene Translation

![](results/street.jpg)

### Yosemite Summer to Winter Translation (HD)

![](results/summer2winter_yosemite.jpg)

### Example-guided Image Translation

![](results/example_guided.jpg)

### Other Implementations

[MUNIT-Tensorflow](https://github.com/taki0112/MUNIT-Tensorflow) by [Junho Kim](https://github.com/taki0112)

[MUNIT-keras](https://github.com/shaoanlu/MUNIT-keras) by [shaoanlu](https://github.com/shaoanlu)

### Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{huang2018munit,
  title={Multimodal Unsupervised Image-to-image Translation},
  author={Huang, Xun and Liu, Ming-Yu and Belongie, Serge and Kautz, Jan},
  booktitle={ECCV},
  year={2018}
}
```


