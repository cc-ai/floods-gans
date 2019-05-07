### PREPROCESSING IMAGE PIPELINE : USING A DEEP PHOTO ENHANCER

The code has been adapted from [nothinglo/Deep-Photo-Enhancer](https://github.com/nothinglo/Deep-Photo-Enhancer/blob/master/README.md) to be compatible with tensorflow 1.0. 

**Remark 1 :** Only the model trained on MIT-Adobe 5K dataset with unpaired data is available here.
**Remark 2 :** There might be an issue with image resizing through the network. (Other version without resizing is available but slower).


**To test the inference ** :

- Download the pre-trained [weights](https://drive.google.com/open?id=193uXlDcM41QYbf8QeFML0wkOKrKIjywd)
- Unzip in the model repertory.
- Use the main.py


### CITATION

```
@INPROCEEDINGS{Chen:2018:DPE,
	AUTHOR    = {Yu-Sheng Chen and Yu-Ching Wang and Man-Hsin Kao and Yung-Yu Chuang},
	TITLE     = {Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs},
	YEAR      = {2018},
	MONTH     = {June},
	BOOKTITLE = {Proceedings of IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2018)},
	PAGES     = {6306--6314},
	LOCATION  = {Salt Lake City},
}
```
