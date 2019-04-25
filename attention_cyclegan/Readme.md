# CCAI - Unsupervised Attention-guided Image-to-Image Translation

This is a fork of the original repo -> https://github.com/AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation, adapted to our needs.

The original readme is [here](Original_README.md) and License is kept.

# How to use

## Preparing new dataset

1. Convert all your images to a single format (for instance .png) (see [preprocess_ccai_images.py](https://github.com/cc-ai/floods-gans/tree/master/input/ccai-all/preprocess_ccai_images.py))
2. Edit the [`cyclegan_datasets.py`](cyclegan_datasets.py) file. For example, if you have a horse2zebra_train dataset which contains 1067 horse images and 1334 zebra images (both in JPG format), you can just edit the [`cyclegan_datasets.py`](cyclegan_datasets.py) as following:

		```python
		DATASET_TO_SIZES = {
		  'horse2zebra_train': 1334
		}

		PATH_TO_CSV = {
		  'horse2zebra_train': './AGGAN/input/horse2zebra/horse2zebra_train.csv'
		}

		DATASET_TO_IMAGETYPE = {
		  'horse2zebra_train': '.jpg'
		}
		``` 
3. Run `create_cyclegan_dataset.py`. For instance for the `train` set of the `ccai-all` dataset:
		```bash
		python -m create_cyclegan_dataset --image_path_a='./input/ccai-all/trainA' --image_path_b='./input/ccai-all/trainB'  --dataset_name="ccai-all_train" --do_shuffle=0
		```

### Training
* Create the configuration file. The configuration file contains basic information for training/testing. An example of the configuration file could be found at [```configs/exp_01.json```](configs/exp_01.json).

* Start training:
	```bash
	python main.py  --to_train=1 --log_dir=./output/AGGAN/exp_01 --config_filename=./configs/ccai_01.json
	```
* Check the intermediate results:
	* Tensorboard
		```bash
		tensorboard --port=6006 --logdir=./output/AGGAN/exp_01/#timestamp# 
		```
	* Check the html visualization at ./output/AGGAN/exp_01/#timestamp#/epoch_#id#.html.  

### Restoring from the previous checkpoint
```bash
python main.py --to_train=2 --log_dir=./output/AGGAN/exp_01 --config_filename=./configs/exp_01.json --checkpoint_dir=./output/AGGAN/exp_01/#timestamp#
```

### Testing
* Create the testing dataset:
	* Edit the cyclegan_datasets.py file the same way as training.
	* Create the csv file as the input to the data loader:
		```bash
		python -m create_cyclegan_dataset --image_path_a='./input/horse2zebra/testB' --image_path_b='./input/horse2zebra/testA' --dataset_name="horse2zebra_test" --do_shuffle=0
		```
* Run testing:
	```bash
	python main.py --to_train=0 --log_dir=./output/AGGAN/exp_01 --config_filename=./configs/exp_01_test.json --checkpoint_dir=./output/AGGAN/exp_01/#old_timestamp# 
	```