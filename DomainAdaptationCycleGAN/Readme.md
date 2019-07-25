# CCAI - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

This is a fork of the original repo -> https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, adapted to our needs.

The original readme is [here](Original_README.md) and License is kept.

# CycleGAN for Domain Adaptation

Training :
```
python train.py --dataroot /network/tmp1/ccai/data/elementai_mapillary/ --name elementai_to_mapillary --no_dropout --checkpoints_dir /network/tmp1/ccai/data/domain_adaptation
```
Testing :
```
python test.py --dataroot /network/tmp1/ccai/data/elementai_data/ --name element_ai_off_to_on --model cycle_gan --no_dropout --checkpoints_dir /network/tmp1/ccai/data/domain_adaptation
```
