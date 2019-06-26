# CCAI - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

This is a fork of the original repo -> https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, adapted to our needs.

The original readme is [here](Original_README.md) and License is kept.

# How to use

Testing:

```
python test.py --dataroot /network/tmp1/ccai/data/val_set --name flood_cg_cropped_street --model test --no_dropout --results_dir /network/tmp1/ccai/results/cycle_gan/val_set_19_06
```