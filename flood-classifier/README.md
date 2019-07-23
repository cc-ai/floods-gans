# FLOOD CLASSIFIER

*Most of the code has been taken from the [TRANSFER LEARNING TUTORIAL](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py)*
## Model Architecture:

ResNet-18 model from
    [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) trained on ImageNet

Load a pretrained model and reset final fully connected layer / replace it with a fully connected layer with 2 outputs.

## Optimization:
All parameters are being optimized with Cross Entropy Loss as objective.

## Checkpoints:

Link to the checkpoints: [resnet-18-epoch24.pth](https://drive.google.com/open?id=1g6LepT70n_Qy3oDKXXIVf5KMPiYEkgJc) 


## Dataset of balanced class:

- Train: 712 images of flood / non-flood
- Eval:  210 images of flood / non-flood

Link to the [Dataset](https://drive.google.com/open?id=1-3ERWJEN4v_ZRqpHkJOoUwCgCYfipjjz)

## Performances:

- train_acc: 97.2 %
- val_acc: 99.5 %

**Remark**:
val_acc> train_acc because we saved the model with best validation score along the epochs.
