# Counting component on toy dataset

This directory contains code for training and evaluating on our toy dataset, designed to evaluate counting ability.
The files of interest in here are the data generator in `data.py`, the training script `train.py`, and maybe the baseline and counting models in `model.py`.

To train the two models, run either of the following two commands (or both):
```
python train.py easy
python train.py hard
```
Both perform a parameter sweep of the maximum side length and noise parameters individually from 0 to 1.
In the `easy` configuration, the parameter that is not swept over is set to 0, in the `hard` configuration it is set to 0.5.

If for some reason the symlink for `counting.py` is not working, just copy it over from the vqa-v2 directory.

Warning: this is basically the exact code that was used to generate the plots in the paper.
As such, the plotting scripts are rather horrendous and hard to follow.
However, that should mean that you can reproduce the toy dataset figures with relative ease.


## Dependencies

This code was confirmed to run with the following environment:

- Python 3.6.2
  - torch 0.3.0
  - torchvision 0.2
  - h5py 2.7.0
  - tqdm 4.19.2

