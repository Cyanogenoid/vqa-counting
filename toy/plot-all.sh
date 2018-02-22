#!/bin/bash

BASE=.
set -x

python plot-acc.py $BASE

python plot.py $BASE/noise-hard.pth
python plot.py $BASE/coord-hard.pth
python plot.py $BASE/noise-hard.pth full
python plot.py $BASE/coord-hard.pth full

# if you have a .pth log from vqa-v2, you can pass it here instead of test.pth to plot its activation functions
python plot.py test.pth full $BASE/noise-hard.pth

python visualise-dataset.py 0.0
python visualise-dataset.py 0.5

