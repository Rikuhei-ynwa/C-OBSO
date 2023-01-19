#!/bin/bash
# convert tracking data for LaurieOnTracking format, the number of game is 34 (0..33).
# for id in {0..33}
# do
#     python tracking_convert.py --id $id
# done
# create train and test data for predicting, predict length is 4sec.
python preprocess_train.py --len 4
