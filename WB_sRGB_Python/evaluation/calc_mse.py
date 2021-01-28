## Calculate mean squared error between source and target images.
#
# Copyright (c) 2018-present, Mahmoud Afifi
# York University, Canada
# mafifi@eecs.yorku.ca | m.3afifi@gmail.com
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown,
# "When color constancy goes wrong: Correcting improperly white-balanced
# images", CVPR 2019.
#
##########################################################################


import numpy as np


def calc_mse(source, target, color_chart_area):
  source = np.reshape(source, [-1, 1]).astype(np.float64)
  target = np.reshape(target, [-1, 1]).astype(np.float64)
  mse = sum(np.power((source - target), 2))
  return mse / ((np.shape(source)[
    0]) - color_chart_area)
