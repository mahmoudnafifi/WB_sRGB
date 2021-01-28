## Calculate mean Delta E76 between source and target images.
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


import cv2
import numpy as np
from skimage import color


def calc_deltaE(source, target, color_chart_area):
  source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
  target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
  source = color.rgb2lab(source)
  target = color.rgb2lab(target)
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  target = np.reshape(target, [-1, 3]).astype(np.float32)
  delta_e = np.sqrt(np.sum(np.power(source - target, 2), 1))
  return sum(delta_e) / (np.shape(delta_e)[0] - color_chart_area)

#################################################
# References:
# [1] http://zschuessler.github.io/DeltaE/learn/
