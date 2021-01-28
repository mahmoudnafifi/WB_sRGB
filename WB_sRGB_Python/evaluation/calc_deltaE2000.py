## Calculate mean Delta 2000 between source and target images.
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


def calc_deltaE2000(source, target, color_chart_area):
  source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
  target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
  source = color.rgb2lab(source)
  target = color.rgb2lab(target)
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  target = np.reshape(target, [-1, 3]).astype(np.float32)
  deltaE00 = deltaE2000(source, target)
  return sum(deltaE00) / (np.shape(deltaE00)[0] - color_chart_area)


def deltaE2000(Labstd, Labsample):
  kl = 1
  kc = 1
  kh = 1
  Lstd = np.transpose(Labstd[:, 0])
  astd = np.transpose(Labstd[:, 1])
  bstd = np.transpose(Labstd[:, 2])
  Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
  Lsample = np.transpose(Labsample[:, 0])
  asample = np.transpose(Labsample[:, 1])
  bsample = np.transpose(Labsample[:, 2])
  Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
  Cabarithmean = (Cabstd + Cabsample) / 2
  G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
    Cabarithmean, 7) + np.power(25, 7))))
  apstd = (1 + G) * astd
  apsample = (1 + G) * asample
  Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
  Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
  Cpprod = (Cpsample * Cpstd)
  zcidx = np.argwhere(Cpprod == 0)
  hpstd = np.arctan2(bstd, apstd)
  hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
  hpsample = np.arctan2(bsample, apsample)
  hpsample = hpsample + 2 * np.pi * (hpsample < 0)
  hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
  dL = (Lsample - Lstd)
  dC = (Cpsample - Cpstd)
  dhp = (hpsample - hpstd)
  dhp = dhp - 2 * np.pi * (dhp > np.pi)
  dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
  dhp[zcidx] = 0
  dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
  Lp = (Lsample + Lstd) / 2
  Cp = (Cpstd + Cpsample) / 2
  hp = (hpstd + hpsample) / 2
  hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
  hp = hp + (hp < 0) * 2 * np.pi
  hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
  Lpm502 = np.power((Lp - 50), 2)
  Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
  Sc = 1 + 0.045 * Cp
  T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
      0.32 * np.cos(3 * hp + np.pi / 30) \
      - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
  Sh = 1 + 0.015 * Cp * T
  delthetarad = (30 * np.pi / 180) * np.exp(
    - np.power((180 / np.pi * hp - 275) / 25, 2))
  Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
  RT = - np.sin(2 * delthetarad) * Rc
  klSl = kl * Sl
  kcSc = kc * Sc
  khSh = kh * Sh
  de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                 np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))
  return de00

#################################################
# References:
# [1] The CIEDE2000 Color-Difference Formula: Implementation Notes,
# Supplementary Test Data, and Mathematical Observations,", G. Sharma,
# W. Wu, E. N. Dalal, Color Research and Application, 2005.
