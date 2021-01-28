## Calculate errors between the corrected image and the ground truth image.
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


from evaluation.calc_deltaE import calc_deltaE
from evaluation.calc_deltaE2000 import calc_deltaE2000
from evaluation.calc_mse import calc_mse
from evaluation.calc_mae import calc_mae


def evaluate_cc(corrected, gt, color_chart_area, opt=1):
  """
    Color constancy (white-balance correction) evaluation of a given corrected
    image.
    :param corrected: corrected image
    :param gt: ground-truth image
    :param color_chart_area: If there is a color chart in the image, that is
     masked out from both images, this variable represents the number of pixels
     of the color chart.
    :param opt: determines the required error metric(s) to be reported.
         Options:
           opt = 1 delta E 2000 (default).
           opt = 2 delta E 2000 and mean squared error (MSE)
           opt = 3 delta E 2000, MSE, and mean angular eror (MAE)
           opt = 4 delta E 2000, MSE, MAE, and delta E 76
    :return: error(s) between corrected and gt images
    """

  if opt == 1:
    return calc_deltaE2000(corrected, gt, color_chart_area)
  elif opt == 2:
    return calc_deltaE2000(corrected, gt, color_chart_area), calc_mse(
      corrected, gt, color_chart_area)
  elif opt == 3:
    return calc_deltaE2000(corrected, gt, color_chart_area), calc_mse(
      corrected, gt, color_chart_area), calc_mae(corrected, gt,
                                                 color_chart_area)
  elif opt == 4:
    return calc_deltaE2000(corrected, gt, color_chart_area), calc_mse(
      corrected, gt, color_chart_area), calc_mae(
      corrected, gt, color_chart_area), calc_deltaE(corrected, gt,
                                                    color_chart_area)
  else:
    raise Exception('Error in evaluate_cc function')
