## Evaluation examples
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

import os
import numpy as np
import cv2

from classes import WBsRGB as wb_srgb
from evaluation.get_metadata import get_metadata
from evaluation.evaluate_cc import evaluate_cc

# Please remove this part when you use your method
################################################################################
# WB model options
# use upgraded_model = 1 to load our new model that is upgraded with new
# training examples.
upgraded_model = 0
# use gamut_mapping = 1 for scaling, 2 for clipping (our paper's results
# reported
gamut_mapping = 2
# using clipping). If the image is over-saturated, scaling is recommended.
# create an instance of the WB model
wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping, upgraded=upgraded_model)
################################################################################


# Example1 (RenderedWB_Set1)
print('Example of evaluating on Set1 from the Rendered WB dataset')
dataset_name = 'RenderedWB_Set1'
imgin = 'Canon1DsMkIII_0087_F_P.png'
in_base = os.path.join('..', 'examples_from_datasets', 'RenderedWB_Set1',
                       'input')
gt_base = os.path.join('..', 'examples_from_datasets', 'RenderedWB_Set1',
                       'groundtruth')
metadata_base = os.path.join('..', 'examples_from_datasets', 'RenderedWB_Set1',
                             'metadata')
print('Reading image:' + imgin)
metadata = get_metadata(imgin, dataset_name, metadata_base)  # get metadata
# round any float to nearest integer
cc_mask = np.round(metadata["cc_mask"]).astype("uint64")
# read the image
I_in = cv2.imread(os.path.join(in_base, imgin), cv2.IMREAD_COLOR)
# read gt image
gt = cv2.imread(os.path.join(gt_base, metadata["gt_filename"]),
                cv2.IMREAD_COLOR)
#  hide the color chart from both images before processing and evaluation
I_in[cc_mask[1]:cc_mask[1] + cc_mask[3], cc_mask[0]:cc_mask[0] + cc_mask[2],
:] = 0
gt[cc_mask[1]:cc_mask[1] + cc_mask[3], cc_mask[0]:cc_mask[0] + cc_mask[2],
:] = 0

# processing (replace this part with your method)
print('Processing image: ' + imgin)
I_corr = wbModel.correctImage(I_in)  # white balance I_in
I_corr = (I_corr * 255).astype('uint8')  # convert to uint8 image

# Evaluation
deltaE00, MSE, MAE, deltaE76 = evaluate_cc(I_corr, gt, metadata["cc_mask_area"],
                                           opt=4)
print('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n'
      % (deltaE00, MSE, MAE, deltaE76))

################################################################################


# Example2 (RenderedWB_Set2)
print('\n\nExample of evaluating on Set2 from the Rendered WB dataset')
dataset_name = 'RenderedWB_Set2'
imgin = 'Mobile_00202.png'
in_base = os.path.join('..', 'examples_from_datasets', 'RenderedWB_Set2',
                       'input')
gt_base = os.path.join('..', 'examples_from_datasets', 'RenderedWB_Set2',
                       'groundtruth')
metadata_base = ''  # no metadata directory required for Set2
# get metadata--just to have a consistent style.
metadata = get_metadata(imgin, dataset_name, metadata_base)
# read the image
I_in = cv2.imread(os.path.join(in_base, imgin), cv2.IMREAD_COLOR)
# read gt image
gt = cv2.imread(os.path.join(gt_base, metadata["gt_filename"]),
                cv2.IMREAD_COLOR)

# processing (replace this part with your method)
print('Processing image: ' + imgin)
I_corr = wbModel.correctImage(I_in)  # white balance I_in
I_corr = (I_corr * 255).astype('uint8')  # convert to uint8 image

# Evaluation
deltaE00, MSE, MAE, deltaE76 = evaluate_cc(I_corr, gt, metadata["cc_mask_area"],
                                           opt=4)
print('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n'
      % (deltaE00, MSE, MAE, deltaE76))

################################################################################


# Example3 (Rendered_Cube+)
print('\n\nExample of evaluating on the Rendered version of Cube+ dataset')
dataset_name = 'Rendered_Cube+'
imgin = '19_F.JPG'
in_base = os.path.join('..', 'examples_from_datasets', 'Rendered_Cube+',
                       'input')
gt_base = os.path.join('..', 'examples_from_datasets', 'Rendered_Cube+',
                       'groundtruth')
metadata_base = ''  # no metadata directory required for the renderd Cube+
# get metadata (we need the cube area for evaluation)
metadata = get_metadata(imgin, dataset_name, metadata_base)
# read the image
I_in = cv2.imread(os.path.join(in_base, imgin), cv2.IMREAD_COLOR)
# read gt image
gt = cv2.imread(os.path.join(gt_base, metadata["gt_filename"]),
                cv2.IMREAD_COLOR)

# processing (replace this part with your method)
print('Processing image: ' + imgin)
I_corr = wbModel.correctImage(I_in)  # white balance I_in
I_corr = (I_corr * 255).astype('uint8')  # convert to uint8 image

# Evaluation
deltaE00, MSE, MAE, deltaE76 = evaluate_cc(I_corr, gt, metadata["cc_mask_area"],
                                           opt=4)
print('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n'
      % (deltaE00, MSE, MAE, deltaE76))
