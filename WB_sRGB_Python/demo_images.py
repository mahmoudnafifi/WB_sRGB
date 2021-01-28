## Demo: White balancing all images in a directory
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
import os
from classes import WBsRGB as wb_srgb

# input and options
in_dir = '../example_images/'
out_dir = '../example_images_WB/'
if not os.path.exists(out_dir):
  os.mkdir(out_dir)
# use upgraded_model = 1 to load our new model that is upgraded with new
# training examples.
upgraded_model = 0
# use gamut_mapping = 1 for scaling, 2 for clipping (our paper's results
# reported using clipping). If the image is over-saturated, scaling is
# recommended.
gamut_mapping = 2


# processing
# create an instance of the WB model
wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                         upgraded=upgraded_model)
imgfiles = []
valid_images = (".jpg", ".jpeg", ".png")
for f in os.listdir(in_dir):
  if f.lower().endswith(valid_images):
    imgfiles.append(os.path.join(in_dir, f))
for in_img in imgfiles:
  print("processing image: " + in_img + "\n")
  filename, file_extension = os.path.splitext(in_img)  # get file parts
  I = cv2.imread(in_img)  # read the image
  outImg = wbModel.correctImage(I)  # white balance it
  cv2.imwrite(out_dir + '/' + os.path.basename(filename) +
              '_WB' + file_extension, outImg * 255)  # save it
