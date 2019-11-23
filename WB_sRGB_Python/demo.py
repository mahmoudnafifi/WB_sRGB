## Demo: White balancing a single image
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

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# input and options
in_img = "../example_images/figure3.jpg" # input image filename
out_dir = "." # output directory
upgraded_model = 1 # use 1 to load our new model that is upgraded with new training examples.
gamut_mapping = 2 # use 1 for scaling, 2 for clipping (our paper's results reported
# using clipping). If the image is over-saturated, scaling is recommended.
imshow = 1 # show input/output image

# processing
wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                         upgraded=upgraded_model) # create an instance of the WB model
os.makedirs(out_dir, exist_ok=True)
I = cv2.imread(in_img) # read the image
outImg = wbModel.correctImage(I) # white balance it
cv2.imwrite(out_dir + '/' + 'result.jpg', outImg * 255) # save it

if imshow == 1:
    cv2.imshow('input',ResizeWithAspectRatio(I, width=600))
    cv2.imshow('our result', ResizeWithAspectRatio(outImg, width=600))
    cv2.waitKey()
    cv2.destroyAllWindows()




