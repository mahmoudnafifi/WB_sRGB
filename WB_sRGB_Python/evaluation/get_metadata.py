## Get the metadata of the given image fileName
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
import re


def get_metadata(fileName, set, metadata_baseDir=''):
    """
    Gets metadata (e.g., ground-truth file name, chart coordinates and area).
    :param fileName: input filename
    :param set: which dataset?--options includes: 'RenderedWB_Set1',
      'RenderedWB_Set2', 'Rendered_Cube+'
    :param metadata_baseDir: metadata directory (required for Set1 only)
    :return: metadata for a given image
    evaluation_examples.py provides some examples of how to use it
    """

    fname, file_extension = os.path.splitext(fileName)  # get file parts
    name = os.path.basename(fname)  # get only filename without the directory

    if set == 'RenderedWB_Set1': # Rendered WB dataset (Set1)
        metadatafile_color = name + '_color.txt' # chart's colors info.
        metadatafile_mask = name + '_mask.txt' # chart's coordinate info.
        # get color info.
        f = open(os.path.join(metadata_baseDir, metadatafile_color), 'r')
        C = f.read()
        colors = np.zeros((3, 24))  # color chart colors
        temp = re.split(',|\n', C)
        # 3 x 24 colors in the color chart
        colors = np.reshape(np.asfarray(temp[:-1], float), (24, 3)).transpose()
        # get coordinate info
        f = open(os.path.join(metadata_baseDir, metadatafile_mask), 'r')
        C = f.read()
        temp = re.split(',|\n', C)
        # take only the first 4 elements (i.e., the color chart coordinates)
        temp = temp[0:4]
        mask = np.asfarray(temp, float)  # color chart mask coordinates
        # get ground-truth file name
        seperator = '_'
        temp = name.split(seperator)
        gt_file = seperator.join(temp[:-2])
        gt_file = gt_file + '_G_AS.png'
        # compute mask area
        mask_area = mask[2] * mask[3]
        # final metadata
        data = {"gt_filename": gt_file, "cc_colors": colors, "cc_mask": mask,
                "cc_mask_area": mask_area}

    elif set == 'RenderedWB_Set2': # Rendered WB dataset (Set2)
        data = {"gt_filename": name + file_extension, "cc_colors": None,
                "cc_mask": None, "cc_mask_area": 0}

    elif set == 'Rendered_Cube+': # Rendered Cube+
        # get ground-truth filename
        temp = name.split('_')
        gt_file = temp[0] + file_extension
        mask_area = 58373  # calibration obj's area is fixed over all images
        data = {"gt_filename": gt_file, "cc_colors": None, "cc_mask": None,
                "cc_mask_area": mask_area}
    else:
        raise Exception(
            "Invalid value for set variable. " +
            "Please use: 'RenderedWB_Set1', 'RenderedWB_Set2', 'Rendered_Cube+'")

    return data
