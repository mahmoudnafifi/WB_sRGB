## White-balance model class
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


from datetime import datetime

# import numpy.matlib
import cupy as cp
import cv2
import numpy as np


class WBsRGB:
    def __init__(self, gamut_mapping=2, upgraded=0):
        cp.cuda.Device(1).use()
        if upgraded == 1:
            self.features = cp.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/features+.npy')  # training encoded features
            self.mappingFuncs = np.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/mappingFuncs+.npy')  # mapping correction functions
            self.encoderWeights = np.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/encoderWeights+.npy')  # weight matrix for histogram encoding
            self.encoderBias = np.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/encoderBias+.npy')  # bias vector for histogram encoding
            self.K = 75  # K value for nearest neighbor searching---for the upgraded model, we found 75 is better
        else:
            self.features = cp.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/features.npy')  # training encoded features
            self.mappingFuncs = np.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/mappingFuncs.npy')  # mapping correction functions
            self.encoderWeights = np.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/encoderWeights.npy')  # weight matrix for histogram encoding
            self.encoderBias = np.load(
                '/opt/instore-app/test/whitebalance/WB_sRGB_Python/models/encoderBias.npy')  # bias vector for histogram encoding
            self.K = 25  # K value for nearest neighbor searching

        self.sigma = 0.25  # fall-off factor for KNN blending
        self.h = 60  # histogram bin width
        # our results reported with gamut_mapping=2, however gamut_mapping=1 gives more compelling results with
        # over-saturated examples
        self.gamut_mapping = gamut_mapping  # options: =1 for scaling, =2 for clipping

    def encode(self, hist):
        """ Generates a compacted feature of a given RGB-uv histogram tensor. """
        histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                    (1, int(hist.size / 3)), order="F")  # reshaped red layer of histogram
        histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                    (1, int(hist.size / 3)), order="F")  # reshaped green layer of histogram
        histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                    (1, int(hist.size / 3)), order="F")  # reshaped blue layer of histogram
        hist_reshaped = np.append(histR_reshaped,
                                  [histG_reshaped, histB_reshaped])  # reshaped histogram n * 3 (n = h*h)
        feature = np.dot(hist_reshaped - self.encoderBias.transpose(),
                         self.encoderWeights)  # compute compacted histogram feature
        return feature

    def rgbUVhist(self, I):
        """ Computes an RGB-uv histogram tensor. """
        sz = np.shape(I)  # get size of current image
        if sz[0] * sz[1] > 202500:  # if it is larger than 450*450
            factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
            newH = int(np.floor(sz[0] * factor))
            newW = int(np.floor(sz[1] * factor))
            I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)  # resize image
        II = I.reshape(int(I.size / 3), 3)  # n*3
        inds = np.where((II[:, 0] > 0) & (II[:, 1] > 0) & (II[:, 2] > 0))  # remove any zero pixels
        R = II[inds, 0]  # red channel
        G = II[inds, 1]  # green channel
        B = II[inds, 2]  # blue channel
        I_reshaped = np.concatenate((R, G, B), axis=0).transpose()  # reshaped image (wo zero values)
        I_reshaped = cp.asarray(I_reshaped)
        eps = 6.4 / self.h
        A = cp.arange(-3.2, 3.19, eps)  # dummy vector
        hist = cp.zeros((A.size, A.size, 3))  # histogram will be stored here
        Iy = cp.sqrt(cp.power(I_reshaped[:, 0], 2) + cp.power(I_reshaped[:, 1], 2) +
                     cp.power(I_reshaped[:, 2], 2))  # intensity vector
        for i in range(3):  # for each histogram layer, do
            time_start_for = datetime.now()
            r = []  # excluded channels will be stored here
            for j in range(3):  # for each color channel do
                if j != i:  # if current color channel does not match current histogram layer,
                    r.append(j)  # exclude it
            Iu = cp.log(I_reshaped[:, i] / I_reshaped[:, r[1]])  # current color channel / the first excluded channel
            Iv = cp.log(I_reshaped[:, i] / I_reshaped[:, r[0]])  # current color channel / the second excluded channel
            diff_u = cp.abs(
                cp.tile(Iu, (cp.size(A), 1)).transpose() - cp.tile(A, (cp.size(Iu), 1)))  # differences in u space
            diff_v = cp.abs(
                cp.tile(Iv, (cp.size(A), 1)).transpose() - cp.tile(A, (cp.size(Iv), 1)))  # differences in v space
            diff_u[diff_u >= (eps / 2)] = 0  # do not count any pixel has difference beyond the threshold in the u space
            diff_u[diff_u != 0] = 1  # remaining pixels will be counted
            diff_v[diff_v >= (eps / 2)] = 0  # do not count any pixel has difference beyond the threshold in the v space
            diff_v[diff_v != 0] = 1  # remaining pixels will be counted
            # here, we will use a matrix multiplication expression to compute eq. 4 in the main paper.
            # why? because it is much faster
            temp = (cp.tile(Iy, (cp.size(A), 1)) * (diff_u).transpose())  # Iy .* diff_u' (.* element-wise mult)
            hist[:, :, i] = cp.dot(temp, diff_v)  # initialize current histogram layer with Iy .* diff' * diff_v
            norm_ = cp.sum(hist[:, :, i], axis=None)  # compute sum of hist for normalization
            hist[:, :, i] = cp.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)

        return hist

    def correctImage(self, I):
        """ White balance a given image I. """
        time_read = datetime.now()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
        I = im2double(I)  # convert to double
        print("Image convert : {}".format(datetime.now() - time_read))
        time_start_hist = datetime.now()
        I_hist = self.rgbUVhist(I)
        I_hist = cp.asnumpy(I_hist)
        print("Hist computation: {}".format(datetime.now() - time_start_hist))
        time_start_encode = datetime.now()
        feature = self.encode(I_hist)
        print("Encode : {}".format(datetime.now() - time_start_encode))
        time_start_dsq = datetime.now()
        feature = cp.asarray(feature)
        D_sq = cp.linalg.norm(self.features - feature, axis=1, keepdims=True)
        D_sq = cp.asnumpy(D_sq)
        print("Dist computation : {}".format(datetime.now() - time_start_dsq))
        time_start_uc = datetime.now()
        idH = D_sq.argpartition(self.K, axis=0)[:self.K]  # get smallest K distances
        mappingFuncs = np.squeeze(self.mappingFuncs[idH, :])
        dH = np.take_along_axis(D_sq, idH, axis=0)  # square root nearest distances to get real euclidean distances
        sorted_idx = dH.argsort(axis=0)  # get sorting indices
        dH = np.take_along_axis(dH, sorted_idx, axis=0)  # sort distances
        weightsH = np.exp(-(np.power(dH, 2)) /
                          (2 * np.power(self.sigma, 2)))  # compute blending weights
        weightsH = weightsH / sum(weightsH)  # normalize blending weights
        mf = sum(np.tile(weightsH, (1, 33)) *
                 mappingFuncs, 0)  # compute the mapping function
        mf = mf.reshape(11, 3, order="F")  # reshape it to be 9 * 3
        print("Mapping and weights : {}".format(datetime.now() - time_start_uc))
        time_start_c = datetime.now()
        I_corr = self.colorCorrection(I, mf)  # apply it!
        print("Correction : {}".format(datetime.now() - time_start_c))
        return I_corr

    def colorCorrection(self, input, m):
        """ Applies a mapping function m to a given input image. """
        sz = np.shape(input)  # get size of input image
        I_reshaped = np.reshape(input, (int(input.size / 3), 3),
                                order="F")  # reshape input to be n*3 (n: total number of pixels)
        shape_mat = np.shape(I_reshaped)
        I_reshaped = cp.asarray(I_reshaped)
        kernel_out = kernelP(I_reshaped, shape_mat)  # raise input image to a higher-dim space
        m = cp.asarray(m)
        out = cp.dot(kernel_out, m)  # apply m to the input image after raising it the selected higher degree
        if self.gamut_mapping == 1:
            out = normScaling(I_reshaped, out)  # scaling based on input image energy
        elif self.gamut_mapping == 2:
            out = outOfGamutClipping(out)  # clip out-of-gamut pixels
        else:
            raise Exception('Wrong gamut_mapping value')
        out = cp.asnumpy(out)
        out = out.reshape(sz[0], sz[1], sz[2], order="F")  # reshape output image back to the original image shape
        out = cv2.cvtColor(out.astype('float32'), cv2.COLOR_RGB2BGR)
        return out


def normScaling(I, I_corr):
    """ Scales each pixel based on original image energy. """
    norm_I_corr = cp.sqrt(cp.sum(cp.power(I_corr, 2), 1))
    inds = norm_I_corr != 0
    norm_I_corr = norm_I_corr[inds]
    norm_I = cp.sqrt(cp.sum(cp.power(I[inds, :], 2), 1))
    I_corr[inds, :] = I_corr[inds, :] / cp.tile(norm_I_corr[:, cp.newaxis], 3) * \
                      cp.tile(norm_I[:, cp.newaxis], 3)
    return I_corr


def kernelP(I, shape_mat):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    repeat_1 = np.repeat(1, shape_mat[0])
    repeat_1 = cp.asarray(repeat_1)
    kernel_out = cp.stack((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                           I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                           I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                           repeat_1))
    return (cp.transpose(kernel_out))


def kernelP_native(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def im2double(im):
    """ Returns a double image [0,1] of the uint8 im [0,255]. """
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
