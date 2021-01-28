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


import numpy as np
import numpy.matlib
import cv2


class WBsRGB:
  def __init__(self, gamut_mapping=2, upgraded=0):
    if upgraded == 1:
      self.features = np.load('models/features+.npy')  # encoded features
      self.mappingFuncs = np.load('models/mappingFuncs+.npy')  # correct funcs
      self.encoderWeights = np.load('models/encoderWeights+.npy')  # PCA matrix
      self.encoderBias = np.load('models/encoderBias+.npy')  # PCA bias
      self.K = 75  # K value for NN searching
    else:
      self.features = np.load('models/features.npy')  # encoded features
      self.mappingFuncs = np.load('models/mappingFuncs.npy')  # correction funcs
      self.encoderWeights = np.load('models/encoderWeights.npy')  # PCA matrix
      self.encoderBias = np.load('models/encoderBias.npy')  # PCA bias
      self.K = 25  # K value for nearest neighbor searching

    self.sigma = 0.25  # fall-off factor for KNN blending
    self.h = 60  # histogram bin width
    # our results reported with gamut_mapping=2, however gamut_mapping=1
    # gives more compelling results with over-saturated examples.
    self.gamut_mapping = gamut_mapping  # options: 1 scaling, 2 clipping

  def encode(self, hist):
    """ Generates a compacted feature of a given RGB-uv histogram tensor."""
    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,
                              [histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - self.encoderBias.transpose(),
                     self.encoderWeights)
    return feature

  def rgb_uv_hist(self, I):
    """ Computes an RGB-uv histogram tensor. """
    sz = np.shape(I)  # get size of current image
    if sz[0] * sz[1] > 202500:  # resize if it is larger than 450*450
      factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
      newH = int(np.floor(sz[0] * factor))
      newW = int(np.floor(sz[1] * factor))
      I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
    II = I.reshape(int(I.size / 3), 3)  # n*3
    inds = np.where((II[:, 0] > 0) & (II[:, 1] > 0) & (II[:, 2] > 0))
    R = II[inds, 0]  # red channel
    G = II[inds, 1]  # green channel
    B = II[inds, 2]  # blue channel
    I_reshaped = np.concatenate((R, G, B), axis=0).transpose()
    eps = 6.4 / self.h
    A = np.arange(-3.2, 3.19, eps)  # dummy vector
    hist = np.zeros((A.size, A.size, 3))  # histogram will be stored here
    Iy = np.sqrt(I_reshaped[:, 0] ** 2 + I_reshaped[:, 1] ** 2 +
                 I_reshaped[:, 2] ** 2)  # intensity vector
    for i in range(3):  # for each histogram layer, do
      r = []  # excluded channels will be stored here
      for j in range(3):  # for each color channel do
        if j != i:
          r.append(j)
      Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
      Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
      diff_u = np.abs(np.matlib.repmat(Iu, np.size(A), 1).transpose() -
                      np.matlib.repmat(A, np.size(Iu), 1))
      diff_v = np.abs(np.matlib.repmat(Iv, np.size(A), 1).transpose() -
                      np.matlib.repmat(A, np.size(Iv), 1))
      diff_u[diff_u >= (eps / 2)] = 0
      diff_u[diff_u != 0] = 1
      diff_v[diff_v >= (eps / 2)] = 0
      diff_v[diff_v != 0] = 1

      # Matrix multiplication expression to compute eq. 4 in the main paper.
      # Iy * diff_u' (element-wise mult)
      temp = (np.matlib.repmat(Iy, np.size(A), 1) * diff_u.transpose())
      # initialize current histogram layer with Iy .* diff' * diff_v
      hist[:, :, i] = np.dot(temp, diff_v)
      # compute sum of hist for normalization
      norm_ = np.sum(hist[:, :, i], axis=None)
      hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
    return hist

  def correctImage(self, I):
    """ White balance a given image I. """
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    I = im2double(I)  # convert to double
    feature = self.encode(self.rgb_uv_hist(I))
    D_sq = np.einsum(
      'ij, ij ->i', self.features, self.features)[:, None] + np.einsum(
      'ij, ij ->i', feature, feature) - 2 * self.features.dot(feature.T)

    # get smallest K distances
    idH = D_sq.argpartition(self.K, axis=0)[:self.K]
    mappingFuncs = np.squeeze(self.mappingFuncs[idH, :])
    dH = np.sqrt(
      np.take_along_axis(D_sq, idH, axis=0))
    sorted_idx = dH.argsort(axis=0)  # get sorting indices
    idH = np.take_along_axis(idH, sorted_idx, axis=0)  # sort distance inds
    dH = np.take_along_axis(dH, sorted_idx, axis=0)  # sort distances
    weightsH = np.exp(-(np.power(dH, 2)) /
                      (2 * np.power(self.sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    mf = sum(np.matlib.repmat(weightsH, 1, 33) *
             mappingFuncs, 0)  # compute the mapping function
    mf = mf.reshape(11, 3, order="F")  # reshape it to be 9 * 3
    I_corr = self.colorCorrection(I, mf)  # apply it!
    return I_corr

  def colorCorrection(self, input, m):
    """ Applies a mapping function m to a given input image. """
    sz = np.shape(input)  # get size of input image
    I_reshaped = np.reshape(input, (int(input.size / 3), 3), order="F")
    kernel_out = kernelP(I_reshaped)
    out = np.dot(kernel_out, m)
    if self.gamut_mapping == 1:
      # scaling based on input image energy
      out = normScaling(I_reshaped, out)
    elif self.gamut_mapping == 2:
      # clip out-of-gamut pixels
      out = outOfGamutClipping(out)
    else:
      raise Exception('Wrong gamut_mapping value')
    # reshape output image back to the original image shape
    out = out.reshape(sz[0], sz[1], sz[2], order="F")
    out = cv2.cvtColor(out.astype('float32'), cv2.COLOR_RGB2BGR)
    return out


def normScaling(I, I_corr):
  """ Scales each pixel based on original image energy. """
  norm_I_corr = np.sqrt(np.sum(np.power(I_corr, 2), 1))
  inds = norm_I_corr != 0
  norm_I_corr = norm_I_corr[inds]
  norm_I = np.sqrt(np.sum(np.power(I[inds, :], 2), 1))
  I_corr[inds, :] = I_corr[inds, :] / np.tile(
    norm_I_corr[:, np.newaxis], 3) * np.tile(norm_I[:, np.newaxis], 3)
  return I_corr


def kernelP(I):
  """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric
          characterization based on polynomial modeling." Color Research &
          Application, 2001. """
  return (np.transpose(
    (I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
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
