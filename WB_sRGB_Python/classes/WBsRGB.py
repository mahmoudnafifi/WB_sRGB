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
    # precompute the norm of all features for later use
    self.features_norm = np.einsum('ij, ij ->i', self.features,
                                   self.features)[:, None]


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
    I_reshaped = I.reshape(-1,3).T.copy() # reshaped and transposed
    I_reshaped = I_reshaped[:,(I_reshaped>0).all(0)].copy()
    hist = np.zeros((self.h, self.h, 3), dtype=np.float32)  # histogram will be stored here
    Iy = np.linalg.norm(I_reshaped, axis=0)  # intensity vector
    I_reshaped_log = np.log(I_reshaped)
    for i in range(3):  # for each histogram layer, do
      r = [j for j in range(3) if i!=j]  # excluded channels
      Iu = I_reshaped_log[i] - I_reshaped_log[r[1]]
      Iv = I_reshaped_log[i] - I_reshaped_log[r[0]]
      hist[:, :, i] = hist2d(Iv, Iu, Iy, (-3.2, 3.2), self.h)
      norm_ = hist[:, :, i].sum()
      hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
    return hist

  def correctImage(self, I):
    """ White balance a given image I. """
    I = I[..., ::-1]  # convert from BGR to RGB
    I = im2double(I)  # convert to double
    # Convert I to float32 may speed up the process.
    feature = self.encode(self.rgb_uv_hist(I))
    # Do
    # ```python
    # feature_diff = self.features - feature
    # D_sq = np.einsum('ij,ij->i', feature_diff, feature_diff)[:, None]
    # ```
    D_sq = self.features_norm + np.einsum(
      'ij, ij ->i', feature, feature) - 2 * self.features.dot(feature.T)
    # get smallest K distances
    idH = D_sq.argpartition(self.K, axis=0)[:self.K]
    mappingFuncs = np.squeeze(self.mappingFuncs[idH, :])
    dH = np.sqrt(np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) /
                      (2 * np.power(self.sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    mf = weightsH.T.dot(mappingFuncs)  # compute the mapping function
    mf = mf.reshape(11, 3, order="F")  # reshape it to be 9 * 3
    I_corr = self.colorCorrection(I, mf)  # apply it!
    return I_corr

  def colorCorrection(self, input, m):
    """ Applies a mapping function m to a given input image. """
    sz = np.shape(input)  # get size of input image
    I_reshaped = np.reshape(input, (-1, 3)).T  # transposed for speed
    kernel_out = kernelP(I_reshaped)
    out = m.T.dot(kernel_out).T
    if self.gamut_mapping == 1:
      # scaling based on input image energy
      out = normScaling(I_reshaped, out)
    elif self.gamut_mapping == 2:
      # clip out-of-gamut pixels
      out = outOfGamutClipping(out)
    else:
      raise Exception('Wrong gamut_mapping value')
    # reshape output image back to the original image shape
    out = out.reshape(sz)
    out = out[..., ::-1]  # convert from BGR to RGB
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


def kernelP(rgb):
  """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric
          characterization based on polynomial modeling." Color Research &
          Application, 2001. """
  r, g, b = (rgb[0], rgb[1], rgb[2])
  out = np.empty((11, rgb.shape[1]), dtype=rgb.dtype)
  out[:3, :] = rgb
  out[3, :] = r*g
  out[4, :] = r*b
  out[5, :] = g*b
  out[6:9, :] = rgb*rgb
  out[9, :] = r*g*b
  out[10, :] = np.ones_like(r)
  return out

def outOfGamutClipping(I):
  """ Clips out-of-gamut pixels. """
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I


def im2double(im):
  """ Returns a double image [0,1] of the uint8 im [0,255]. """
  return cv2.normalize(im, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)


def hist2d(x, y, weight, limits, bins):
    """ Computes a 2D histogram of values using only numpy"""
    eps = (limits[1]-limits[0]) / bins
    lower_lim = limits[0]-eps/2
    y = np.floor((y-lower_lim)/eps).astype(np.int16)
    x = np.floor((x-lower_lim)/eps).astype(np.int16)
    valid = (0<=x)*(x<bins)*(0<=y)*(y<bins)
    hist = np.bincount(y[valid]*bins+x[valid], weight[valid], bins**2)
    return hist.reshape(bins, bins)
