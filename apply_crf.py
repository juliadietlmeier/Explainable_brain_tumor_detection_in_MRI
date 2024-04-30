#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy 
import pydensecrf.densecrf as dcrf

d = dcrf.DenseCRF2D(640, 480, 5)  # width, height, nlabels
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import cv2
from skimage import color, data, filters, graph, measure, morphology
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat  # noqa
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.set_axis_off()
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.set_axis_off()

def apply_crf(im, pred):
  im = numpy.ascontiguousarray(im)
  #if im.shape[:2] != pred.shape[:2]:
  #  im = imresize(im, pred.shape[:2])

  #pred = numpy.ascontiguousarray(pred.swapaxes(0, 2).swapaxes(1, 2))

  d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], 2)  # width, height, nlabels
  
  unaries = unary_from_softmax(pred)
  d.setUnaryEnergy(unaries)

  d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
  d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=im, compat=1.40326787165)
  processed = d.inference(12)
  res = numpy.argmax(processed, axis=0).reshape(im.shape[:2])

  return res 
