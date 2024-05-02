#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import numpy as np

def computeCRF(im, pred):
  im = np.ascontiguousarray(im)

  d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], 2)  # width, height, nlabels
  
  unaries = unary_from_softmax(pred)
  d.setUnaryEnergy(unaries)

  d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
  d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=im, compat=1.40326787165)
  processed = d.inference(12)
  res = np.argmax(processed, axis=0).reshape(im.shape[:2])

  return res 
