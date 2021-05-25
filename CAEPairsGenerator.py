import tensorflow as tf
import numpy as np
import math
import random

class CAEPairsGenerator(tf.keras.utils.Sequence):
  def __init__(self, X, batch_size, noise):
    super().__init__()
    self._X = X
    self._batchSize = batch_size
    self._noise = noise
    return
  
  def __getitem__(self, batchIndex):
    current = (batchIndex * self._batchSize) % len(self._X)
    samples = self._X[current:current+self._batchSize]
    if 0 < self._noise:
      corrupted = samples.copy()
      for i, data in enumerate(corrupted):
        H, W = data.shape[:2]
        ptsN = int(self._noise * H * W)
        noiseX = np.random.choice(np.arange(W), size=(ptsN,), replace=True)
        noiseY = np.random.choice(np.arange(H), size=(ptsN,), replace=True)
        corrupted[i, noiseX, noiseY] = 1.0 - corrupted[i, noiseX, noiseY]
      return(corrupted, samples)
    
    return(samples, samples)
  
  def __len__(self):
      return math.ceil(len(self._X) / self._batchSize)