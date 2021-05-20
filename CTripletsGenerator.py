import tensorflow as tf
import numpy as np
import math
import random

class CTripletsGenerator(tf.keras.utils.Sequence):
  def __init__(self, X, Y, batch_size):
    super().__init__()
    self._X = X
    self._Y = np.squeeze(Y)
    self._batchSize = batch_size
    
    self._positives = {}
    self._labels = list(range(1 + self._Y.max()))
    for label in self._labels:
      self._positives[label] = np.where(label == self._Y)[0]
    
    return
  
  def on_epoch_end(self):
    return
  
  def _negatives(self, label):
    neglabel = label
    while neglabel == label:
      neglabel = random.choice(self._labels)
    return self._positives[neglabel]
  
  def _triplet(self, anchorIndex):
    X, Y = self._X, self._Y
    label = Y[anchorIndex]
    return(
      X[anchorIndex],
      X[np.random.choice(self._positives[label])],
      X[np.random.choice(self._negatives(label))],
    )
  
  def __getitem__(self, batchIndex):
    current = (batchIndex * self._batchSize) % len(self._X)
    shape = (self._batchSize, ) + self._X.shape[1:]
    x_anchors = np.zeros(shape)
    x_positives = np.zeros(shape)
    x_negatives = np.zeros(shape)
    
    for i in range(len(x_anchors)):
      anchor, pos, neg = self._triplet(current)
      current = (current + 1) % len(self._X)
      
      x_anchors[i] = anchor
      x_positives[i] = pos
      x_negatives[i] = neg
        
    return(x_anchors, x_positives, x_negatives)
  
  def __len__(self):
      return math.ceil(len(self._X) / self._batchSize)